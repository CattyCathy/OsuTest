"""Compares a trained detector with the beatmaps it is being judged against.

Training numbers say how well the model reproduces its own labels on held-out tracks. They do not say whether the beats
it reports are the ones a player would tap, and the two are different questions here: the labels come from a map's
timing points, a dense map snaps its objects to a quarter of the beat as readily as to the beat, and a model that
learns the labels perfectly can still report twice as many beats as the music has.

What is measured, against a map's own timing points and hit objects:

  coverage   the share of the map's beats with a reported beat within 60ms of it
  error      the middle distance from a reported beat to the nearest map beat
  rate       reported beats a second against the map's own beats a second, which is the octave question stated as a
             number: 1.00 is the level, 2.00 is one level too fine and 0.50 is one too coarse

Run it on the same corpus the exporter used, so the audio it scores against is the audio the model was trained on.
Held-out tracks are reported separately: a model can look right on its training set while being wrong about the level.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import (  # noqa: E402
    FPS, read_track, log_mel, load_manifest, pick_peaks, build_model,
)

import torch  # noqa: E402


def map_beats(osz: Path, until_ms: float) -> np.ndarray:
    """A map's beats from its uninherited timing points, in milliseconds."""
    import zipfile
    import io
    import struct

    with zipfile.ZipFile(osz) as archive:
        name = next((n for n in archive.namelist() if n.lower().endswith(".osu")), None)

        if name is None:
            return np.empty(0)

        text = archive.read(name).decode("utf-8", errors="replace")

    points = []
    in_timing = False

    for raw in text.splitlines():
        line = raw.strip()

        if line.startswith("["):
            in_timing = line.lower() == "[timingpoints]"
            continue

        if not in_timing or not line:
            continue

        fields = line.split(",")

        if len(fields) < 2:
            continue

        try:
            time = float(fields[0])
            length = float(fields[1])
        except ValueError:
            continue

        # Negative lengths are inherited points: slider and scroll speed, and nothing to do with the beat.
        if length > 0:
            points.append((time, length))

    points.sort()

    if not points:
        return np.empty(0)

    beats = []

    for i, (time, length) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else max(until_ms, time)

        t = time
        while t < end:
            beats.append(t)
            t += length

    return np.array(sorted(beats))


def reported_beats(probability: np.ndarray, threshold: float, minimum_gap: int = 0, method: str = "peaks",
                   minimum_period: int = 15, maximum_period: int = 100, radius: int = 3) -> np.ndarray:
    if method == "pulse":
        peaks = pulse_beats(probability, minimum_period, maximum_period)
    elif minimum_gap > 0:
        peaks = pick_peaks_separated(probability, threshold, radius, minimum_gap)
    else:
        peaks = pick_peaks(probability, radius=radius, threshold=threshold)

    return peaks / FPS * 1000.0


def nearest_distances(reported: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reported.size == 0 or reference.size == 0:
        return np.empty(0)

    order = np.argsort(reference)
    reference = reference[order]

    index = np.searchsorted(reference, reported)
    index = np.clip(index, 1, len(reference) - 1)

    before = reference[index - 1]
    after = reference[np.clip(index, 0, len(reference) - 1)]

    return np.minimum(np.abs(reported - before), np.abs(reported - after))


def pick_peaks_separated(probability: np.ndarray, threshold: float, radius: int, minimum_gap: int) -> np.ndarray:
    """Peaks, with a minimum separation between them, kept strongest first.

    A peak picker on its own has no notion of how far apart beats are, so on music with an onset on every eighth it
    reports every eighth. The separation is what turns a per-frame detector into a pulse, and sweeping it is how the
    level the model is actually reporting gets separated from the level the map is written at.
    """
    candidates = [(float(probability[i]), int(i)) for i in pick_peaks(probability, radius=radius, threshold=threshold)]
    candidates.sort(reverse=True)

    chosen: list[int] = []

    for _, index in candidates:
        if all(abs(index - other) >= minimum_gap for other in chosen):
            chosen.append(index)

    return np.array(sorted(chosen), dtype=np.int64)


def constrained_beats(probability: np.ndarray, minimum_period: int, maximum_period: int,
                      lookahead: int = 3, period_penalty: float = 0.02) -> np.ndarray:
    """Chases a pulse through the detector's own curve, letting its period change as the music does.

    A single pulse over a whole track is the wrong instrument for the material this exists for: a track whose tempo
    changes has no single period, and forcing one either reports the average of two tempi, which is neither, or the
    level of whichever passage is longest.

    So the chase is local. From a beat, the next one is whichever frame up to a lookahead away scores best, with a
    penalty for changing the period so that a steady pulse is preferred to a wandering one. The penalty is small
    because the point is to follow a real tempo change and merely to stop the pulse from jumping to a louder onset a
    third of a beat away.
    """
    frames = len(probability)

    if frames < minimum_period * 3:
        return np.empty(0, dtype=np.int64)

    # Start where the model is most confident, which is the only frame with no period behind it to agree with.
    first = int(np.argmax(probability))
    beats = [first]

    period = float(minimum_period + maximum_period) / 2
    current = first

    while True:
        best_score = -np.inf
        best_next = -1
        best_period = period

        low = max(minimum_period, int(period) - lookahead)
        high = min(maximum_period, int(period) + lookahead)

        for candidate_period in range(low, high + 1):
            nxt = current + candidate_period

            if nxt >= frames:
                break

            # The value of the frame the beat would land on, less what it costs to have moved the period there.
            score = float(probability[nxt]) - period_penalty * abs(candidate_period - period)

            if score > best_score:
                best_score = score
                best_next = nxt
                best_period = float(candidate_period)

        if best_next < 0:
            break

        # A period that only ever grows or only ever shrinks would walk away from the music, so the search is bounded
        # to the range it was given; inside it the period is free to move a frame at a time.
        period = best_period
        beats.append(best_next)
        current = best_next

        if len(beats) > 1_000_000:
            break

    return np.array(beats, dtype=np.int64)


def constrained_grid(probability: np.ndarray, minimum_period: int, maximum_period: int) -> np.ndarray:
    """The beat grid the detector's own curve best supports, searched over period and phase.

    The detector is a per-frame classifier and has no notion of how far apart beats are, so it fires on every onset the
    music has - on one track at 4.00 times the map's beat count, at the sixteenth note. What it does know is how much
    it believes in each frame, and that is enough to choose a grid: a pulse at a period and a phase, scored by the
    model's own probability along it, and the best-scoring pulse is the level the model is reporting.

    This is the same combinatorial problem the peak picker avoids by not asking it, and solving it rather than picking
    peaks is what makes the level a decision instead of an accident of how many onsets the music happens to have.
    """
    frames = len(probability)

    if frames < minimum_period * 4:
        return np.empty(0, dtype=np.int64)

    # Beat positions have to be whole frames, so a period is scored on its own grid rather than by interpolation.
    best_score = -np.inf
    best_period = 0
    best_offset = 0

    for period in range(minimum_period, maximum_period + 1):
        for offset in range(period):
            positions = np.arange(offset, frames, period)

            if positions.size < 4:
                continue

            # The mean rather than the sum, so a shorter period which simply has more positions is not favoured by
            # having more chances to collect.
            score = float(probability[positions].mean())

            if score > best_score:
                best_score = score
                best_period = period
                best_offset = offset

    if best_period == 0:
        return np.empty(0, dtype=np.int64)

    return np.arange(best_offset, frames, best_period)


def pulse_beats(probability: np.ndarray, minimum_period: int, maximum_period: int,
                window: int = 1500, overhead: float = 0.35) -> np.ndarray:
    """A pulse chosen per window by scoring every period and phase, rather than chased frame by frame.

    Chasing the pulse forward was tried first and is much worse: the next beat is whichever frame nearby scores best,
    and over a few hundred beats that walk drifts into whatever passage happens to be loudest, reporting a rate of 0.23
    of the map's beats with a coverage of six per cent. Enumerating instead is bounded, has no state to drift, and
    cannot do worse than the best pulse in the window.

    The period range is wide - a fifth of a second to two seconds, which is 300 to 30 BPM - because a map's timing
    points are a snap resolution and the level the music is at can be an octave either side of them. Choosing between
    those octaves is the whole point, so the search must be free to reach them.

    A window is scored on the mean probability along its pulse, and overlapping windows vote by taking each one's beats
    in its own middle, so a beat near an edge is decided by a window that contains it properly.
    """
    frames = len(probability)
    beats: set[int] = set()

    if frames < minimum_period * 4:
        return np.empty(0, dtype=np.int64)

    step = window // 2

    for start in range(0, frames, step):
        end = min(frames, start + window)

        if end - start < minimum_period * 4:
            break

        best_score = -np.inf
        best_positions = None

        # Every period and every phase within the window, but scored by striding rather than by testing each position
        # against the phase. The direct way costs a pass over the window for every (period, phase) pair - about a
        # quarter of a million operations for one window - where a stride is a view and the mean is over the elements
        # it selects.
        segment = probability[start:end]

        for period in range(minimum_period, maximum_period + 1):
            if segment.size // period < 4:
                continue

            for phase in range(period):
                positions = np.arange(phase, segment.size, period)

                if positions.size < 4:
                    continue

                score = float(segment[positions].mean())

                if score > best_score:
                    best_score = score
                    best_positions = positions + start

        if best_positions is None:
            continue

        # Only the middle half of a window's beats are kept, so neighbouring windows agree about the beats they share
        # and a beat decided at one window's edge is decided again by the window that holds it properly.
        middle = (end - start) // 4

        for position in best_positions:
            if start + middle <= position < end - middle:
                beats.add(int(position))

    return np.array(sorted(beats), dtype=np.int64)


def gap_report(dataset: Path, corpus: Path, model_path: Path, track_id: str, limit: int = 400) -> None:
    """The distribution of reported gaps against the distribution of the map's own gaps."""
    tracks = {t.id: t for t in load_manifest(dataset)}
    track = tracks.get(track_id)

    if track is None:
        print(f"no track {track_id} in the manifest")
        return

    archive = next((p for p in corpus.glob("*.osz") if p.name.split(" ")[0] == track_id), None)

    if archive is None:
        print(f"no archive for {track_id}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    samples, labels = read_track(track)
    mels = log_mel(samples)

    probabilities = []
    segment = 1024
    step = segment - 64

    for start in range(0, mels.shape[0], step):
        window = mels[start:start + segment]

        if window.shape[0] < 64:
            break

        with torch.no_grad():
            x = torch.from_numpy(window.T.astype(np.float32)).unsqueeze(0).to(device)
            probabilities.append(torch.sigmoid(model(x))[0].cpu().numpy())

    probability = np.concatenate(probabilities)
    reported = pick_peaks(probability, radius=3, threshold=0.5) / FPS * 1000.0

    beats = map_beats(archive, len(labels) / FPS * 1000)

    def histogram(times: np.ndarray, name: str) -> None:
        gaps = np.diff(np.sort(times))
        gaps = gaps[gaps > 0]

        if gaps.size == 0:
            print(f"{name}: no gaps")
            return

        buckets: dict[int, int] = {}

        for gap in gaps:
            buckets[int(gap // 40) * 40] = buckets.get(int(gap // 40) * 40, 0) + 1

        print(f"{name}: {len(times)} beats, median gap {np.median(gaps):.0f}ms")

        for bucket in sorted(buckets)[:limit]:
            print(f"   {bucket:5}ms {'#' * min(60, buckets[bucket])} {buckets[bucket]}")

    print(f"track {track_id}")
    print("")
    histogram(beats, "the map's own beats")
    print("")
    histogram(reported, "the detector's beats")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--segment", type=int, default=1024, help="frames per inference window")
    parser.add_argument("--limit", type=int, default=0, help="score at most this many tracks")
    parser.add_argument("--min-gap", type=int, default=0, help="minimum frames between reported beats")
    parser.add_argument("--gaps", type=str, default=None, help="print gap distributions for this track id")
    parser.add_argument("--method", choices=("peaks", "pulse"), default="peaks",
                        help="how beats are chosen from the detector's curve")
    parser.add_argument("--min-period", type=int, default=15, help="shortest beat period in frames for --method pulse")
    parser.add_argument("--max-period", type=int, default=100, help="longest beat period in frames for --method pulse")
    args = parser.parse_args()

    if args.gaps:
        gap_report(args.dataset, args.corpus, args.model, args.gaps)
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    tracks = load_manifest(args.dataset)
    by_id = {t.id: t for t in tracks}

    # The exported id is the beatmapset id, so the set archive is found by that prefix.
    archives = {}

    for osz in args.corpus.glob("*.osz"):
        archives[osz.name.split(" ")[0]] = osz

    shared = sorted(set(by_id) & set(archives))

    if args.limit:
        shared = shared[:args.limit]

    print(f"scoring {len(shared)} tracks with a threshold of {args.threshold}")

    rows = []

    for track_id in shared:
        track = by_id[track_id]

        try:
            samples, labels = read_track(track)
            mels = log_mel(samples)
            beats = map_beats(archives[track_id], len(labels) / FPS * 1000)

            if beats.size < 16:
                continue

            # Windows with a little overlap, because a beat on a window edge would otherwise be cut.
            probabilities = []
            step = args.segment - 64

            for start in range(0, mels.shape[0], step):
                window = mels[start:start + args.segment]

                if window.shape[0] < 64:
                    break

                with torch.no_grad():
                    x = torch.from_numpy(window.T.astype(np.float32)).unsqueeze(0).to(device)
                    probabilities.append(torch.sigmoid(model(x))[0].cpu().numpy())

            probability = np.concatenate(probabilities) if probabilities else np.empty(0)

            if probability.size < 16:
                continue

            reported = reported_beats(probability, args.threshold, args.min_gap, args.method,
                                      args.min_period, args.max_period)

            if reported.size < 4:
                continue

            distance = nearest_distances(reported, beats)

            # Coverage is counted from the map's side: the share of its beats that something was reported near.
            covered = 0

            for beat in beats:
                if reported.size and np.abs(reported - beat).min() <= 60:
                    covered += 1

            rows.append({
                "id": track_id,
                "reported": reported.size,
                "map": beats.size,
                "rate": (reported.size / (len(labels) / FPS)) / (beats.size / (len(labels) / FPS)),
                "median": float(np.median(distance)) if distance.size else 0.0,
                "p90": float(np.percentile(distance, 90)) if distance.size else 0.0,
                "coverage": 100.0 * covered / beats.size,
            })
        except Exception as error:  # noqa: BLE001
            print(f"  {track_id}: {type(error).__name__}: {error}")

    if not rows:
        print("nothing scored")
        return 1

    reported = np.array([r["reported"] for r in rows])
    rates = np.array([r["rate"] for r in rows])
    medians = np.array([r["median"] for r in rows])
    coverages = np.array([r["coverage"] for r in rows])

    print("")
    print(f"  {'track':>10} {'reported':>9} {'map':>7} {'rate':>6} {'median':>8} {'p90':>7} {'coverage':>9}")

    for row in sorted(rows, key=lambda r: -r["rate"])[:24]:
        print(f"  {row['id']:>10} {row['reported']:>9} {row['map']:>7} {row['rate']:>6.2f} "
              f"{row['median']:>7.0f}ms {row['p90']:>6.0f}ms {row['coverage']:>8.1f}%")

    print("")
    print(f"  tracks scored        {len(rows)}")
    print(f"  rate  median {np.median(rates):.2f}  mean {rates.mean():.2f}  "
          f"within 0.75-1.33 of the map: {int(((rates > 0.75) & (rates < 1.33)).sum())}/{len(rates)}")
    print(f"  error median {np.median(medians):.0f}ms  p90 {np.median(np.array([r['p90'] for r in rows])):.0f}ms")
    print(f"  coverage     median {np.median(coverages):.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
