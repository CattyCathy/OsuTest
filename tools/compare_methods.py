"""Scores the current analysis and the trained detector on the same tracks, the same way.

There is no point reporting a detector's numbers without the numbers of the thing it would replace. The current
analysis is peak picking over Beat This!'s activation, optionally followed by a search for a steady tempo; this script
runs both, and the detector, over the same audio and against the same reference, so the comparison is like for like.

The reference is a beatmap's uninherited timing points, which is what the exporter's labels came from and what every
measurement in this project has used. Two things about it are worth restating because they decide how the numbers
read:

  It is a snap grid, not a statement of the musical beat. A dense map snaps objects to a quarter of the beat, so a
  reading at twice the grid's rate is not necessarily wrong about the music - but it is not what the map says, and the
  map is the only reference available.

  Coverage alone is misleading when a reading reports several times as many beats as the grid has, because a large
  enough scatter of beats will land near everything. Precision is reported beside it for that reason, as the share of
  the reading's own beats that are near a grid beat.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, read_track, log_mel, load_manifest, build_model  # noqa: E402
from score_detector import map_beats, pulse_beats, pick_peaks  # noqa: E402

import torch  # noqa: E402


def beat_this_activation(samples: np.ndarray, model_path: Path, threads: int = 2) -> np.ndarray:
    """The activation ParaTactus currently reads.

    Not available from Python: the frontend and the model session live in the C# side, and reimplementing the model's
    inference here would be measuring a different thing. The current analysis is therefore left out of this comparison
    and is measured by the C# probes instead - which is where every number this project has for it came from. What is
    compared here is the two ways of reading the trained detector, because those are the ones in question.
    """
    raise NotImplementedError


def score(reported_ms: np.ndarray, reference_ms: np.ndarray, seconds: float) -> dict:
    if reported_ms.size == 0 or reference_ms.size == 0:
        return {"rate": 0.0, "coverage": 0.0, "precision": 0.0, "median": 0.0, "count": reported_ms.size}

    order = np.argsort(reference_ms)
    reference = reference_ms[order]

    index = np.searchsorted(reference, reported_ms)
    index = np.clip(index, 1, len(reference) - 1)

    nearest = np.minimum(np.abs(reported_ms - reference[index - 1]),
                         np.abs(reported_ms - reference[np.clip(index, 0, len(reference) - 1)]))

    covered = 0

    for beat in reference:
        position = np.searchsorted(reported_ms, beat)
        low = max(0, position - 2)
        high = min(len(reported_ms), position + 2)

        if high > low and np.abs(reported_ms[low:high] - beat).min() <= 60:
            covered += 1

    return {
        "rate": (reported_ms.size / seconds) / (reference.size / seconds),
        "coverage": 100.0 * covered / reference.size,
        "precision": 100.0 * (nearest <= 60).sum() / reported_ms.size,
        "median": float(np.median(nearest)),
        "count": reported_ms.size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-period", type=int, default=12)
    parser.add_argument("--max-period", type=int, default=90)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    tracks = load_manifest(args.dataset)
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}
    shared = sorted(set(t.id for t in tracks) & set(archives))[:args.limit]

    print(f"scoring {len(shared)} tracks")
    print("")
    print(f"  {'track':>10} {'map':>6} | {'detector: rate':>14} {'cov':>6} {'prec':>6} {'med':>7} | "
          f"{'pulse: rate':>11} {'cov':>6} {'prec':>6} {'med':>7}")

    detector_rows = []
    pulse_rows = []

    for track_id in shared:
        track = next(t for t in tracks if t.id == track_id)

        samples, labels = read_track(track)
        seconds = len(labels) / FPS
        mels = log_mel(samples)

        reference = map_beats(archives[track_id], seconds * 1000)

        if reference.size < 16:
            continue

        probabilities = []
        segment, step = 1024, 960

        for start in range(0, mels.shape[0], step):
            window = mels[start:start + segment]

            if window.shape[0] < 64:
                break

            with torch.no_grad():
                x = torch.from_numpy(window.T.astype(np.float32)).unsqueeze(0).to(device)
                probabilities.append(torch.sigmoid(model(x))[0].cpu().numpy())

        if not probabilities:
            continue

        probability = np.concatenate(probabilities)

        peaks = pick_peaks(probability, radius=3, threshold=0.5) / FPS * 1000.0
        pulse = pulse_beats(probability, args.min_period, args.max_period) / FPS * 1000.0

        detector = score(peaks, reference, seconds)
        pulsed = score(pulse, reference, seconds)

        detector_rows.append(detector)
        pulse_rows.append(pulsed)

        print(f"  {track_id:>10} {reference.size:>6} | {detector['rate']:>14.2f} {detector['coverage']:>5.0f}% "
              f"{detector['precision']:>5.0f}% {detector['median']:>6.0f}ms | {pulsed['rate']:>11.2f} "
              f"{pulsed['coverage']:>5.0f}% {pulsed['precision']:>5.0f}% {pulsed['median']:>6.0f}ms")

    for name, rows in (("detector peaks", detector_rows), ("tempo pulse", pulse_rows)):
        if not rows:
            continue

        rates = np.array([r["rate"] for r in rows])
        coverage = np.array([r["coverage"] for r in rows])
        precision = np.array([r["precision"] for r in rows])
        median = np.array([r["median"] for r in rows])

        print("")
        print(f"  {name}:")
        print(f"    rate      median {np.median(rates):.2f}   within 0.75-1.33 of the map: "
              f"{int(((rates > 0.75) & (rates < 1.33)).sum())}/{len(rates)}")
        print(f"    coverage  median {np.median(coverage):.0f}%")
        print(f"    precision median {np.median(precision):.0f}%   (share of its own beats near a map beat)")
        print(f"    error     median {np.median(median):.0f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
