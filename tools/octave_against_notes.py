"""Which metrical level the music is actually at, using the map's hit objects as the evidence.

The declared timing point is not the music's pulse and on this corpus it is often not even close. Measured on four
tracks, the gap between consecutive hit objects is a quarter of the declared beat - the map's timing point is a scroll
speed and a snap grid, and the notes are placed at a quarter of it. On another the objects are two and a third times
the declared beat. So "the map's level" is not a reference a detector can be judged against, and a reading that
disagrees with it may be the one that agrees with the music.

Where the notes are is a better reference, because a mapper places them on events. If the detector's beats sit on notes
and the notes also fall between its beats, then the detector is at half the level the notes are at. If the notes sit
only on its beats and not between them, it is at their level or coarser and there is nothing between its beats to
catch. That is a comparison of two distances and it needs no opinion about tempo at all:

    distance from a detector beat to the nearest note
    distance from halfway between two detector beats to the nearest note

When the second is about as small as the first, the notes are denser than the detector's beats and it is reading an
octave low. When the second is much larger, the detector's beats are as dense as the notes and its level is the
music's.

The three levels are then scored the same way - the reading as it is, the same reading thinned by half, and the same
reading divided in two - so that the comparison is between readings and not against a declared number.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402
from refine_choice import centroid  # noqa: E402


def hit_objects(path: Path) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".osu"))
        text = archive.read(name).decode("utf-8", "replace")

    inside = False
    out: list[float] = []

    for raw in text.splitlines():
        line = raw.strip()

        if line.startswith("["):
            inside = line.lower() == "[hitobjects]"
            continue

        if not inside or not line:
            continue

        fields = line.split(",")

        if len(fields) >= 3:
            try:
                out.append(float(fields[2]))
            except ValueError:
                pass

    return np.array(sorted(out))


def distances(reported: np.ndarray, notes: np.ndarray) -> np.ndarray:
    """Distance from each reported time to the closest note."""
    if reported.size == 0 or notes.size == 0:
        return np.full(reported.shape, np.inf)

    index = np.clip(np.searchsorted(notes, reported), 1, notes.size - 1)

    return np.minimum(np.abs(reported - notes[index - 1]), np.abs(reported - notes[index]))


def score(beats: np.ndarray, notes: np.ndarray) -> tuple[float, float]:
    """The middle distance from a beat to a note, and from halfway between beats to a note."""
    if beats.size < 4:
        return float("nan"), float("nan")

    at = distances(beats, notes)
    between = distances((beats[:-1] + beats[1:]) / 2, notes)

    return float(np.median(at)), float(np.median(between))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    rows = list(csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t"))
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}

    if args.limit:
        rows = rows[:args.limit]

    print("Is the detector at the level the notes are at? 'on' is the middle distance from a beat to a note,")
    print("'between' the same from halfway between two beats. When 'between' is as small as 'on', the notes are")
    print("denser than the beats and the reading is an octave low.")
    print("")
    print(f"  {'track':>9} {'notes':>6} {'beats':>6} | {'on':>7} {'between':>8} {'ratio':>6} | "
          f"{'thin/2 on':>10} {'thin/2 btw':>11} | {'double on':>10} {'double btw':>11} | level")

    verdicts = {"as it is": 0, "halved": 0, "doubled": 0, "unclear": 0}

    for row in rows:
        track_id = row["id"]
        cached = args.cache / f"{track_id}.npy"
        archive = archives.get(track_id)

        if not cached.exists() or archive is None:
            continue

        notes = hit_objects(archive)

        if notes.size < 32:
            continue

        mels = np.load(cached).astype(np.float32)
        probability, period = infer(model, mels, device)
        period = smooth(period, 1.0)
        frames = peaks(probability, period, 0.6, 0.5)
        beats = centroid(probability, frames, 2)

        if beats.size < 32:
            continue

        on, between = score(beats, notes)

        # Thinned to every other beat and to every fourth, which are the two coarser readings, and divided in two,
        # which is the finer one. Each is scored the same way so the comparison is between readings.
        halved = beats[::2]
        doubled = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))

        halved_on, halved_between = score(halved, notes)
        doubled_on, doubled_between = score(doubled, notes)

        ratio = between / on if on > 0 else float("inf")

        # The reading whose own beats land on notes while its midpoints do not is the one at the notes' level. A
        # midpoint distance close to the beat distance means the notes sit between the beats as much as on them.
        candidates = {
            "as it is": between / on if on > 0 else float("inf"),
            "halved": halved_between / halved_on if halved_on > 0 else float("inf"),
            "doubled": doubled_between / doubled_on if doubled_on > 0 else float("inf"),
        }

        best = max(candidates, key=lambda k: candidates[k])

        if candidates[best] < 1.5:
            verdicts["unclear"] += 1
        else:
            verdicts[best] += 1

        print(f"  {track_id:>9} {notes.size:>6} {beats.size:>6} | {on:>6.1f}ms {between:>7.1f}ms {ratio:>6.2f} | "
              f"{halved_on:>9.1f}ms {halved_between:>10.1f}ms | {doubled_on:>9.1f}ms {doubled_between:>10.1f}ms | "
              f"{best}" + ("  <- the notes are between its beats" if best == "doubled" else ""))

    print("")
    print("which reading puts its beats on the notes and not its midpoints:")

    for name, count in verdicts.items():
        print(f"  {name:>10}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
