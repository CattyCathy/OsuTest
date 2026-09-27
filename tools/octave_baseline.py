"""How much of a track the detector reads at the map's own metrical level, measured without the metric that misled.

An earlier attempt at this compared the beat curve's strength on a grid against its strength halfway between the grid's
points, and concluded that most sections were an octave out. That conclusion was wrong and the reason is worth keeping:
halfway between the points of a half-beat grid are the map's own beats, so the finer grid is scored on the coarser
grid's events and always wins. The metric was biased towards finer grids by construction and would have said "half the
beat" on a track with no octave error at all, which it did.

This measures it the plain way instead. The track is cut into runs of beats whose local spacing agrees, the map's
declared beat length is read at the middle of each run, and the run is counted as being at the map's level, an octave
above it, an octave below, or neither. Weighted by duration rather than by run count, because the question is how much
of the music a player hears at the wrong level and not how many boundaries a rule drew.
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
from diagnose_level import uninherited  # noqa: E402


def local_gap(beats: np.ndarray, width: int = 16) -> np.ndarray:
    """The spacing at each beat, as the middle of the gaps around it."""
    gaps = np.diff(beats)
    out = np.zeros(beats.size)

    for i in range(beats.size):
        low = max(0, i - width)
        high = min(gaps.size, i + width)

        out[i] = float(np.median(gaps[low:high])) if high > low else np.nan

    return out


def map_length_at(points: list[tuple[float, float]], at: float) -> float:
    """The beat length the map declares at a time, or zero if it declares none there."""
    best = 0.0

    for time, length in points:
        if time <= at:
            best = length
        else:
            break

    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--limit", type=int, default=0, help="0 for every track in the manifest")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    rows = list(csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t"))
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}

    if args.limit:
        rows = rows[:args.limit]

    # Time spent at each relationship between the detector's spacing and the map's beat length.
    buckets = {"at the map's level": 0.0, "an octave above": 0.0, "an octave below": 0.0,
               "other": 0.0, "unmeasured": 0.0}
    overall = {k: 0.0 for k in buckets}
    per_track = []

    for row in rows:
        track_id = row["id"]
        cached = args.cache / f"{track_id}.npy"
        archive = archives.get(track_id)

        if not cached.exists() or archive is None:
            continue

        mels = np.load(cached).astype(np.float32)
        probability, period = infer(model, mels, device)
        period = smooth(period, 1.0)
        frames = peaks(probability, period, 0.6, 0.5)
        beats = centroid(probability, frames, 2)

        if beats.size < 64:
            continue

        samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
        until = len(samples) / 22050.0 * 1000.0
        points = uninherited(archive)
        spacing = local_gap(beats, args.width)

        # Runs of beats whose spacing agrees, so that a tempo change and a level change are both boundaries.
        run_start = 0
        track_buckets = {k: 0.0 for k in buckets}

        for i in range(1, beats.size + 1):
            ends = i == beats.size

            if not ends and spacing[i] > 0 and abs(spacing[i] - spacing[i - 1]) <= 0.06 * spacing[i - 1]:
                continue

            # The run is [run_start, i), and the map's beat length is read at its middle.
            middle = (beats[run_start] + beats[min(i, beats.size - 1)]) / 2
            declared = map_length_at(points, middle)
            span = beats[min(i, beats.size - 1)] - beats[run_start]
            measured = float(np.median(spacing[run_start:i]))

            if declared <= 0 or span <= 0 or measured <= 0:
                track_buckets["unmeasured"] += span
            else:
                ratio = measured / declared

                if 0.8 <= ratio <= 1.25:
                    track_buckets["at the map's level"] += span
                elif 1.6 <= ratio <= 2.5:
                    track_buckets["an octave above"] += span
                elif 0.4 <= ratio <= 0.65:
                    track_buckets["an octave below"] += span
                else:
                    track_buckets["other"] += span

            run_start = i

        total = sum(track_buckets.values())

        if total > 0:
            per_track.append((track_id, track_buckets, total))

            for key in buckets:
                overall[key] += track_buckets[key]

    # Summarised from the per-track totals, so that a long track counts for more than a short one.
    print(f"{len(per_track)} tracks measured, weighted by duration")
    print("")
    print(f"  {'track':>9} {'at level':>9} {'octave up':>10} {'octave down':>12} {'other':>7} {'unmeasured':>11}")

    for track_id, counts, total in sorted(per_track, key=lambda t: -t[1]["an octave above"] / t[2])[:args.show]:
        print(f"  {track_id:>9} {100 * counts['at the map\'s level'] / total:8.1f}% "
              f"{100 * counts['an octave above'] / total:9.1f}% {100 * counts['an octave below'] / total:11.1f}% "
              f"{100 * counts['other'] / total:6.1f}% {100 * counts['unmeasured'] / total:10.1f}%")

    grand = sum(counts[k] for _, counts, _ in per_track for k in buckets)

    print("")
    print("over every track, as a share of the music's duration:")

    for key in ("at the map's level", "an octave above", "an octave below", "other", "unmeasured"):
        value = sum(counts[key] for _, counts, _ in per_track)
        print(f"  {key:>18}: {100 * value / grand:5.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
