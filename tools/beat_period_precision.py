"""Whether the beats can name a section's tempo, since the period head cannot.

The period head turns out to be useless for this and not by a small margin: over thirty-nine sections the map holds at
one tempo, its middle error is nineteen per cent and it is frequently a whole factor of two out, which is a head that
cannot tell a beat from half of one. That is survivable where it is used, because suppression only needs a radius
roughly right, and it is fatal for a section, because a section is exactly one number.

The beats themselves are a different measurement - placed from the curve, landing a median seven milliseconds from the
map's grid - so the tempo can be taken from them instead. What is measured here is whether that is accurate enough:
the period implied by the gaps between reported beats, against the beat length the map holds, over the sections the map
holds one tempo for.

The distinction that matters is between the middle of the gaps, which is what a section's number would be, and how far
those gaps wander, which is what limits how long a section can be held constant. Both are reported, and so is the drift
the middle error implies by the end of the section.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model, log_mel  # noqa: E402
from diagnose_level import uninherited  # noqa: E402
from tempo_sections import sections as map_sections  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402
from refine_choice import centroid  # noqa: E402


def local_period(beats: np.ndarray, window: int) -> np.ndarray:
    """The period at each beat, as the middle of the gaps around it."""
    gaps = np.diff(beats)
    out = np.zeros(beats.size)

    for i in range(beats.size):
        low = max(0, i - window)
        high = min(gaps.size, i + window)

        out[i] = np.median(gaps[low:high]) if high > low else np.nan

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--ids", default="1335372,1533028,2155472,1633250,1192164,1023679,1850986,67565,594884,875117")
    parser.add_argument("--share", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--long", type=float, default=10.0)
    parser.add_argument("--windows", default="4,8,16,32")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    rows = {r["id"]: r for r in csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t")}
    windows = [int(w) for w in args.windows.split(",")]

    print("The beats' own gaps against the map's beat length, over sections the map holds one tempo for")
    print(f"{args.long:.0f}s or more. A 30ms error over a minute is a tenth of a beat.")
    print("")
    print(f"  {'track':>9} {'section':>17} {'map':>24} {'gaps':>24} {'error':>8} {'drift/min':>10} {'gap scatter':>12}")

    totals: dict[int, list[float]] = {w: [] for w in windows}
    drifts: dict[int, list[float]] = {w: [] for w in windows}

    for track_id in args.ids.split(","):
        if track_id not in rows:
            continue

        cached = args.cache / f"{track_id}.npy"
        archive = next(args.corpus.glob(f"{track_id} *.osz"), None)

        if not cached.exists() or archive is None:
            continue

        mels = np.load(cached).astype(np.float32)
        samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
        until = len(samples) / 22050.0 * 1000.0

        probability, period = infer(model, mels, device)
        period = smooth(period, args.smoothing)
        frames = peaks(probability, period, args.share, args.threshold)
        beats = centroid(probability, frames, 2)

        runs = [s for s in map_sections(uninherited(archive), until, 0.002) if s[1] - s[0] >= args.long * 1000.0]

        for start, end, length in sorted(runs, key=lambda s: s[1] - s[0], reverse=True)[:3]:
            inside = np.flatnonzero((beats >= start) & (beats < end))

            if inside.size < 24:
                continue

            span = (end - start) / 1000.0
            line = f"  {track_id:>9} {start / 1000:6.0f}..{end / 1000:6.0f}s {length:>9.2f}ms {60000 / length:6.1f}BPM"

            for window in windows:
                gaps = np.diff(beats[inside[0]:inside[-1] + 1])
                middle = float(np.median(gaps))
                error = (middle - length) / length
                scatter = float(np.percentile(gaps, 75) - np.percentile(gaps, 25))

                totals[window].append(error)
                drifts[window].append(abs(error) * (end - start))

                line += (f" | {middle:7.2f}ms {error * 100:+6.2f}% {abs(error) * 60_000:7.0f}ms {scatter:7.0f}ms")

            print(line)

    print("")
    print("  error is the middle gap against the map's beat length; drift is that error carried over a minute")
    print("")

    for window in windows:
        if not totals[window]:
            continue

        error = np.abs(np.array(totals[window]))
        drift = np.array(drifts[window])

        print(f"  middle gap over the section, for all {len(error)} sections:")
        print(f"    middle error   {np.median(error) * 100:6.3f}%   p90 {np.percentile(error, 90) * 100:6.3f}%")
        print(f"    drift a minute {np.median(error) * 60_000:6.0f}ms  p90 {np.percentile(error, 90) * 60_000:6.0f}ms")
        print(f"    within 0.05%   {int((error <= 0.0005).sum())}/{error.size}    within 0.10%  {int((error <= 0.001).sum())}/{error.size}")
        print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
