"""Which sub-frame refinement to use, decided rather than assumed.

A beat is reported on the frame its neighbourhood peaks at, and a frame is twenty milliseconds - two thirds of the
budget a rhythm game allows. Three ways of putting the peak between frames are compared, because they are not the same
estimator and the difference is worth more than the cost of measuring it:

  none        the frame the peak fell on, which is what the library does now.
  parabola    the vertex of the parabola through the peak and its two neighbours. Exact for a quadratic, which the
              curve nearly is over three frames, but it reads only three frames and one noisy one moves it a lot.
  centroid    the middle of the neighbourhood weighted by how far above the local floor each frame is. Reads more
              frames, so it is steadier, at the price of being pulled by the curve's asymmetry.
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
from diagnose_level import grid, nearest, uninherited  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402


def parabola(probability: np.ndarray, frames: np.ndarray) -> np.ndarray:
    out = []

    for index in frames:
        index = int(index)

        if index <= 0 or index + 1 >= probability.size:
            out.append(float(index))
            continue

        left, middle, right = float(probability[index - 1]), float(probability[index]), float(probability[index + 1])
        denominator = left - 2 * middle + right
        offset = 0.0

        if abs(denominator) > 1e-9:
            offset = max(-0.5, min(0.5, 0.5 * (left - right) / denominator))

        out.append(index + offset)

    return np.array(out) / FPS * 1000.0


def centroid(probability: np.ndarray, frames: np.ndarray, radius: int = 2) -> np.ndarray:
    out = []

    for index in frames:
        index = int(index)
        low = max(0, index - radius)
        high = min(probability.size, index + radius + 1)
        window = probability[low:high].astype(np.float64)

        if window.size < 2:
            out.append(float(index))
            continue

        floor = window.min()
        weight = window - floor

        if weight.sum() <= 1e-9:
            out.append(float(index))
            continue

        out.append(low + float((weight * np.arange(window.size)).sum() / weight.sum()))

    return np.array(out) / FPS * 1000.0


def measure(reported: np.ndarray, reference: np.ndarray, tolerance: float) -> tuple[float, float, float]:
    if reported.size == 0 or reference.size == 0:
        return 0.0, 0.0, float("nan")

    error = nearest(reported, reference)

    return (100.0 * float((nearest(reference, reported) <= tolerance).mean()),
            100.0 * float((error <= tolerance).mean()),
            float(np.median(error)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--ids", default="1335372,1533028,1192164")
    parser.add_argument("--share", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=30.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    rows = {r["id"]: r for r in csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t")}
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}

    totals: dict[str, list] = {}

    for track_id in args.ids.split(","):
        if track_id not in rows or track_id not in archives:
            continue

        cached = args.cache / f"{track_id}.npy"

        if cached.exists():
            mels = np.load(cached).astype(np.float32)
        else:
            samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
            mels = log_mel(samples).astype(np.float32)

        samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
        until = len(samples) / 22050.0 * 1000.0
        map_ms = grid(uninherited(archives[track_id]), until)
        map_ms = map_ms[map_ms <= until]

        probability, period = infer(model, mels, device)
        period = smooth(period, args.smoothing)
        frames = peaks(probability, period, args.share, args.threshold)

        print("")
        print(f"{track_id}: map {map_ms.size} beats, curve {frames.size} peaks")

        for name, reported in (
            ("none", frames / FPS * 1000.0),
            ("parabola", parabola(probability, frames)),
            ("centroid r1", centroid(probability, frames, 1)),
            ("centroid r2", centroid(probability, frames, 2)),
            ("centroid r3", centroid(probability, frames, 3)),
        ):
            coverage, precision, median = measure(reported, map_ms, args.tolerance)
            totals.setdefault(name, []).append((coverage, precision, median))
            print(f"  {name:>12}: coverage {coverage:5.1f}%  precision {precision:5.1f}%  median error {median:5.1f}ms")

    print("")
    print("averaged:")
    for name, values in totals.items():
        coverage = float(np.mean([v[0] for v in values]))
        precision = float(np.mean([v[1] for v in values]))
        median = float(np.nanmean([v[2] for v in values]))
        print(f"  {name:>12}: coverage {coverage:5.1f}%  precision {precision:5.1f}%  median error {median:5.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
