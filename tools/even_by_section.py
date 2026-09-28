"""Whether a passage rendered evenly, at the model's own level, is steadier without being less accurate.

The user's answer settles the level question the acronym way: the marks on screen move at about the pace of the map, so
the model's tactus is right and the complaint is the spacing and not the tempo. That is a much narrower thing to fix.

The detector's beats have the right spacing on average - the median gap over a passage is within a per cent of the
map's beat - and are uneven beat by beat, because it inserts one every twenty or thirty and misses one every fifty.
Rendering a passage evenly means taking its tempo from the middle of its own gaps and then stepping from one anchor:
each gap becomes that tempo exactly, and the inserts and the omissions both disappear, which a repair that only looks
at neighbouring gaps cannot do for a passage that has settled onto the wrong side of a beat.

The risk is drift, and it is the reason earlier attempts failed: a period a fifth of a per cent out is a hundred
milliseconds out by the hundred and twenty-fifth beat. The phase is therefore re-taken from the first beat of every
passage, so an error can only grow within one passage and is bounded by its length.

Both things are measured: how even the output is, and how far it sits from the map's own beats. Even and far is no
use, and this is the measurement that says which of the two it is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model, load_manifest  # noqa: E402
from diagnose_level import uninherited, grid, nearest  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402
from refine_choice import centroid  # noqa: E402
from sections_persist import analyse, robust  # noqa: E402


def evenly(beats: np.ndarray, passes: int = 1) -> np.ndarray:
    """Each tempo passage stepped out from its own first beat, at the middle of its own gaps."""
    gaps = robust(np.diff(beats))
    runs = analyse(gaps, 16, 0.06, 20, 16)

    if not runs:
        return beats

    out: list[float] = []
    last = -np.inf

    for a, b in runs:
        if b <= a:
            continue

        period = float(np.median(gaps[a:b]))

        if period <= 0:
            continue

        anchor = beats[min(a, beats.size - 1)]
        at = anchor

        while at < beats[min(b, beats.size - 1)]:
            if at > last + (period * 0.5):
                out.append(at)
                last = at

            at += period

    return np.array(sorted(out)) if len(out) >= 4 else beats


def uniformity(beats: np.ndarray, width: int = 8) -> float:
    """The share of gaps within a quarter of the local period."""
    if beats.size < 3:
        return float("nan")

    gaps = np.diff(beats)
    local = np.array([np.median(gaps[max(0, i - width):min(gaps.size, i + width)]) for i in range(gaps.size)])

    return 100.0 * float((np.abs(gaps / local - 1) <= 0.25).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch, wide_period=True).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}
    tracks = [t for t in load_manifest(args.dataset) if t.id in archives][:args.limit]

    print(f"  {'track':>9} | {'err raw':>8} {'err even':>8} | {'cov raw':>8} {'cov even':>8} | "
          f"{'uniform raw':>11} {'even':>6}")

    totals = {"err_raw": [], "err_even": [], "cov_raw": [], "cov_even": [], "uni_raw": [], "uni_even": []}

    for track in tracks:
        cached = args.cache / f"{track.id}.npy"

        if not cached.exists():
            continue

        mels = np.load(cached).astype(np.float32)
        samples = np.fromfile(args.dataset / "audio" / f"{track.id}.f32", dtype=np.float32)
        until = len(samples) / 22050.0 * 1000.0
        truth = grid(uninherited(archives[track.id]), until)
        truth = truth[truth <= until]

        if truth.size < 32:
            continue

        probability, period = infer(model, mels, device)
        frames = peaks(probability, smooth(period, 1.0), 0.6, 0.5)
        beats = centroid(probability, frames, 2)

        if beats.size < 32:
            continue

        even = evenly(beats)

        error_raw = float(np.median(nearest(beats, truth)))
        error_even = float(np.median(nearest(even, truth)))
        cov_raw = 100.0 * float((nearest(truth, beats) <= 60).mean())
        cov_even = 100.0 * float((nearest(truth, even) <= 60).mean())

        totals["err_raw"].append(error_raw)
        totals["err_even"].append(error_even)
        totals["cov_raw"].append(cov_raw)
        totals["cov_even"].append(cov_even)
        totals["uni_raw"].append(uniformity(beats))
        totals["uni_even"].append(uniformity(even))

        print(f"  {track.id:>9} | {error_raw:>7.1f}ms {error_even:>7.1f}ms | {cov_raw:>7.1f}% {cov_even:>7.1f}% | "
              f"{uniformity(beats):>10.0f}% {uniformity(even):>5.0f}%")

    print("")
    print("  medians:")
    print(f"    error      {np.median(totals['err_raw']):6.1f}ms -> {np.median(totals['err_even']):6.1f}ms")
    print(f"    coverage   {np.median(totals['cov_raw']):6.1f}%  -> {np.median(totals['cov_even']):6.1f}%")
    print(f"    uniformity {np.median(totals['uni_raw']):6.0f}%  -> {np.median(totals['uni_even']):6.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
