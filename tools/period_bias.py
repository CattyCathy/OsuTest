"""Is the over-reporting caused by the period head telling the suppression to look too narrowly?

Suppression keeps the strongest frame and then refuses anything within a share of the predicted period. So the radius
is the period the model predicts, and a period that comes out too short is a radius too small - which is exactly how a
half-beat survives: on a passage whose beat is 700ms, a period predicted at 350 lets a beat through 350ms away, and the
reading reports the beat and its half.

That is a specific and checkable claim, and it is separable from the model simply firing too readily. Against the gaps
between the training labels - which sit on the maps' grids, so they are the spacing that is actually there - the
predicted period is compared per frame. If it is biased short, the fix is in the period head and not in the beat
classifier, and the two need different work.

Read beside it is what a perfect period would do: the reading is run again with the radius taken from the labels' own
gaps instead of from the head, which is not something that can ship but says whether the head is the whole problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model, read_track, load_manifest  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402


def true_period(labels: np.ndarray) -> np.ndarray:
    """The gap to the next labelled beat, per frame, or the previous when there is none ahead."""
    out = np.zeros(labels.size, dtype=np.float32)
    beats = np.flatnonzero(labels > 0.5)

    if beats.size < 2:
        return np.full(labels.size, 20.0, dtype=np.float32)

    for i in range(labels.size):
        index = int(np.searchsorted(beats, i, side="left"))
        low = max(0, index - 1)
        high = min(beats.size - 1, index)

        if high <= low:
            gap = beats[min(beats.size - 1, low + 1)] - beats[low]
        else:
            gap = beats[high] - beats[low]

        out[i] = max(4.0, float(gap))

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model-v5\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--share", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    tracks = load_manifest(args.dataset)[:args.limit]

    ratios: list[float] = []
    print(f"  {'track':>9} {'pred/true period':>17} {'rate':>6} {'rate with the true radius':>26}")

    for track in tracks:
        cached = args.cache / f"{track.id}.npy"

        if not cached.exists():
            continue

        mels = np.load(cached).astype(np.float32)
        _, labels = read_track(track)
        truth = np.flatnonzero(labels > 0.5) / FPS * 1000.0

        if truth.size < 32:
            continue

        probability, period = infer(model, mels, device)
        smoothed = smooth(period, args.smoothing)

        exact = true_period(labels)
        ratio = float(np.median(smoothed / exact))
        ratios.append(ratio)

        turned = peaks(probability, smoothed, args.share, args.threshold)
        perfect = peaks(probability, exact, args.share, args.threshold)

        print(f"  {track.id:>9} {ratio:>17.3f} {turned.size / truth.size:>6.2f} "
              f"{perfect.size / truth.size:>26.2f}")

    if ratios:
        values = np.array(ratios)
        print("")
        print(f"  middle ratio of predicted to true period over {values.size} tracks: {np.median(values):.3f}")
        print(f"  tracks where the predicted period is short by more than a tenth: {int((values < 0.9).sum())}")
        print(f"  and where it is long by more than a tenth:                        {int((values > 1.1).sum())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
