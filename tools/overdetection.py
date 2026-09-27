"""Why the detector reports more beats than there are: is it the model or the reading?

Over-detection is the thing that blocks a steady grid, and it has two possible homes that need opposite fixes. The
model reports a probability per frame and the reading turns that into beats with a threshold and a suppression radius,
so a model that is perfectly calibrated but read too loosely gives the same symptom as a model that fires on onsets
that are not there.

They are told apart by measuring against the training labels rather than against a map. The labels are what the model
was asked to reproduce; a reading that reports more beats than the labels is a reading admitting too much, and a reading
that matches the labels while the labels exceed the map is a label problem. Against a map neither can be told from the
other, because a map's own beat count is a third thing.

Reported for the reading as it ships and for the threshold and suppression radius swept, because if it is the reading
then the fix is a number and not a model.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model, read_track, load_manifest  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model-v5\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--thresholds", default="0.3,0.5,0.7,0.8,0.9")
    parser.add_argument("--shares", default="0.5,0.6,0.75,0.9")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    tracks = load_manifest(args.dataset)[:args.limit]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    shares = [float(s) for s in args.shares.split(",")]

    # Beat rate against the labels, and how much of the labels is covered, for each setting.
    results: dict[tuple[float, float], list[tuple[float, float]]] = {}

    print(f"{len(tracks)} tracks, measured against the dataset's labels")
    print("")
    print(f"  {'threshold':>9} {'share':>6} | {'rate':>6} {'coverage':>9} {'precision':>10}")

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
        period = smooth(period, 1.0)

        for threshold in thresholds:
            for share in shares:
                picked = peaks(probability, period, share, threshold) / FPS * 1000.0

                if picked.size == 0:
                    continue

                index = np.clip(np.searchsorted(truth, picked), 1, truth.size - 1)
                nearest = np.minimum(np.abs(picked - truth[index - 1]), np.abs(picked - truth[index]))

                covered = np.clip(np.searchsorted(picked, truth), 1, picked.size - 1)
                nearest_back = np.minimum(np.abs(truth - picked[covered - 1]), np.abs(truth - picked[covered]))

                results.setdefault((threshold, share), []).append((
                    picked.size / truth.size,
                    100.0 * float((nearest_back <= 25).mean()),
                    100.0 * float((nearest <= 25).mean()),
                ))

    for (threshold, share), values in sorted(results.items()):
        rate = float(np.median([v[0] for v in values]))
        coverage = float(np.median([v[1] for v in values]))
        precision = float(np.median([v[2] for v in values]))

        print(f"  {threshold:>9.2f} {share:>6.2f} | {rate:>6.2f} {coverage:>8.1f}% {precision:>9.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
