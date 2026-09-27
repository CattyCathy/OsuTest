"""What the threshold and the suppression radius do against the maps, rather than against the labels.

Measured against its own training labels the model at the shipped reading reports nine per cent more beats than it was
taught, and raising the threshold to seven tenths takes that to two per cent under while making its beats more accurate
rather than less - precision against the labels goes from 77.6 to 85.0 per cent and coverage only from 84.6 to 82.1. A
higher threshold is normally a trade against coverage; here it is not, because the frames between 0.5 and 0.7 are
predictions the model is not confident in and most of them are on beats that are not there.

The labels are the model's own target though, and agreeing with them is not the same as agreeing with the music, so the
same sweep is run against each map's declared timing points. What is wanted is a reported rate of about one - one beat
reported per beat that exists - with the coverage and the accuracy kept, and this is where a rate near one can be told
from a reading that merely reports fewer beats.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model  # noqa: E402
from diagnose_level import uninherited, grid, nearest  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model-v5\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.75,0.8")
    parser.add_argument("--shares", default="0.6,0.75,0.9")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    shares = [float(s) for s in args.shares.split(",")]

    rows = []
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}

    from train_detector import load_manifest

    tracks = [t for t in load_manifest(args.dataset) if t.id in archives][:args.limit]

    # Read each track once: the curve does not depend on the threshold, so re-inferring per setting would be the same
    # arithmetic repeated.
    prepared = []

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
        prepared.append((probability, smooth(period, 1.0), truth))

    print(f"{len(prepared)} tracks, against each map's declared timing points")
    print("")
    print(f"  {'threshold':>9} {'share':>6} | {'rate':>6} {'coverage':>9} {'precision':>10} {'median err':>11}")

    for threshold in thresholds:
        for share in shares:
            rates, coverages, precisions, errors = [], [], [], []

            for probability, period, truth in prepared:
                picked = peaks(probability, period, share, threshold) / FPS * 1000.0

                if picked.size == 0:
                    continue

                error = nearest(picked, truth)
                back = nearest(truth, picked)

                rates.append(picked.size / truth.size)
                coverages.append(100.0 * float((back <= 60).mean()))
                precisions.append(100.0 * float((error <= 60).mean()))
                errors.append(float(np.median(error)))

            print(f"  {threshold:>9.2f} {share:>6.2f} | {np.median(rates):>6.2f} {np.median(coverages):>8.1f}% "
                  f"{np.median(precisions):>9.1f}% {np.median(errors):>10.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
