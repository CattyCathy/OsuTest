"""Correcting the suppression radius from the beats it picked, instead of from the period head.

The head is the thing that decides how far apart two beats have to be before both are kept, and measuring it against
the gaps between the training labels shows it is not biased but wildly imprecise: over twenty-four tracks its middle
ratio to the true period is 0.98, which is right, while six tracks are more than a tenth short and three of those are
short by a third or a half. A radius that is half the truth keeps a beat every half a beat, and that is what the
over-reporting looks like - on those three tracks the reading reports 2.16, 1.98 and 1.74 times the beats there are,
and with the radius taken from the labels' own gaps instead of from the head those same three come to 1.07, 1.08 and
1.03.

So the head is not wrong about the tempo, it is wrong about it locally, and there is a better estimate of the same
quantity sitting in the reading already: the gaps between the beats the first pass picked. Their middle is the spacing
that is actually there, measured from the beats that survived, and it needs no extra model.

What this tries is that estimate used as a second pass - pick with the head, take the middle of what came out, and pick
again with the radius from that - and the risk is stated plainly: the first pass is the one with the bad radius, so if
it let twice as many beats through then the middle gap of its output is half the truth and the second pass inherits the
error rather than fixing it. Whether it does is the measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model, read_track, load_manifest  # noqa: E402
from diagnose_level import uninherited, grid, nearest  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model-v5\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--share", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    reads = {}
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}
    tracks = [t for t in load_manifest(args.dataset) if t.id in archives][:args.limit]

    print(f"  {'track':>9} | {'rate 1':>7} {'rate 2':>7} {'rate 3':>7} | {'err 1':>7} {'err 3':>7} | {'cov 1':>7} {'cov 3':>7}")

    totals = {1: [[], [], []], 2: [[], [], []], 3: [[], [], []]}

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
        head = smooth(period, 1.0)

        # The first pass uses the head. Each later pass takes the radius from the middle gap of the pass before, held as
        # a period per frame so that a track whose tempo moves is still followed rather than given one number.
        radius = head
        results = {}

        for attempt in range(1, args.passes + 1):
            picked = peaks(probability, radius, args.share, args.threshold)

            if picked.size < 4:
                break

            times = picked / FPS * 1000.0
            gaps = np.diff(times)
            middle = float(np.median(gaps))

            error = nearest(times, truth)
            covered = nearest(truth, times)

            results[attempt] = (times.size / truth.size, float(np.median(error)),
                                100.0 * float((covered <= 60).mean()))

            for key, value in zip(("rate", "err", "cov"), results[attempt]):
                totals[attempt][{"rate": 0, "err": 1, "cov": 2}[key]].append(value)

            # Locally as well as globally, because a correction that is one number for a track cannot help a track
            # whose tempo moves - each frame takes the middle of the gaps near it.
            local = np.zeros(head.size, dtype=np.float32)

            for i in range(local.size):
                near = times[(times >= i / FPS * 1000.0 - 2000) & (times <= i / FPS * 1000.0 + 2000)]
                local[i] = float(np.median(np.diff(near))) * FPS / 1000.0 if near.size >= 3 else middle * FPS / 1000.0

            radius = local

        line = f"  {track.id:>9} |"
        for attempt in (1, 2, 3):
            line += f" {results.get(attempt, (float('nan'),) * 3)[0]:>7.2f}"
        line += " |"
        for attempt in (1, 3):
            line += f" {results.get(attempt, (0, float('nan'), 0))[1]:>7.1f}"
        line += " |"
        for attempt in (1, 3):
            line += f" {results.get(attempt, (0, 0, float('nan')))[2]:>6.1f}%"

        print(line)

    print("")
    print("  medians:")
    for attempt in (1, 2, 3):
        if totals[attempt][0]:
            print(f"    pass {attempt}: rate {np.median(totals[attempt][0]):.2f}   "
                  f"err {np.median(totals[attempt][1]):.1f}ms   cov {np.median(totals[attempt][2]):.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
