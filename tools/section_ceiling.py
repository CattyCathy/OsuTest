"""Whether a section reading fails because of beat jitter or because of its own method.

The detector's beats carry placement error of a few milliseconds and the maps are often at 300 BPM, where a beat is
200ms - so the jitter is a per cent or two of the gap and any rule that compares a short window of gaps against another
will fire on it. That is a property of the input, not of the rule, and the two are told apart by running the same rule
over the dataset's labels, which the exporter placed exactly on the maps' own grid and which therefore carry no
placement error at all.

If the labels give one section where the map has one, the rule is sound and the beats are the problem. If the labels
also fragment, the rule is the problem. The two need opposite fixes, so it is worth one measurement to know which.

A second reading is taken at the same time: the best a constant tempo can do over a run of gaps, which is the standard
deviation of the gaps against a median. No partition can beat that - it is the residual of the best possible
description of the run as one number - so it says how long a section can be before a constant tempo stops being a good
description, whatever rule is used to find it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def boundaries(gaps: np.ndarray, window: int, share: float) -> list[int]:
    """Where a rule comparing a near window of gaps against a far one would split."""
    out = []

    for i in range(1, len(gaps)):
        low = max(0, i - window)
        high = min(len(gaps), i + window)
        far_low = max(0, i - (2 * window) - 1)
        far_high = max(0, i - window)

        if high <= low or far_high <= far_low:
            continue

        near = np.median(gaps[low:high])
        far = np.median(gaps[far_low:far_high])

        if far > 0 and abs(near - far) / far >= share:
            out.append(i)

    return out


def segments(gaps: np.ndarray, cuts: list[int], shortest: int) -> list[tuple[int, int]]:
    starts = [0]

    for cut in cuts:
        if cut - starts[-1] >= shortest:
            starts.append(cut)

    return [(starts[i], starts[i + 1] if i + 1 < len(starts) else len(gaps)) for i in range(len(starts))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--ids", default="1533028,1335372,2155472,1192164,1023679,67565")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--share", type=float, default=0.08)
    parser.add_argument("--shortest", type=int, default=8)
    args = parser.parse_args()

    print(f"rule: middle of {args.window} gaps either side against the {args.window} before them, split at {args.share:.0%}")
    print("")

    for track_id in args.ids.split(","):
        path = args.dataset / "labels" / f"{track_id}.i8"

        if not path.exists():
            continue

        labels = np.fromfile(path, dtype=np.int8)
        beats = np.flatnonzero(labels > 0) / 50.0 * 1000.0

        if beats.size < 32:
            continue

        gaps = np.diff(beats)
        cuts = boundaries(gaps, args.window, args.share)
        runs = segments(gaps, cuts, args.shortest)

        print(f"=== {track_id}: {beats.size} labelled beats")

        for name, cut_list, run_list in (
            ("labels", cuts, runs),
        ):
            periods = [float(np.median(gaps[a:b])) for a, b in run_list if b > a]
            rounded = np.round(periods).astype(int)
            values, counts = np.unique(rounded, return_counts=True)
            order = np.argsort(-counts)

            print(f"   {name}: {len(run_list)} sections from {len(cut_list)} cuts")
            print(f"     periods seen, most common first: "
                  + ", ".join(f"{values[i]}ms x{counts[i]}" for i in order[:8]))

            # The longest runs, which are the ones a constant tempo has to describe for a long time.
            longest = sorted(run_list, key=lambda r: r[1] - r[0], reverse=True)[:3]

            for a, b in longest:
                run = gaps[a:b]
                middle = float(np.median(run))
                spread = float(np.std(run))
                print(f"     run of {b - a:5} gaps ({beats[a] / 1000:6.1f}s..{beats[b] / 1000:6.1f}s): "
                      f"middle {middle:8.2f}ms  scatter {spread:6.2f}ms ({100 * spread / middle:5.2f}% of the beat)  "
                      f"drift if held constant {spread / middle * (beats[b] - beats[a]):8.0f}ms")

        print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
