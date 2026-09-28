"""Finds the constant-tempo sections properly, and says what a section can be worth.

The sliding rule failed on its own terms and the measurement that showed it is worth recording: run over the dataset's
labels - which the exporter placed exactly on the maps' own grid, so they carry no placement error and 872 consecutive
gaps of a 300 BPM passage come out at exactly 200.00ms with no scatter at all - the same rule still cut that passage
forty-three times. The rule was at fault, not the beats, and the cause is that a median over a local window cannot
absorb a rare doubled gap: one beat missed puts a gap of two periods among gaps of one, the window's median jumps, and
a tempo change is called where nothing changed.

So two things are done differently.

The gaps are first made robust: a gap that is a whole multiple of the surrounding tempo is a beat the detector missed
or a beat it placed twice, not a tempo change, so it is divided back down and the section is measured on what is left.
A gap that is merely out by a fifth is a badly placed beat and is pulled to the surrounding tempo, because a section is
a statement about the tempo and one bad beat should not be able to move it.

Then the sections are found by the standard method rather than by a threshold: the partition into k runs minimising the
sum of squared deviations from each run's own median, for every k, by dynamic programming. That is the best any
partition into k constant tempos can do, so it puts a floor under the error and makes the remaining question only how
many sections to report - which is a decision about the material and can be read off the numbers rather than guessed.

The error quoted for a run is the scatter of its gaps about its own median, as a share of the beat, and the drift that
scatter would accumulate if the section were held constant over its whole length. A run where that drift is large is a
run that no constant tempo describes, however it was found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def robust(gaps: np.ndarray, share: float = 0.25) -> np.ndarray:
    """Gaps with the beat-level faults taken out of them.

    A gap of about a whole multiple of its neighbours is a beat that was missed, so it is divided by that multiple; a
    gap merely a share out is a beat placed badly, so it is pulled to the middle of its neighbourhood. Both are needed
    and they fix different things: the first is what stops a missed beat reading as a tempo change, and the second is
    what lets a section be measured without one bad beat dragging its number.
    """
    lengths = np.diff(np.concatenate([[0.0], np.cumsum(gaps)]))
    out = gaps.astype(np.float64).copy()

    for i in range(out.size):
        low = max(0, i - 8)
        high = min(out.size, i + 9)
        local = np.median(np.concatenate([out[low:i], out[i + 1:high]])) if high - low > 2 else out[i]

        if local <= 0:
            continue

        multiple = max(1, int(round(out[i] / local)))

        if multiple >= 2 and abs(out[i] - multiple * local) <= share * local:
            out[i] = local
        elif abs(out[i] - local) > share * local:
            out[i] = local

    return out


def partition(values: np.ndarray, k: int, minimum: int) -> tuple[float, list[int]]:
    """The best split of a sequence into k runs, each scored by its squared deviation from its own median.

    Dynamic programming over the run ends. The cost of a run is measured against its own median rather than its mean,
    because a tempo is a number that half the gaps are above and half below and a mean is moved by the tail.
    """
    n = values.size

    if k * minimum > n:
        return np.inf, []

    # The cost of a run is the squared deviation of its values from their own middle. Measured against the middle
    # rather than the mean, because a tempo is a number half the gaps are above and half below and a mean is moved by
    # the tail - and the tail here is the beats the detector placed badly.
    def run_cost(i: int, j: int) -> float:
        window = values[i:j]
        middle = float(np.median(window))

        return float(np.sum((window - middle) ** 2))

    best = np.full((k + 1, n + 1), np.inf)
    cut = np.zeros((k + 1, n + 1), dtype=int)
    best[0, 0] = 0.0

    for parts in range(1, k + 1):
        for end in range(parts * minimum, n + 1):
            for start in range((parts - 1) * minimum, end - minimum + 1):
                if not np.isfinite(best[parts - 1, start]):
                    continue

                candidate = best[parts - 1, start] + run_cost(start, end)

                if candidate < best[parts, end]:
                    best[parts, end] = candidate
                    cut[parts, end] = start

    if not np.isfinite(best[k, n]):
        return np.inf, []

    ends = []
    end = n

    for parts in range(k, 0, -1):
        start = int(cut[parts, end])
        ends.append((start, end))
        end = start

    ends.reverse()

    return float(best[k, n]), [s for s, _ in ends] + [n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--ids", default="1533028,1335372,2155472,1192164,1023679,67565")
    parser.add_argument("--minimum", type=int, default=12, help="gaps a section may have at the fewest")
    parser.add_argument("--max-sections", type=int, default=40)
    parser.add_argument("--map", action="store_true", help="use the labels rather than the detector's beats")
    args = parser.parse_args()

    source = "the dataset's labels, which sit on the maps' own grid" if args.map else "the dataset's labels"
    print(f"reading {source}; sections are at least {args.minimum} gaps")
    print("")

    for track_id in args.ids.split(","):
        path = args.dataset / "labels" / f"{track_id}.i8"

        if not path.exists():
            continue

        labels = np.fromfile(path, dtype=np.int8)
        beats = np.flatnonzero(labels > 0) / 50.0 * 1000.0

        if beats.size < 64:
            continue

        raw = np.diff(beats)
        gaps = robust(raw)

        print(f"=== {track_id}: {beats.size} beats, {raw.size} gaps")
        print(f"   raw gap scatter   {np.std(raw):7.2f}ms   after making them robust   {np.std(gaps):7.2f}ms")

        for k in (1, 2, 3, 5, 8, 12, 20):
            total, starts = partition(gaps, k, args.minimum)

            if not starts or not np.isfinite(total):
                continue

            runs = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]
            residual = np.sqrt(total / gaps.size)

            # The longest run and what a constant tempo costs over it, which is the number the whole idea stands on.
            a, b = max(runs, key=lambda r: r[1] - r[0])
            run = gaps[a:b]
            middle = float(np.median(run))
            scatter = float(np.std(run))
            span = beats[b] - beats[a]

            print(f"   k={k:3}  residual {residual:7.3f}ms   longest {b - a:5} gaps "
                  f"({beats[a] / 1000:6.1f}s..{beats[b] / 1000:6.1f}s = {span / 1000:5.1f}s)  "
                  f"middle {middle:7.2f}ms  scatter {scatter:6.2f}ms  held-constant drift {scatter / max(middle, 1e-9) * span:8.0f}ms")

        print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
