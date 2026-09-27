"""A change is where the tempo stays different, not where a window of it wobbles.

The sliding rule that failed measured the gaps near a beat against the gaps just before it, so a single rare gap could
move one of the two middles past the threshold and a boundary was called. Running that rule over the dataset's labels -
which sit exactly on the maps' own grid, 872 consecutive gaps of one passage coming out at exactly 200.00ms - still cut
that passage forty-three times, which is how the fault was found: it was the rule and not the beats.

What is tried here is the same comparison with the two things that made it fragile removed. The gaps are first made
robust, so that a beat the detector missed or placed twice is not a tempo; and a boundary is only accepted where the
near tempo stays away from the far one for several consecutive gaps, so that a change is something the reading holds to
rather than something it passes through. The second is the one that matters, because a real change of tempo persists
by definition and jitter does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def smooth_gaps(gaps: np.ndarray, width: int = 8) -> np.ndarray:
    """The gaps with each one replaced by the middle of its neighbourhood.

    This is the step the whole reading turns on, and the measurement that says so: a detector at 300 BPM reports a
    median gap of exactly 200.00ms over a passage the map holds at 200.00ms, but only 41% of its individual gaps are
    within two per cent of that median - the rest are beats placed a fraction late or early, and a few are beats
    inserted or missed. So the tempo is right and the individual gaps are not, which means a rule that reads one gap
    against another is reading noise: the tempo has to be taken from a run before anything can be said about where the
    run ends.

    Done here rather than inside the boundary test, because the boundary test compares a run against a run and both have
    to be of smoothed values or a single fault still moves one side past the threshold.
    """
    out = np.zeros_like(gaps, dtype=np.float64)

    for i in range(gaps.size):
        low = max(0, i - width)
        high = min(gaps.size, i + width + 1)

        out[i] = float(np.median(gaps[low:high]))

    return out


def robust(gaps: np.ndarray, share: float = 0.25) -> np.ndarray:
    """Gaps with the beat-level faults taken out of them.

    A gap of about a whole multiple of its neighbours is a beat that was missed, so it is divided by that multiple; a
    gap merely a share out is a beat placed badly, so it is pulled to the middle of its neighbourhood. Both are needed
    and they fix different things: the first is what stops a missed beat reading as a tempo change, and the second is
    what lets a section be measured without one bad beat dragging its number.

    The neighbourhood is taken wide and then narrow. A middle over twelve gaps either side is set by the tempo rather
    than by the two or three faults in it, where a middle over four can be moved by a fault on one side - but a wide
    window also smooths over a genuine short fault, so the estimate it gives is then used by a narrower pass to catch
    what the wide one missed. Run over the detector's beats as well as the labels, this is the step that decides whether
    a missed beat reads as a missed beat or as a change of tempo.
    """
    out = gaps.astype(np.float64).copy()

    for width in (12, 4):
        previous = out.copy()

        for i in range(out.size):
            low = max(0, i - width)
            high = min(out.size, i + width + 1)
            around = np.concatenate([previous[low:i], previous[i + 1:high]])
            local = float(np.median(around)) if around.size > 2 else float(previous[i])

            if local <= 0:
                continue

            multiple = max(1, int(round(previous[i] / local)))

            if multiple >= 2 and abs(previous[i] - multiple * local) <= share * local:
                out[i] = local
            elif abs(previous[i] - local) > share * local:
                out[i] = local
            else:
                out[i] = previous[i]

    return out


def analyse(gaps: np.ndarray, window: int, share: float, persist: int, shortest: int) -> list[tuple[int, int]]:
    """Section boundaries over a run of gaps, as (first gap, one past the last gap)."""
    n = gaps.size

    if n < 2 * window:
        return [(0, n)]

    # Where the near tempo differs from the far one by enough, before persistence is asked for.
    flagged = []

    for i in range(1, n):
        low = max(0, i - window)
        high = min(n, i + window)
        far_low = max(0, i - (2 * window) - 1)
        far_high = max(0, i - window)

        if far_high <= far_low or high <= low:
            flagged.append(False)
            continue

        near = np.median(gaps[low:high])
        far = np.median(gaps[far_low:far_high])

        flagged.append(far > 0 and abs(near - far) / far >= share)

    # A change is a run of flags. The run is taken from the first flag to the last one within a small gap, and the
    # boundary goes at the middle of that span - which is where the change is - rather than after the persistence has
    # been satisfied, which is where the change stopped being visible. Requiring the flags to be strictly consecutive
    # places the boundary late by however long the jitter takes to break the run, and on a clean step that was measured
    # at eighteen beats.
    starts = [0]
    run_first = -1
    run_last = -1

    def flush():
        if run_first < 0:
            return

        if run_last - run_first + 1 >= persist:
            boundary = (run_first + run_last) // 2

            if boundary - starts[-1] >= shortest:
                starts.append(boundary)

    for i, flag in enumerate(flagged):
        if flag:
            # A flag near the last one continues the same run; a flag far from it starts a new one.
            if run_first < 0 or i - run_last > 2:
                flush()
                run_first = i

            run_last = i
        elif run_first >= 0 and i - run_last > 2:
            flush()
            run_first = run_last = -1

    flush()

    return [(starts[i], starts[i + 1] if i + 1 < len(starts) else n) for i in range(len(starts))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--ids", default="1533028,1335372,2155472,1192164,1023679,67565")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--share", type=float, default=0.06)
    parser.add_argument("--persist", type=int, default=6)
    parser.add_argument("--shortest", type=int, default=16)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    for track_id in args.ids.split(","):
        path = args.dataset / "labels" / f"{track_id}.i8"

        if not path.exists():
            continue

        labels = np.fromfile(path, dtype=np.int8)
        beats = np.flatnonzero(labels > 0) / 50.0 * 1000.0

        if beats.size < 64:
            continue

        gaps = robust(np.diff(beats))

        print(f"=== {track_id}: {beats.size} beats")

        if args.sweep:
            for window in (16, 24, 32, 48):
                for share in (0.04, 0.06, 0.08, 0.10):
                    for persist in (3, 6, 12):
                        runs = analyse(gaps, window, share, persist, args.shortest)
                        periods = [float(np.median(gaps[a:b])) for a, b in runs]
                        rounded = sorted({int(round(p / 10) * 10) for p in periods})
                        print(f"   window {window:3} share {share:4.2f} persist {persist:3} -> {len(runs):3} sections, "
                              f"tempos {rounded[:10]}")
        else:
            runs = analyse(gaps, args.window, args.share, args.persist, args.shortest)
            print(f"   window {args.window} share {args.share:.2f} persist {args.persist} "
                  f"-> {len(runs)} sections")
            print(f"   {'from':>8} {'to':>8} {'BPM':>8} {'period':>9} {'gaps':>6} {'gap sd':>7} "
                  f"{'period se':>10} {'drift':>8}")

            for a, b in sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:12]:
                run = gaps[a:b]
                middle = float(np.median(run))
                scatter = float(np.std(run))
                span = beats[b] - beats[a]

                # What a section actually costs is the error in the period, not the scatter of the gaps it was measured
                # from: the period is a middle of n gaps and a middle of n noisy values is about sqrt(n) times steadier
                # than one of them. Reporting the scatter as the drift is a factor of sqrt(n) too pessimistic, which on
                # a section of eight hundred gaps is nearly thirty times.
                error = scatter / np.sqrt(max(1, b - a))
                drift = error / middle * span

                print(f"   {beats[a] / 1000:7.1f}s {beats[b] / 1000:7.1f}s {60000 / middle:8.2f} {middle:8.2f}ms "
                      f"{b - a:6} {scatter:6.2f}ms {error:9.3f}ms {drift:7.0f}ms")

        print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
