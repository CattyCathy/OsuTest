"""Prints the tempo curve as text, so its shape can be read without a window.

The panel draws this curve on a logarithmic axis two octaves either side of 150 BPM, which is 37 to 600. Whether a
curve reads as a steady line or as a staircase is the whole question the panel exists to answer, and that question can
be answered from the numbers: this bins the track into columns, takes the middle tempo in each, and prints the octave
that tempo falls in.

A column that is blank means no sample in it had a tempo the axis could place, which is what a panel drawn empty would
look like. A run of columns at one level followed by a run at another is a change of metrical level, and the size of
the step is what says whether it is a doubling.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

GUIDE_BPM = 150.0
OCTAVES = 2


def fraction_for(bpm: float) -> float | None:
    """Where a tempo sits on the panel's axis, or None when it is off it."""
    if bpm <= 0:
        return None

    fraction = 0.5 - (math.log2(bpm / GUIDE_BPM) / (2 * OCTAVES))

    return fraction if 0.0 <= fraction <= 1.0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("curve", type=Path)
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--rows", type=int, default=17)
    args = parser.parse_args()

    samples: list[tuple[float, float, float]] = []

    for raw in args.curve.read_text().splitlines():
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")

        if len(fields) < 2:
            continue

        try:
            samples.append((float(fields[0]), float(fields[1]), float(fields[2]) if len(fields) > 2 else 0.0))
        except ValueError:
            continue

    if not samples:
        print("no samples")
        return 1

    start = samples[0][0]
    end = samples[-1][0]
    span = max(1.0, end - start)

    print(f"{args.curve.name}: {len(samples)} samples, {start/1000:0.1f}s to {end/1000:0.1f}s")
    print(f"axis covers {GUIDE_BPM * 2**-OCTAVES:0.0f} to {GUIDE_BPM * 2**OCTAVES:0.0f} BPM")
    print()

    # One column per slice of the track, holding that slice's middle tempo.
    columns: list[float | None] = []
    confidences: list[float] = []

    for index in range(args.columns):
        low = start + (span * index / args.columns)
        high = start + (span * (index + 1) / args.columns)
        bucket = [s for s in samples if low <= s[0] < high]

        if not bucket:
            columns.append(None)
            confidences.append(0.0)
            continue

        tempos = sorted(s[1] for s in bucket)
        columns.append(tempos[len(tempos) // 2])
        confidences.append(sum(s[2] for s in bucket) / len(bucket))

    # The grid, one row per equal step of the axis so that a step of one row is one octave divided by the rows.
    grid = [[" " for _ in range(args.columns)] for _ in range(args.rows + 1)]

    for index, bpm in enumerate(columns):
        if bpm is None:
            continue

        fraction = fraction_for(bpm)

        if fraction is None:
            continue

        row = round(fraction * args.rows)
        grid[max(0, min(args.rows, row))][index] = "#"

    for row in range(args.rows + 1):
        fraction = row / args.rows
        bpm = GUIDE_BPM * 2 ** ((0.5 - fraction) * 2 * OCTAVES)
        guide = "-" if abs(fraction - 0.5) < 1e-9 else " "

        print(f"{bpm:6.0f} {guide} |{''.join(grid[row])}|")

    print(" " * 7 + "+" + "-" * args.columns + "+")

    # The same numbers, in runs, which is what a staircase actually is.
    print()
    print("runs of a steady reading, in octaves above the lowest seen:")

    placed = [(i, b) for i, b in enumerate(columns) if b is not None]

    if not placed:
        print("  nothing on the axis")
        return 1

    runs: list[tuple[int, int, float, float]] = []
    run_start, run_bpms = placed[0][0], [placed[0][1]]

    for index, bpm in placed[1:]:
        if index == run_start + len(run_bpms):
            run_bpms.append(bpm)
        else:
            runs.append((run_start, len(run_bpms), min(run_bpms), max(run_bpms)))
            run_start, run_bpms = index, [bpm]

    runs.append((run_start, len(run_bpms), min(run_bpms), max(run_bpms)))

    off_axis = sum(1 for b in columns if b is None)

    for start_index, length, low, high in sorted(runs, key=lambda r: -r[1])[:12]:
        if length < 2:
            continue

        middle = (low + high) / 2
        print(f"  columns {start_index:3}-{start_index+length-1:3} ({length:3} wide): "
              f"{low:5.0f}-{high:5.0f} BPM, middle {middle:5.0f}")

    print()
    print(f"columns with nothing on the axis: {off_axis} of {args.columns}")
    print(f"mean confidence: {sum(confidences)/len(confidences):0.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
