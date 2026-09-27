"""How close are the maps to a handful of constant-tempo sections?

The grid a rhythm game is drawn against is not a beat list, it is a list of sections with a tempo each, and a player
reads a section as one pulse. That is a much smaller thing to get right than every beat, and this measures whether the
maps this detector is built for actually have that shape.

The complication is that a map's timing points are not all heard tempo changes. A mapper nudges a point by a tenth of a
BPM to move the scroll speed, and an editor writes a burst of points a tenth of a second apart; neither is a change
anyone can hear. So the points are first merged into sections whose tempos agree to within a share of each other, and
what is reported is the length of the sections that survive - because a map that is genuinely one tempo with four
hundred points written over it is a map this whole approach is easy for, and a map that changes tempo every two bars is
one where it cannot help.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np


def uninherited(osz: Path) -> list[tuple[float, float]]:
    with zipfile.ZipFile(osz) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".osu"))
        text = archive.read(name).decode("utf-8", "replace")

    points: list[tuple[float, float]] = []
    inside = False

    for raw in text.splitlines():
        line = raw.strip()

        if line.startswith("["):
            inside = line.lower() == "[timingpoints]"
            continue

        if not inside or not line:
            continue

        fields = line.split(",")

        if len(fields) < 2:
            continue

        try:
            time, length = float(fields[0]), float(fields[1])
        except ValueError:
            continue

        if length > 0:
            points.append((time, length))

    return sorted(points)


def sections(points: list[tuple[float, float]], until: float, tolerance: float) -> list[tuple[float, float, float]]:
    """(start, end, beat length) with neighbours whose tempos agree to within the tolerance merged."""
    if not points:
        return []

    out: list[list[float]] = []

    for i, (time, length) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else max(until, time)

        # A section whose tempo is within the tolerance of the one before it is the same pulse, so it extends it. The
        # length kept is the earlier one, because that is the one already in force and a nudge should not move a grid
        # that is already sounding right.
        if out and abs(length - out[-1][2]) <= tolerance * out[-1][2]:
            out[-1][1] = end
        else:
            out.append([time, end, length])

    return [(a, b, c) for a, b, c in out]


def mad(points: list[tuple[float, float]], tolerance: float) -> float:
    """How far a section's grid sits from the raw points it was merged from, in milliseconds."""
    if not points:
        return 0.0

    until = points[-1][0] + 10_000
    errors: list[float] = []

    for start, end, length in sections(points, until, tolerance):
        # Every raw point inside this section, at the phase the section's own grid gives it.
        for time, _ in points:
            if time < start or time >= end:
                continue

            offset = (time - start) % length
            errors.append(min(offset, length - offset))

    return float(np.median(errors)) if errors else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--tolerance", type=float, default=0.03, help="share of the beat a tempo may drift and merge")
    parser.add_argument("--long", type=float, default=5.0, help="seconds a section must last to count as a section")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    rows = list(csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t"))
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}

    print(f"tolerance {args.tolerance:.0%} of the beat, sections of at least {args.long:.0f}s count")
    print("")
    print(f"  {'track':>9} {'points':>7} {'sections':>9} {'kept':>6} {'longest':>8} {'median':>7} "
          f"{'coverage':>9} {'section BPM':>22}")

    totals: list[tuple[float, float, int, int]] = []

    for row in rows:
        track_id = row["id"]
        archive = archives.get(track_id)

        if archive is None:
            continue

        labels = np.fromfile(args.dataset / "labels" / f"{track_id}.i8", dtype=np.int8)
        until = len(labels) / 50.0 * 1000.0
        points = uninherited(archive)

        if len(points) < 2:
            continue

        merged = sections(points, until, args.tolerance)
        kept = [s for s in merged if s[1] - s[0] >= args.long * 1000.0]

        if not merged:
            continue

        covered = sum(s[1] - s[0] for s in kept)
        lengths = sorted((s[1] - s[0]) / 1000.0 for s in kept)

        totals.append((100.0 * covered / until, float(len(merged)), len(kept), len(points)))

        bpms = [f"{60000 / s[2]:.0f}" for s in sorted(kept, key=lambda s: s[1] - s[0], reverse=True)[:4]]

        print(f"  {track_id:>9} {len(points):>7} {len(merged):>9} {len(kept):>6} "
              f"{lengths[-1] if lengths else 0:>7.0f}s {np.median(lengths) if lengths else 0:>6.0f}s "
              f"{100.0 * covered / until:>8.1f}% {' '.join(bpms):>22}")

    print("")
    coverage = np.array([t[0] for t in totals])
    sections_n = np.array([t[1] for t in totals])
    kept_n = np.array([t[2] for t in totals])
    points_n = np.array([t[3] for t in totals])

    print(f"over {len(totals)} tracks:")
    print(f"  raw timing points        median {np.median(points_n):5.0f}   max {points_n.max():5.0f}")
    print(f"  after merging tempos     median {np.median(sections_n):5.0f}   max {sections_n.max():5.0f}")
    print(f"  sections of {args.long:.0f}s or more      median {np.median(kept_n):5.0f}   max {kept_n.max():5.0f}")
    print(f"  duration they cover      median {np.median(coverage):5.1f}%  min {coverage.min():5.1f}%  "
          f"above 90%: {int((coverage > 90).sum())}/{coverage.size}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
