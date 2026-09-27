"""Audits the exported labels against the maps' own grids, across the whole corpus.

The labels are the detector's entire notion of what a beat is, and they are built by an exporter that picks the octave
with the help of Beat This! - the very model being replaced. That circularity is invisible from the player, so it has to
be measured here.

Two questions per track, and they are different:

  is each label on the map's grid at all   the exporter writes a grid point, so it must be. Anything else is a bug.
  is the label level the map's own level   the exporter is allowed to choose an octave, and a dense map's snap
                                           resolution genuinely is above the beat. What matters is whether that choice is
                                           stable, so the per-window ratio of map gap to label gap is the measurement.

A ratio of one means the labels sit exactly on the map's beat. Two means an octave below it - the labels carry half the
beats the map does. Drifting between the two between windows is the failure that matters, because a detector trained on
such labels has been taught that the level changes when the tempo did not.
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


def grid(timing: list[tuple[float, float]], until: float, scale: float = 1.0) -> np.ndarray:
    out: list[float] = []

    for i, (time, length) in enumerate(timing):
        step = length * scale

        if step <= 5:
            continue

        end = timing[i + 1][0] if i + 1 < len(timing) else max(until, time)

        while time < end and len(out) < 5_000_000:
            out.append(time)
            time += step

    return np.array(sorted(out))


def nearest(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    target = np.sort(target)

    if target.size == 0 or source.size == 0:
        return np.full(source.shape, np.inf)

    index = np.clip(np.searchsorted(target, source), 1, target.size - 1)

    return np.minimum(np.abs(source - target[index - 1]), np.abs(source - target[index]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--window", type=float, default=20.0, help="seconds per ratio window")
    parser.add_argument("--show", type=int, default=15, help="how many tracks to print in full")
    args = parser.parse_args()

    rows = list(csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t"))
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}

    print(f"{len(rows)} tracks in the manifest")
    print("")
    print(f"  {'track':>10} {'labels':>7} {'map':>6} {'map/label':>9} {'on grid %':>10} {'stable %':>9} {'ratio med':>10}")

    all_on_grid = []
    all_stable = []
    all_ratio = []
    worst = []

    for row in rows:
        track_id = row["id"]
        archive = archives.get(track_id)

        if archive is None:
            continue

        labels = np.fromfile(args.dataset / "labels" / f"{track_id}.i8", dtype=np.int8)
        label_ms = np.flatnonzero(labels > 0) / 50.0 * 1000.0

        if label_ms.size < 8:
            continue

        until = len(labels) / 50.0 * 1000.0
        timing = uninherited(archive)
        map_ms = grid(timing, until)
        map_ms = map_ms[map_ms <= until]

        if map_ms.size < 8:
            continue

        # On the grid: each label against the nearest map beat, but only counted when the label is at a grid point at
        # all. A tight tolerance, because a label that is not on the grid is a bug rather than a rounding.
        on_grid = 100.0 * float((nearest(label_ms, map_ms) <= 25).mean())

        label_gaps = np.diff(label_ms)
        map_gaps = np.diff(map_ms)

        ratios = []
        count_ratios = []

        for start in np.arange(0, until, args.window * 1000.0):
            in_map = map_gaps[(map_ms[:-1] >= start) & (map_ms[:-1] < start + args.window * 1000.0)]
            in_label = label_gaps[(label_ms[:-1] >= start) & (label_ms[:-1] < start + args.window * 1000.0)]

            if in_map.size < 8 or in_label.size < 8:
                continue

            ratios.append(float(np.median(in_map) / np.median(in_label)))

            # The count ratio is the more robust statement of the level: how many map beats one labelled beat stands
            # for. A median gap ratio can be thrown by one transition inside the window.
            count_ratios.append(float(in_map.size / in_label.size))

        ratios = np.array(ratios)

        if ratios.size == 0:
            continue

        # Stable: the ratio holds one value for the track. A track that is genuinely double-time in one half and not
        # the other is rare, so the count of windows near the track's own median is a fair reading.
        ratio_median = float(np.median(ratios))
        stable = 100.0 * float((np.abs(ratios - ratio_median) < 0.35).mean())

        all_on_grid.append(on_grid)
        all_stable.append(stable)
        all_ratio.append(ratio_median)
        worst.append((stable, track_id, ratio_median, on_grid, label_ms.size, map_ms.size))

    print("")
    all_on_grid = np.array(all_on_grid)
    all_stable = np.array(all_stable)
    all_ratio = np.array(all_ratio)

    print(f"across {all_on_grid.size} tracks:")
    print(f"  labels on the map grid    median {np.median(all_on_grid):5.1f}%   worst {all_on_grid.min():5.1f}%")
    print(f"  level stable over time    median {np.median(all_stable):5.1f}%   worst {all_stable.min():5.1f}%")
    print(f"  map/label ratio           median {np.median(all_ratio):5.2f}    "
          f"near 1 {int(((all_ratio > 0.8) & (all_ratio < 1.25)).sum())}  "
          f"near 2 {int(((all_ratio > 1.6) & (all_ratio < 2.5)).sum())}  "
          f"other {int((~(((all_ratio > 0.8) & (all_ratio < 1.25)) | ((all_ratio > 1.6) & (all_ratio < 2.5)))).sum())}")

    print("")
    print(f"  the {args.show} least stable tracks:")
    worst.sort()

    for stable, track_id, ratio, on_grid, labels_n, map_n in worst[:args.show]:
        print(f"    {track_id:>10}  stable {stable:5.1f}%  ratio {ratio:4.2f}  on grid {on_grid:5.1f}%  "
              f"labels {labels_n:5}  map {map_n:5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
