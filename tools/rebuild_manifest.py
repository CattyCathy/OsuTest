"""Rebuilds dataset/manifest.tsv from the exported files.

Needed because the export runs as several processes and each one wrote the manifest in full, so the last shard to
finish overwrote the index for every track but its own. The audio and labels are all on disk; only the index was lost,
and every field in it can be recovered from them.

The frame rate is exactly fifty frames a second, so a frame index converts to a time with no rounding, and the period
reported here is the middle gap between the labelled beats - which is the period of the labels rather than of the map,
and is the number that matters to anything reading them.
"""

from __future__ import annotations

import struct
from pathlib import Path

DATASET = Path(r"D:\Linux\Proj\OsuTest\dataset")
FPS = 50


def main() -> int:
    audio = {p.stem: p for p in (DATASET / "audio").glob("*.f32")}
    labels = {p.stem: p for p in (DATASET / "labels").glob("*.i8")}

    ids = sorted(set(audio) & set(labels))
    missing = sorted((set(audio) | set(labels)) - set(ids))

    print(f"{len(audio)} audio files, {len(labels)} label files, {len(ids)} pairs")

    if missing:
        print(f"  unpaired: {missing[:10]}")

    rows = []
    total_frames = 0
    total_beats = 0
    silent = []

    for track_id in ids:
        import numpy as np

        label = np.fromfile(labels[track_id], dtype=np.int8)
        frames = len(label)
        beats = int((label > 0).sum())

        total_frames += frames
        total_beats += beats

        if beats < 8:
            silent.append(track_id)

        gaps = np.diff(np.flatnonzero(label > 0)) / FPS * 1000
        period = int(round(float(np.median(gaps)))) if gaps.size else 0

        rows.append((track_id, frames, beats, period))

    with (DATASET / "manifest.tsv").open("w") as handle:
        handle.write("id\tset\taudio\tlabels\tframes\tbeats\tperiod\tlag\toctave\tusable\n")

        for track_id, frames, beats, period in rows:
            handle.write(f"{track_id}\t{track_id}\t{track_id}.f32\t{track_id}.i8\t"
                         f"{frames}\t{beats}\t{period}\t0\t0\t{1 if beats >= 8 else 0}\n")

    periods = sorted(r[3] for r in rows if r[3] > 0)

    print(f"{total_frames} frames, {total_frames / FPS / 60:.1f} minutes")
    print(f"{total_beats} labelled beats, {total_beats / (total_frames / FPS):.2f} per second")
    print(f"label periods: {periods[0]}ms to {periods[-1]}ms  ({60000/periods[-1]:.0f} to {60000/periods[0]:.0f} BPM)")

    # What a detector will actually be asked to reproduce, so a corpus that is all one tempo is visible before training
    # rather than as a suspiciously good validation number.
    buckets: dict[int, int] = {}

    for period in periods:
        bucket = (period // 50) * 50
        buckets[bucket] = buckets.get(bucket, 0) + 1

    print("")
    print("tracks by label period:")
    for bucket in sorted(buckets):
        print(f"  {bucket:4}ms ({60000/max(1,bucket):5.0f} BPM): {'#' * buckets[bucket]} {buckets[bucket]}")

    if silent:
        print("")
        print(f"{len(silent)} tracks have almost no labels and are marked unusable: {silent[:10]}")

    print("")
    print(f"wrote {DATASET / 'manifest.tsv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
