"""Whether the labels, and then the detector, sit at the level the map declares.

The level is a per-track decision and on this corpus it was wrong on eleven of a hundred and thirteen - the exporter
preferred the model's own reading of the level over the map's unless the map won by a wide margin, and on those eleven
the model won. Every one of them came out at half the declared rate or, on one, at twice it.

The export can now be told to take the map's level unconditionally, and this checks the two things that follow from it
and that are genuinely different questions:

    the labels    do they sit at the declared level, which is a fact about the exporter and can be read off the file;
    the detector  does the trained model put its beats at that level, which is a fact about the model and is the one
                  that could still fail - on those eleven tracks the beat curve preferred the other level, and no
                  setting of the period head changes what the classifier fires on.

Run after an export, and again after a training run.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402
from refine_choice import centroid  # noqa: E402


def declared_lengths(path: Path, until: float) -> dict[int, float]:
    """Each beat length a map declares, weighted by how long it declares it for."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".osu"))
        text = archive.read(name).decode("utf-8", "replace")

    inside = False
    points: list[tuple[float, float]] = []

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

    points.sort()
    weight: dict[int, float] = collections.Counter()

    for i, (time, length) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else max(until, time)
        weight[round(length)] += max(0.0, end - time)

    return weight


def classify(measured: float, declared: float) -> str:
    if declared <= 0 or measured <= 0:
        return "unmeasured"

    ratio = measured / declared

    if 0.8 <= ratio <= 1.25:
        return "at the map's level"
    if 1.6 <= ratio <= 2.5:
        return "half the declared rate"
    if 0.4 <= ratio <= 0.65:
        return "twice the declared rate"

    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=None, help="omit to check only the labels")
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--show", type=int, default=16)
    args = parser.parse_args()

    rows = list(csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t"))
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}

    model = None
    device = None

    if args.model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(torch).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

    label_counts = collections.Counter()
    beat_counts = collections.Counter()
    worst: list[tuple[float, str, float, float]] = []

    for row in rows:
        track_id = row["id"]
        archive = archives.get(track_id)

        if archive is None:
            continue

        labels = np.fromfile(args.dataset / "labels" / f"{track_id}.i8", dtype=np.int8)
        labelled = np.flatnonzero(labels > 0)

        if labelled.size < 8:
            continue

        until = len(labels) / FPS * 1000.0
        weight = declared_lengths(archive, until)

        if not weight:
            continue

        declared = max(weight, key=weight.get)
        label_period = float(np.median(np.diff(labelled))) * 1000.0 / FPS

        label_class = classify(label_period, declared)
        label_counts[label_class] += 1

        if label_class != "at the map's level":
            worst.append((label_period / declared, track_id, declared, label_period))

        if model is None:
            continue

        cached = args.cache / f"{track_id}.npy"

        if not cached.exists():
            continue

        mels = np.load(cached).astype(np.float32)
        probability, period = infer(model, mels, device)
        period = smooth(period, 1.0)
        frames = peaks(probability, period, 0.6, 0.5)
        beats = centroid(probability, frames, 2)

        if beats.size < 8:
            continue

        beat_period = float(np.median(np.diff(beats)))
        beat_counts[classify(beat_period, declared)] += 1

        if classify(beat_period, declared) != "at the map's level":
            worst.append((beat_period / declared, f"{track_id} (beats)", declared, beat_period))

    total = sum(label_counts.values())

    print(f"{total} tracks")
    print("")
    print("the labels, against each map's declared beat length, weighted by how long it is declared:")
    for name in ("at the map's level", "half the declared rate", "twice the declared rate", "other"):
        if label_counts[name]:
            print(f"  {name:>24}: {label_counts[name]:3}  ({100 * label_counts[name] / max(1, total):5.1f}%)")

    if model is not None:
        beat_total = sum(beat_counts.values())
        print("")
        print(f"the detector, over {beat_total} tracks:")
        for name in ("at the map's level", "half the declared rate", "twice the declared rate", "other"):
            if beat_counts[name]:
                print(f"  {name:>24}: {beat_counts[name]:3}  ({100 * beat_counts[name] / max(1, beat_total):5.1f}%)")

    if worst:
        print("")
        print("  the tracks that still differ, worst first:")
        for ratio, track_id, declared, measured in sorted(worst, key=lambda w: abs(np.log2(w[0])), reverse=True)[:args.show]:
            print(f"    {track_id:>18}  map {declared:6.0f}ms ({60000 / declared:5.1f} BPM)  "
                  f"reading {measured:6.0f}ms ({60000 / measured:5.1f} BPM)  ratio {ratio:4.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
