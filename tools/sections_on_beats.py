"""The section reading run over the detector's own beats, which is what the player would get.

The same algorithm measured over the dataset's labels gives three sections for a track whose map has three, and eleven
over the detector's beats for the same track - so the difference between the two inputs is where the remaining fault
is, and it is worth knowing whether it is the gaps or the section rule before changing either.

Run on the detector's beats by replicating the C# reading exactly: the same suppression, the same sub-frame placement,
so that a difference between this and the probe that runs the real library is a difference in the algorithm and not in
the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402
from refine_choice import centroid  # noqa: E402
from sections_persist import analyse, robust  # noqa: E402


def beats_of_map(path) -> list[float]:
    """The map's own beats, from its uninherited timing points."""
    import zipfile

    if str(path).lower().endswith(".osz"):
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.lower().endswith(".osu"))
            text = archive.read(name).decode("utf-8", "replace")
    else:
        text = Path(path).read_text(encoding="utf-8", errors="replace")

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

    points.sort()
    out: list[float] = []

    for i, (time, length) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else time

        while time < end:
            out.append(time)
            time += length

    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--ids", default="1533028,1023679,1335372,1192164,67565")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--share", type=float, default=0.06)
    parser.add_argument("--persist", type=int, default=6)
    parser.add_argument("--shortest", type=int, default=16)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    for track_id in args.ids.split(","):
        cached = args.cache / f"{track_id}.npy"

        if not cached.exists():
            continue

        mels = np.load(cached).astype(np.float32)
        probability, period = infer(model, mels, device)
        period = smooth(period, 1.0)
        frames = peaks(probability, period, 0.6, 0.5)
        beats = centroid(probability, frames, 2)

        raw = np.diff(beats)
        gaps = robust(raw)
        runs = analyse(gaps, args.window, args.share, args.persist, args.shortest)

        archive = next(args.corpus.glob(f"{track_id} *.osz"), None)
        map_ms = np.array([])

        if archive is not None:
            map_ms = np.array(beats_of_map(archive))

        print(f"=== {track_id}: {beats.size} beats, gap scatter raw {np.std(raw):6.2f}ms -> robust {np.std(gaps):6.2f}ms")
        print(f"   {len(runs)} sections reported; the map holds its tempo for {len(map_ms)} beats")
        print(f"   {'from':>8} {'to':>8} {'BPM':>8} {'period':>9} {'gaps':>6} {'pos err':>8} {'span err':>9} {'drift':>8}")

        for a, b in sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:10]:
            run = gaps[a:b]
            middle = float(np.median(run))
            scatter = float(np.std(run))
            error = scatter / np.sqrt(max(1, b - a))
            span = beats[b] - beats[a]

            # Whether the section's edges land on the map's own beats, which is what matters for a UI: a boundary a
            # second from where the tempo changed is a tempo change shown in the wrong place, whatever the tempo is.
            position = float(np.min(np.abs(map_ms - beats[a]))) if map_ms.size else float("nan")

            print(f"   {beats[a] / 1000:7.1f}s {beats[b] / 1000:7.1f}s {60000 / middle:8.2f} {middle:8.2f}ms "
                  f"{b - a:6} {position:7.0f}ms {span / 1000:8.1f}s {error / middle * span:7.0f}ms")

        print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
