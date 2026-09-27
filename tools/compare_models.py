"""Scores two trained detectors side by side against the maps' own timing points.

The validation number a training run prints is measured against the labels, so a model that reproduces its training
target perfectly still scores zero if the target is wrong - and a model trained on a denser target is rewarded for
reporting more beats. Neither is a statement about the music.

So both models are read the same way the player reads them, with the same suppression and the same smoothing, and the
result is measured against the map's grid on the same tracks. Reported per track as well as in aggregate, because the
aggregate is what hid this the first time: a detector can look fine over a corpus and be useless on the one track a
player actually opened.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model, log_mel  # noqa: E402
from diagnose_level import grid, nearest, uninherited  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402
from refine_choice import centroid  # noqa: E402


def measure(reported: np.ndarray, reference: np.ndarray, tight: float, loose: float) -> dict:
    if reported.size == 0 or reference.size == 0:
        return {"coverage": 0.0, "precision": 0.0, "median": float("nan"),
                "tight": 0.0, "rate": 0.0, "count": int(reported.size)}

    error = nearest(reported, reference)

    return {
        "coverage": 100.0 * float((nearest(reference, reported) <= loose).mean()),
        "precision": 100.0 * float((error <= loose).mean()),
        "median": float(np.median(error)),
        "tight": 100.0 * float((nearest(reference, reported) <= tight).mean()),
        "rate": reported.size / reference.size,
        "count": int(reported.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--models", default=r"D:\Linux\Proj\OsuTest\model\detector.pt;D:\Linux\Proj\OsuTest\model-v2\detector.pt",
                        help="semicolon separated, because a comma is eaten by the shell before it gets here")
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--limit", type=int, default=24, help="how many tracks, taken from the manifest in order")
    parser.add_argument("--ids", default="", help="semicolon separated track ids; overrides --limit when given")
    parser.add_argument("--share", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--tight", type=float, default=20.0, help="the error a rhythm game feels")
    parser.add_argument("--loose", type=float, default=60.0)
    parser.add_argument("--refine", default="centroid",
                        help="none, centroid or parabola; the library's own placing is the weighted middle")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []

    for path in args.models.replace(",", ";").split(";"):
        # The period head's shape changed between models, so a checkpoint states which one it is. Read from the file
        # rather than guessed, because guessing wrong is a shape error rather than a wrong answer.
        state = torch.load(path, map_location="cpu")
        wide = "period.4.weight" in state

        network = build_model(torch, wide_period=wide).to(device)
        network.load_state_dict(state)
        network.eval()
        models.append((Path(path).parent.name, network))

    rows = list(csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t"))
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}

    if args.ids:
        wanted = [i for i in args.ids.replace(";", ",").split(",") if i]
        tracks = [i for i in wanted if i in archives]
    else:
        tracks = [r["id"] for r in rows if r["id"] in archives][:args.limit]

    print(f"{len(tracks)} tracks, share {args.share}, threshold {args.threshold}, smoothing {args.smoothing}s, "
          f"placing {args.refine}")
    print(f"coverage and precision at {args.loose:.0f}ms, tight coverage at {args.tight:.0f}ms")
    print("")
    print(f"  {'track':>9} {'map':>5} | " + " | ".join(f"{name:>26}" for name, _ in models))
    print(f"  {'':>9} {'':>5} | " + " | ".join(f"{'n  rate  cov  prec  tight  med':>26}" for _ in models))

    totals: dict[str, list] = {}

    for track_id in tracks:
        cached = args.cache / f"{track_id}.npy"

        if cached.exists():
            mels = np.load(cached).astype(np.float32)
        else:
            samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
            mels = log_mel(samples).astype(np.float32)
            np.save(cached, mels.astype(np.float16))

        samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
        until = len(samples) / 22050.0 * 1000.0
        map_ms = grid(uninherited(archives[track_id]), until)
        map_ms = map_ms[map_ms <= until]

        if map_ms.size < 16:
            continue

        line = f"  {track_id:>9} {map_ms.size:>5} | "

        for name, network in models:
            probability, period = infer(network, mels, device)
            period = smooth(period, args.smoothing)
            frames = peaks(probability, period, args.share, args.threshold)

            if args.refine == "centroid":
                reported = centroid(probability, frames, 2)
            elif args.refine == "parabola":
                from refine_choice import parabola
                reported = parabola(probability, frames)
            else:
                reported = frames / FPS * 1000.0

            result = measure(reported, map_ms, args.tight, args.loose)
            totals.setdefault(name, []).append(result)

            line += (f"{result['count']:>5} {result['rate']:>5.2f} {result['coverage']:>5.1f} "
                     f"{result['precision']:>5.1f} {result['tight']:>6.1f} {result['median']:>4.0f} | ")

        print(line)

    print("")
    print("median over the tracks (the middle is reported so one bad track cannot carry the average):")

    for name, values in totals.items():
        coverage = np.array([v["coverage"] for v in values])
        precision = np.array([v["precision"] for v in values])
        tight = np.array([v["tight"] for v in values])
        median = np.array([v["median"] for v in values])
        rate = np.array([v["rate"] for v in values])

        print(f"  {name}:")
        print(f"    rate      {np.median(rate):.2f}   on level (0.9-1.1) {int(((rate > 0.9) & (rate < 1.1)).sum())}/{rate.size}")
        print(f"    coverage  {np.median(coverage):.1f}%")
        print(f"    precision {np.median(precision):.1f}%")
        print(f"    tight     {np.median(tight):.1f}% within {args.tight:.0f}ms")
        print(f"    error     {np.median(median):.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
