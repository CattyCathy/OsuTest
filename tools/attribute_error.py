"""Is the detector at fault, or the labels it was told to reproduce?

The sweep over the reading put a hard ceiling on what the trained detector can be read into: at the threshold and
suppression share that maximise accuracy it still places one beat in five away from the map, and it cannot reach both
accuracy and coverage at once. A ceiling like that is a property of the curve, not of how the curve is read.

So the curve is checked against the labels it was trained on, and the labels against the map. The three statements are
different and each points somewhere else:

    the curve reproduces the labels, and the labels miss the map   the training target is the problem
    the curve does not reproduce its own labels                    the model is under-trained or under-sized
    the labels match the map and the curve does not                 capacity or optimisation

Reported at several tolerances, because which of the three is true is a matter of how far off things are, not whether
they are off at all.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model, log_mel, pick_peaks_suppressed  # noqa: E402
from diagnose_level import grid, nearest, uninherited  # noqa: E402


def align(reported: np.ndarray, reference: np.ndarray) -> dict:
    """How well two sets of beat times agree, in the terms a rhythm game cares about."""
    if reported.size == 0 or reference.size == 0:
        return {"coverage": 0.0, "precision": 0.0, "error": float("nan"), "rate": 0.0}

    error = nearest(reported, reference)

    return {
        # Coverage is counted from the reference's side: what share of the map's beats has a reported beat near it.
        "coverage": 100.0 * float((nearest(reference, reported) <= 30).mean()),
        "precision": 100.0 * float((error <= 30).mean()),
        "error": float(np.median(error)),
        "rate": reported.size / reference.size,
        "within_30": 100.0 * float((error <= 30).mean()),
    }


def infer(model, mels: np.ndarray, device, segment: int = 1024, step: int = 960):
    frames = mels.shape[0]
    probability = np.zeros(frames, dtype=np.float32)
    log_period = np.zeros(frames, dtype=np.float32)
    weight = np.zeros(frames, dtype=np.float32)

    for start in range(0, frames, step):
        end = min(frames, start + segment)
        window = mels[start:end]

        if window.shape[0] < 64:
            break

        with torch.no_grad():
            x = torch.from_numpy(window.T.astype(np.float32)).unsqueeze(0).to(device)
            logits, period_logits = model(x)
            probability[start:end] += torch.sigmoid(logits)[0].cpu().numpy()
            log_period[start:end] += period_logits[0].cpu().numpy()

        weight[start:end] += 1

    weight[weight == 0] = 1

    return probability / weight, np.exp(log_period / weight)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--ids", default="1335372,1533028,1192164")
    parser.add_argument("--tolerance", type=float, default=30.0)
    parser.add_argument("--share", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    rows = {r["id"]: r for r in csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t")}
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}

    print(f"tolerance {args.tolerance:.0f}ms, suppression share {args.share}, threshold {args.threshold}")
    print("")
    print(f"  {'track':>9} {'pairs':>28} {'coverage':>9} {'precision':>10} {'median err':>11} {'rate':>6}")

    totals: dict[str, list[tuple[float, float, float]]] = {}

    for track_id in args.ids.split(","):
        if track_id not in rows or track_id not in archives:
            print(f"  {track_id}: missing")
            continue

        cached = args.cache / f"{track_id}.npy"

        if cached.exists():
            mels = np.load(cached).astype(np.float32)
        else:
            samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
            mels = log_mel(samples).astype(np.float32)
            np.save(cached, mels.astype(np.float16))

        labels = np.fromfile(args.dataset / "labels" / f"{track_id}.i8", dtype=np.int8)
        label_ms = np.flatnonzero(labels > 0) / FPS * 1000.0
        until = len(labels) / FPS * 1000.0
        map_ms = grid(uninherited(archives[track_id]), until)
        map_ms = map_ms[map_ms <= until]

        probability, period = infer(model, mels, device)
        picked = pick_peaks_suppressed(probability, period, share=args.share,
                                       threshold=args.threshold) / FPS * 1000.0

        for name, reported, reference in (
            ("curve -> map", picked, map_ms),
            ("curve -> labels", picked, label_ms),
            ("labels -> map", label_ms, map_ms),
        ):
            result = align(reported, reference)
            totals.setdefault(name, []).append((result["coverage"], result["precision"], result["error"]))

        print(f"  {track_id:>9} {'curve -> map':>28} {align(picked, map_ms)['coverage']:>8.1f}% "
              f"{align(picked, map_ms)['precision']:>9.1f}% {align(picked, map_ms)['error']:>10.1f}ms "
              f"{align(picked, map_ms)['rate']:>6.2f}")
        print(f"  {'':>9} {'curve -> labels':>28} {align(picked, label_ms)['coverage']:>8.1f}% "
              f"{align(picked, label_ms)['precision']:>9.1f}% {align(picked, label_ms)['error']:>10.1f}ms "
              f"{align(picked, label_ms)['rate']:>6.2f}")
        print(f"  {'':>9} {'labels -> map':>28} {align(label_ms, map_ms)['coverage']:>8.1f}% "
              f"{align(label_ms, map_ms)['precision']:>9.1f}% {align(label_ms, map_ms)['error']:>10.1f}ms "
              f"{align(label_ms, map_ms)['rate']:>6.2f}")
        print("")

    print("averaged:")
    for name, values in totals.items():
        coverage = float(np.mean([v[0] for v in values]))
        precision = float(np.mean([v[1] for v in values]))
        error = float(np.nanmean([v[2] for v in values]))
        print(f"  {name:>16}: coverage {coverage:5.1f}%  precision {precision:5.1f}%  median error {error:5.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
