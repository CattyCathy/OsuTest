"""Sweeps the two numbers in the reading, before any retraining.

The trained detector's raw curve already lands on the maps' beats - on the track that prompted this, the strongest two
per cent of its frames sit a median 12ms from the map's grid and 76% of them are inside 60ms. What the player gets is
27ms and 57%, because the policy that turns that curve into beats gives most of it back.

So the policy is measured first. It is a threshold and a suppression share, and sweeping them is minutes where a
retrain is hours and a day. If a setting recovers the curve's own accuracy then the model is not what needs changing.

Reports per (threshold, share) the median error, coverage and precision against the map, so that a setting which is
accurate but sparse is not mistaken for a good one.
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
from diagnose_level import nearest, uninherited, grid  # noqa: E402


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


def smooth(period: np.ndarray, seconds: float) -> np.ndarray:
    if seconds <= 0:
        return period

    frames = period.size
    half = int(round(seconds * FPS))
    out = np.copy(period)

    for i in range(frames):
        out[i] = np.median(period[max(0, i - half):min(frames, i + half + 1)])

    return out


def evaluate(reported: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    """Coverage, precision and median error of reported beats against a reference grid."""
    if reported.size == 0 or reference.size == 0:
        return 0.0, 0.0, float("nan")

    covered = int((nearest(reference, reported) <= 60).sum())
    precision = float((nearest(reported, reference) <= 60).mean())

    return 100.0 * covered / reference.size, 100.0 * precision, float(np.median(nearest(reported, reference)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--ids", default="1335372", help="comma separated; the tracks that prompted this")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--shares", default="0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--smoothing", default="0,1")
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"torch on {device}")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    shares = [float(s) for s in args.shares.split(",")]
    smoothings = [float(s) for s in args.smoothing.split(",")]
    ids = args.ids.split(",")

    rows = {r["id"]: r for r in csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t")}
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}
    args.cache.mkdir(parents=True, exist_ok=True)

    results: dict[tuple[float, float, float], list[tuple[float, float, float]]] = {}

    for track_id in ids:
        row = rows.get(track_id)

        if row is None or track_id not in archives:
            print(f"  {track_id}: not in the manifest or the corpus, skipped")
            continue

        cached = args.cache / f"{track_id}.npy"

        if cached.exists():
            mels = np.load(cached).astype(np.float32)
        else:
            samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
            mels = log_mel(samples).astype(np.float32)

        labels = np.fromfile(args.dataset / "labels" / f"{track_id}.i8", dtype=np.int8)
        until = len(labels) / FPS * 1000.0
        map_ms = grid(uninherited(archives[track_id]), until)
        map_ms = map_ms[map_ms <= until]

        frames = min(mels.shape[0], len(labels))
        probability, period = infer(model, mels, device)

        # The label grid beside the map grid, because the labels are what the model was trained to produce and the
        # question is whether it is reproducing them or the map.
        label_ms = np.flatnonzero(labels > 0) / FPS * 1000.0
        print("")
        print(f"{track_id}: map {len(map_ms)} beats (median gap {np.median(np.diff(map_ms)):.0f}ms), "
              f"labels {len(label_ms)} (median gap {np.median(np.diff(label_ms)):.0f}ms), "
              f"curve {frames} frames")

        for smoothing in smoothings:
            used = smooth(period, smoothing)

            for threshold in thresholds:
                for share in shares:
                    picked = pick_peaks_suppressed(probability, used, share=share, threshold=threshold) / FPS * 1000.0
                    results.setdefault((smoothing, threshold, share), []).append(
                        evaluate(picked, map_ms))

    print("")
    print("against the map grid, averaged over the tracks asked for:")
    print(f"  {'smooth':>6} {'thresh':>6} {'share':>6} | {'coverage':>8} {'precision':>9} {'median err':>10}")

    best = None

    for (smoothing, threshold, share), values in sorted(results.items()):
        coverage = float(np.mean([v[0] for v in values]))
        precision = float(np.mean([v[1] for v in values]))
        error = float(np.nanmean([v[2] for v in values]))
        print(f"  {smoothing:>6.1f} {threshold:>6.2f} {share:>6.2f} | {coverage:>7.1f}% {precision:>8.1f}% "
              f"{error:>9.1f}ms")

        # Ranked on coverage at a fixed accuracy, because a setting cannot be bought with precision alone: reporting
        # one beat in ten with perfect accuracy is not a detector.
        score = coverage if error <= 60 else 0.0

        if best is None or score > best[0]:
            best = (score, smoothing, threshold, share, coverage, precision, error)

    print("")
    print(f"  best by coverage where the median error is within a sixtieth of a second: "
          f"smoothing {best[1]:.1f}s threshold {best[2]:.2f} share {best[3]:.2f} -> "
          f"coverage {best[4]:.1f}% precision {best[5]:.1f}% error {best[6]:.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
