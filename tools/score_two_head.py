"""Scores a trained detector against the maps, reading both of its heads.

The same job as score_detector.py, pointed at the two-head model and using the period it predicts to suppress its own
peaks. That suppression is the whole reason the period head exists: on its own the beat curve reports every onset the
music has, which came out at two to four times the map's beat count.

Reports the rate - reported beats over the map's beats - because that is the metrical level stated as one number, and
reports precision beside coverage because coverage on its own flatters a reading that scatters beats everywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import (  # noqa: E402
    FPS, read_track, log_mel, load_manifest, build_model, pick_peaks, pick_peaks_suppressed,
)
from score_detector import map_beats  # noqa: E402

import torch  # noqa: E402


def infer(model, mels: np.ndarray, device, segment: int = 1024, step: int = 960):
    """Both heads over a whole track, averaged over overlapping windows."""
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
    probability /= weight
    log_period /= weight

    return probability, np.exp(log_period)


def evaluate(reported: np.ndarray, reference: np.ndarray, seconds: float) -> dict:
    if reported.size == 0 or reference.size == 0:
        return {"rate": 0.0, "coverage": 0.0, "precision": 0.0, "median": 0.0, "count": int(reported.size)}

    order = np.argsort(reference)
    reference = reference[order]

    index = np.clip(np.searchsorted(reference, reported), 1, len(reference) - 1)
    nearest = np.minimum(np.abs(reported - reference[index - 1]),
                         np.abs(reported - reference[np.clip(index, 0, len(reference) - 1)]))

    covered = 0

    for beat in reference:
        position = np.searchsorted(reported, beat)
        low, high = max(0, position - 2), min(len(reported), position + 2)

        if high > low and np.abs(reported[low:high] - beat).min() <= 60:
            covered += 1

    return {
        "rate": (reported.size / seconds) / (reference.size / seconds),
        "coverage": 100.0 * covered / reference.size,
        "precision": 100.0 * float((nearest <= 60).sum()) / reported.size,
        "median": float(np.median(nearest)),
        "count": int(reported.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--share", type=float, default=0.6, help="suppression radius as a share of the predicted period")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    tracks = load_manifest(args.dataset)
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}
    shared = sorted(set(t.id for t in tracks) & set(archives))[:args.limit]

    print(f"scoring {len(shared)} tracks, suppression at {args.share} of the predicted period")
    print("")
    print(f"  {'track':>10} {'map':>6} {'pred':>6} | {'raw: n':>7} {'rate':>6} | "
          f"{'suppressed: n':>14} {'rate':>6} {'cov':>6} {'prec':>6} {'med':>7}")

    raw_rows, suppressed_rows = [], []

    for track_id in shared:
        track = next(t for t in tracks if t.id == track_id)

        samples, labels = read_track(track)
        seconds = len(labels) / FPS
        mels = log_mel(samples)

        reference = map_beats(archives[track_id], seconds * 1000)

        if reference.size < 16:
            continue

        probability, period = infer(model, mels, device)

        raw = pick_peaks(probability, radius=3, threshold=args.threshold) / FPS * 1000.0
        suppressed = pick_peaks_suppressed(probability, period, share=args.share,
                                           threshold=args.threshold) / FPS * 1000.0

        raw_result = evaluate(raw, reference, seconds)
        suppressed_result = evaluate(suppressed, reference, seconds)

        raw_rows.append(raw_result)
        suppressed_rows.append(suppressed_result)

        predicted = float(np.median(period)) / FPS * 1000.0
        map_period = float(np.median(np.diff(np.sort(reference))))

        print(f"  {track_id:>10} {map_period:>5.0f}ms {predicted:>5.0f}ms | {raw_result['count']:>7} "
              f"{raw_result['rate']:>6.2f} | {suppressed_result['count']:>14} {suppressed_result['rate']:>6.2f} "
              f"{suppressed_result['coverage']:>5.0f}% {suppressed_result['precision']:>5.0f}% "
              f"{suppressed_result['median']:>6.0f}ms")

    print("")

    for name, rows in (("raw peaks", raw_rows), ("period-suppressed", suppressed_rows)):
        if not rows:
            continue

        rates = np.array([r["rate"] for r in rows])
        coverage = np.array([r["coverage"] for r in rows])
        precision = np.array([r["precision"] for r in rows])
        median = np.array([r["median"] for r in rows])
        good = int(((rates > 0.75) & (rates < 1.33)).sum())

        print(f"  {name}:")
        print(f"    rate      median {np.median(rates):.2f}   within 0.75-1.33 of the map: {good}/{len(rates)}")
        print(f"    coverage  median {np.median(coverage):.0f}%")
        print(f"    precision median {np.median(precision):.0f}%")
        print(f"    error     median {np.median(median):.0f}ms")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
