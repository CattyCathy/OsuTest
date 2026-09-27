"""Fits a phase-locked grid to the detector's beats, and measures what that buys.

The detector reports a beat per frame that clears its own confidence, each placed where the curve peaks and each
independent of the others. Measured over the tracks that prompted this, one beat in five is more than 30ms from the
map's grid and the middle error is 16 to 22ms - and the frame rate is a twentieth of a second, so part of that is the
grid the answer is rounded onto rather than the answer.

Both of those have the same fix and it is not a bigger model. A rhythm game is not asking where each beat is; it is
asking for a pulse, and a pulse is one period and one phase. So the beats are fitted rather than taken:

  sub-frame peaks   the curve is smooth over several frames, so a parabola through a peak's neighbours puts the peak
                    between frames instead of on one. Worth up to half a frame, or 10ms - a third of the budget.

  phase-locked fit  over a window of beats, the report is modelled as an index times a period plus a phase, fitted by
                    weighted least squares. Isolated errors stop being displacements of the grid and become outliers
                    the fit absorbs, which is exactly what a pulse is: the beats agree with each other.

The window is what makes it honest. Too long and a real tempo change is averaged away; too short and there is nothing
to fit. Both ends are reported here rather than argued about.
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

MILLISECONDS = 1000.0 / FPS


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


def refine(probability: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """A parabola through each peak and its neighbours, evaluated at its own vertex.

    The curve is a blur of a target seven tenths of a frame wide, so over three frames it is very nearly a parabola and
    the vertex is where the beat actually is rather than the frame it rounded to. The vertex is clamped to within half a
    frame of the peak, because a peak on a plateau or on a straight run would otherwise be moved arbitrarily far.
    """
    out = []

    for index in frames:
        index = int(index)

        if index <= 0 or index + 1 >= probability.size:
            out.append(float(index))
            continue

        left, middle, right = float(probability[index - 1]), float(probability[index]), float(probability[index + 1])
        denominator = left - 2 * middle + right
        offset = 0.0

        if abs(denominator) > 1e-9:
            offset = 0.5 * (left - right) / denominator
            offset = max(-0.5, min(0.5, offset))

        out.append(index + offset)

    return np.array(out) / FPS * 1000.0


def peaks(probability: np.ndarray, periods: np.ndarray, share: float, threshold: float) -> np.ndarray:
    """The same strongest-first suppression the library uses, in milliseconds."""
    candidates = [(float(probability[i]), int(i)) for i in np.flatnonzero(probability >= threshold)]
    candidates.sort(reverse=True)

    chosen: list[int] = []
    taken = np.zeros(probability.size, dtype=bool)

    for _, index in candidates:
        if taken[index]:
            continue

        chosen.append(index)
        radius = int(max(1, share * float(periods[index])))

        for other in range(max(0, index - radius + 1), min(probability.size, index + radius)):
            taken[other] = True

    return np.array(sorted(chosen), dtype=np.int64)


def lock(beats: np.ndarray, weights: np.ndarray, window: int) -> np.ndarray:
    """A grid fitted through the beats, one fit per beat over its neighbours.

    Weighted least squares of time against beat index. The indices are what make this a pulse: they say the beats are
    consecutive, so the fit is of a spacing rather than of a trend, and an error in one beat is pulled towards agreement
    with its neighbours instead of being reproduced.

    The phase is carried through the fit by predicting each beat from the previous fit's period and then re-fitting
    around that prediction, which is what stops a window from wrapping a whole period when the tempo is misjudged.
    """
    count = beats.size

    if count < 3:
        return beats

    fitted = np.copy(beats)
    half = max(1, window // 2)

    for i in range(count):
        low = max(0, i - half)
        high = min(count, i + half + 1)

        if high - low < 3:
            continue

        t = beats[low:high]
        w = weights[low:high]

        # Indices centred on the beat being fitted, so that the fit's intercept is that beat rather than the window's
        # start, which keeps the arithmetic away from large numbers.
        index = np.arange(low, high, dtype=np.float64) - i

        total = w.sum()

        if total <= 0:
            continue

        mean_t = (w * t).sum() / total
        mean_i = (w * index).sum() / total

        covariance = (w * (index - mean_i) * (t - mean_t)).sum()
        variance = (w * (index - mean_i) ** 2).sum()

        if variance <= 0:
            continue

        period = covariance / variance

        # A fit that comes out at a nonsense spacing is a window with no pulse in it, and is left as it was.
        if not (0.5 * np.median(np.diff(t)) < period < 2.0 * np.median(np.diff(t))):
            continue

        fitted[i] = mean_t - period * mean_i

    return fitted


def report(name: str, reported: np.ndarray, reference: np.ndarray, tolerance: float) -> tuple[float, float, float]:
    if reported.size == 0 or reference.size == 0:
        return 0.0, 0.0, float("nan")

    error = nearest(reported, reference)
    coverage = 100.0 * float((nearest(reference, reported) <= tolerance).mean())
    precision = 100.0 * float((error <= tolerance).mean())
    median = float(np.median(error))

    return coverage, precision, median


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--ids", default="1335372,1533028,1192164")
    parser.add_argument("--share", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--windows", default="2,4,8,16,32,64")
    parser.add_argument("--tolerance", type=float, default=30.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    rows = {r["id"]: r for r in csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t")}
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}
    windows = [int(w) for w in args.windows.split(",")]

    print(f"tolerance {args.tolerance:.0f}ms, share {args.share}, threshold {args.threshold}, smoothing {args.smoothing}s")

    # The reference is the map's grid, at the level the corpus labels settled on, because that is the pulse a player is
    # being asked to follow and it is not the training target - scoring against the target would only say whether the
    # model learned its teacher.
    totals: dict[str, list] = {}

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

        # The frame count comes from the audio rather than from the labels, so that this can be run while the labels are
        # being rebuilt - which is exactly when the reading wants judging.
        label_path = args.dataset / "labels" / f"{track_id}.i8"

        if label_path.exists():
            until = len(np.fromfile(label_path, dtype=np.int8)) / FPS * 1000.0
        else:
            samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
            until = len(samples) / 22050.0 * 1000.0

        map_ms = grid(uninherited(archives[track_id]), until)
        map_ms = map_ms[map_ms <= until]

        probability, period = infer(model, mels, device)
        period = smooth(period, args.smoothing)

        frames = peaks(probability, period, args.share, args.threshold)
        on_frame = frames / FPS * 1000.0
        sub_frame = refine(probability, frames)
        weights = probability[frames].astype(np.float64)

        print("")
        print(f"{track_id}: map {map_ms.size} beats, curve reports {frames.size}")

        for name, reported in (("frame-quantised", on_frame), ("sub-frame", sub_frame)):
            coverage, precision, median = report(name, reported, map_ms, args.tolerance)
            totals.setdefault(name, []).append((coverage, precision, median))
            print(f"  {name:>16}: coverage {coverage:5.1f}%  precision {precision:5.1f}%  median error {median:5.1f}ms")

        for window in windows:
            locked = lock(sub_frame, weights, window)
            coverage, precision, median = report("locked", locked, map_ms, args.tolerance)
            totals.setdefault(f"locked {window}", []).append((coverage, precision, median))
            print(f"  {'locked ' + str(window):>16}: coverage {coverage:5.1f}%  precision {precision:5.1f}%  "
                  f"median error {median:5.1f}ms")

    print("")
    print("averaged over the tracks:")
    for name, values in totals.items():
        coverage = float(np.mean([v[0] for v in values]))
        precision = float(np.mean([v[1] for v in values]))
        median = float(np.nanmean([v[2] for v in values]))
        print(f"  {name:>16}: coverage {coverage:5.1f}%  precision {precision:5.1f}%  median error {median:5.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
