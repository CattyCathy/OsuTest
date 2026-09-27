"""Fit the detector's beats as constant-tempo sections, and measure the drift that comes with it.

A grid is a list of sections with a tempo each, not a list of beats, and fitting one is a different bargain from
placing every beat: a section's tempo is estimated from every beat in it, so a handful of misplaced beats stop moving
the grid at all, and in exchange the whole section is wrong together if the tempo is wrong. How wrong that is grows
with the section, which is the thing to measure rather than assume - a beat placed 20ms out is 20ms out, but a tempo a
per cent out is a whole beat out by the end of a hundred seconds.

So both numbers are reported per section: how many of the map's beats the fitted grid covers, and how far the grid has
drifted from the map by the section's end. A scheme that covers as many beats as the raw reading but ends a long
section half a beat out has traded one fault for a worse one.
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


def sections_of(beats: np.ndarray, periods: np.ndarray, share: float = 0.12,
                shortest: int = 6) -> list[tuple[int, int, float]]:
    """Split the beats into runs whose local tempo agrees, as (first, last, period in ms).

    The tempo at a beat is the middle of the periods the model predicts around it, which is already smoothed, and a run
    continues while the next beat's tempo is within the share of the run's. The share is generous because the periods
    are noisy frame by frame and what is being looked for is a change of section, not a change of frame.
    """
    out: list[tuple[int, int, float]] = []
    start = 0

    for i in range(1, len(beats)):
        current = float(np.median(periods[start:i + 1]))

        if current <= 0:
            continue

        if abs(float(periods[i]) - current) > share * current:
            if i - start >= shortest:
                out.append((start, i - 1, current))

            start = i

    if len(beats) - start >= shortest:
        out.append((start, len(beats) - 1, float(np.median(periods[start:]))))

    return out


def fit(beats: np.ndarray, first: int, last: int) -> tuple[float, float, float]:
    """Least squares of beat time against index, then the residual scatter.

    The index is what makes this a pulse: consecutive beats are asserted to be one period apart, so the fit is of a
    spacing rather than of a trend. The scatter is returned because it says whether the section really is one tempo -
    a run of beats that only fits a line loosely is a run where the tempo is moving and one number is the wrong shape
    for it.
    """
    t = beats[first:last + 1].astype(np.float64)
    index = np.arange(t.size, dtype=np.float64)
    period, intercept = np.polyfit(index, t, 1)
    residual = t - (period * index + intercept)

    return float(period), float(intercept), float(np.std(residual))


def fitted_grid(beats: np.ndarray, periods: np.ndarray, until: float, share: float, shortest: int,
                fill: bool) -> np.ndarray:
    """The grid the sections imply, from the first beat to the end of the track."""
    out: list[float] = []
    runs = sections_of(beats, periods, share, shortest)

    for first, last, _ in runs:
        period, intercept, _ = fit(beats, first, last)
        start = period * first + intercept
        end = period * (last + 1) + intercept

        # A section is extended backwards to the one before it and forwards to the one after, so that the grid is
        # unbroken. This is the choice that matters when the beats are sparse: a gap in the reading is not a gap in the
        # music, and leaving it unfilled is what makes a player see the pulse stop.
        if fill:
            if out:
                start = max(start, out[-1] + period)

            stop = min(until, end) if last + 1 < len(beats) else until
        else:
            stop = min(until, end)

        time = start

        while time <= stop:
            if time >= 0:
                out.append(time)

            time += period

    return np.array(sorted(out))


def measure(reported: np.ndarray, reference: np.ndarray, tolerance: float) -> tuple[float, float, float]:
    if reported.size == 0 or reference.size == 0:
        return 0.0, 0.0, float("nan")

    error = nearest(reported, reference)

    return (100.0 * float((nearest(reference, reported) <= tolerance).mean()),
            100.0 * float((error <= tolerance).mean()),
            float(np.median(error)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--ids", default="1335372,1533028,2155472,1633250,1192164,1850986,67565")
    parser.add_argument("--share", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--split", type=float, default=0.12, help="tempo change that starts a new section")
    parser.add_argument("--shortest", type=int, default=6, help="beats a section needs to be fitted")
    parser.add_argument("--tolerance", type=float, default=30.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    rows = {r["id"]: r for r in csv.DictReader((args.dataset / "manifest.tsv").open(), delimiter="\t")}
    archives = {p.name.split(" ")[0]: p for p in args.corpus.glob("*.osz")}

    totals: dict[str, list] = {}

    for track_id in args.ids.split(","):
        if track_id not in rows or track_id not in archives:
            continue

        cached = args.cache / f"{track_id}.npy"

        if cached.exists():
            mels = np.load(cached).astype(np.float32)
        else:
            samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
            mels = log_mel(samples).astype(np.float32)

        samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
        until = len(samples) / 22050.0 * 1000.0
        map_ms = grid(uninherited(archives[track_id]), until)
        map_ms = map_ms[map_ms <= until]

        probability, period = infer(model, mels, device)
        period = smooth(period, args.smoothing)
        frames = peaks(probability, period, args.share, args.threshold)
        beats = centroid(probability, frames, 2)
        periods = period[frames] / FPS * 1000.0

        print("")
        print(f"=== {track_id}: map {map_ms.size} beats, detector {beats.size}")

        for name, reported in (
            ("raw beats", beats),
            ("sections", fitted_grid(beats, periods, until, args.split, args.shortest, False)),
            ("sections filled", fitted_grid(beats, periods, until, args.split, args.shortest, True)),
        ):
            coverage, precision, error = measure(reported, map_ms, args.tolerance)
            totals.setdefault(name, []).append((coverage, precision, error))
            print(f"  {name:>17}: n {reported.size:5}  coverage {coverage:5.1f}%  precision {precision:5.1f}%  "
                  f"median error {error:5.1f}ms")

        # The drift, which is the fault this scheme introduces and the raw reading does not have. Measured over each
        # fitted section as how far the grid is from the map at the section's own end.
        runs = sections_of(beats, periods, args.split, args.shortest)
        worst = 0.0

        for first, last, _ in runs:
            period_ms, intercept, scatter = fit(beats, first, last)
            start = period_ms * first + intercept
            end = period_ms * (last + 1) + intercept
            span = (end - start) / 1000.0

            # How far the map's own beat nearest the section's end is from the grid there.
            if map_ms.size:
                drift = float(nearest(np.array([end]), map_ms)[0])
                worst = max(worst, drift / max(1.0, span))

        print(f"  sections: {len(runs)}  worst drift {worst * 100:.2f}ms per second  "
              f"(which is {worst * 100 * 60:.1f}ms a minute)")

    print("")
    print(f"median over {len(next(iter(totals.values())))} tracks, at {args.tolerance:.0f}ms:")
    for name, values in totals.items():
        coverage = float(np.mean([v[0] for v in values]))
        precision = float(np.mean([v[1] for v in values]))
        error = float(np.nanmean([v[2] for v in values]))
        print(f"  {name:>17}: coverage {coverage:5.1f}%  precision {precision:5.1f}%  median error {error:5.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
