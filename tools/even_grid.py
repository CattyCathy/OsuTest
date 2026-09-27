"""Does laying each tempo passage out as an even pulse give a steadier grid without losing accuracy?

The detector's tempo is right and its beats are not. On the reference track's opening passage the map holds 417ms, the
detector's middle gap there is 411ms - the right tempo to within a per cent - and its actual gaps run from 311 to 626,
because it inserts a beat every twenty or thirty. So a display driven from those beats is visibly uneven through music
that is perfectly even, which is the fault this is for.

Two things are reported and they pull against each other. The steadiness is how constant a gap is inside one passage,
which is what the eye reads and what should come out near zero. The accuracy is how far the beats sit from the map's
own, which can only get worse: snapping to a passage's average tempo moves the beats that were in the right place. What
matters is whether the accuracy loss is small against how much steadiness is bought, and whether coverage is kept - a
passage laid out from a wrong period would drop beats rather than move them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model  # noqa: E402
from diagnose_level import uninherited, grid, nearest  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402
from refine_choice import centroid  # noqa: E402
from sections_persist import analyse, robust  # noqa: E402


def locked(beats: np.ndarray, periods: np.ndarray, gain: float = 0.05) -> np.ndarray:
    """A pulse that follows the beats, corrected a little at each one rather than chained from the last.

    Three ways of laying out an even pulse were tried before this and all three failed the same way, which is worth
    recording because it is the whole difficulty. Snapping each passage to its first beat, fitting a line through the
    passage, and stepping from each beat using the middle of the gaps around it all gave an exactly even pulse and moved
    the middle distance from a map beat from eleven milliseconds to seventy-eight or more, taking the share within sixty
    milliseconds from eighty-two per cent to about forty.

    The reason is arithmetic and it cannot be tuned away. A grid extended by adding a period accumulates the error in
    that period: at four hundred milliseconds a beat, an error of a fifth of one per cent is a hundred milliseconds by
    the hundred and twenty-fifth beat, which is a quarter of a beat - the point at which the reading is halfway between
    the beats and no better than a guess. A period good to a fifth of a per cent is what a section's own beats give, so
    a chained grid is always a quarter of a beat out by the end of a section, whatever the period is measured from.

    The beats themselves do not have this problem, and that is the observation the fix rests on: each one is placed
    from the audio to within a few milliseconds and nothing accumulates. So the pulse is corrected towards each beat as
    it arrives, a share of the way, which keeps it locked to the music and stops the error growing. It is the standard
    answer to exactly this - a phase-locked loop - and the gain is the trade: too little and the pulse takes the jitter
    of the beats it is following, too much and it is a chained grid again.
    """
    out = np.zeros(beats.size)
    out[0] = beats[0]
    period = float(periods[0]) if periods.size else 400.0

    for i in range(1, beats.size):
        local = float(periods[min(i, periods.size - 1)]) if periods.size else period

        if local > 0:
            period = (1 - gain) * period + gain * local

        out[i] = out[i - 1] + period

        # Corrected towards the beat by a share of how far the pulse has drifted from it. A share rather than all of it
        # because the beat carries the placement error being corrected.
        error = beats[i] - out[i]

        if abs(error) < period * 0.5:
            out[i] += gain * error

    return out


def steadiness(beats: np.ndarray, at: float, span: float) -> tuple[float, float]:
    """The spread of gaps inside a time window, and how many beats are in it."""
    inside = beats[(beats >= at) & (beats < at + span)]

    if inside.size < 3:
        return float("nan"), float(inside.size)

    gaps = np.diff(inside)

    return float(np.percentile(gaps, 90) - np.percentile(gaps, 10)), float(inside.size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model-v5\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--ids", default="1335372,1023679,1192164,67565,2155472,1633250,1850986")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--share", type=float, default=0.06)
    parser.add_argument("--persist", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print(f"  {'track':>9} | {'beats':>6} {'even':>6} | {'gap spread as found':>19} {'laid out':>9} | "
          f"{'median err as found':>19} {'laid out':>9} | {'within 60ms':>17}")

    totals: dict[str, list] = {"spread before": [], "spread after": [], "err before": [], "err after": [],
                               "cov before": [], "cov after": []}

    for track_id in args.ids.split(","):
        cached = args.cache / f"{track_id}.npy"
        archive = next(args.corpus.glob(f"{track_id} *.osz"), None)

        if not cached.exists() or archive is None:
            continue

        mels = np.load(cached).astype(np.float32)
        samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
        until = len(samples) / 22050.0 * 1000.0
        mapg = grid(uninherited(archive), until)
        mapg = mapg[mapg <= until]

        probability, period = infer(model, mels, device)
        period = smooth(period, 1.0)
        frames = peaks(probability, period, 0.6, 0.5)
        beats = centroid(probability, frames, 2)

        gaps = robust(np.diff(beats))
        runs = analyse(gaps, args.window, args.share, args.persist, 16)
        periods = [float(np.median(gaps[a:b])) for a, b in runs]
        laid = even(beats, runs, periods)

        # Steadiness over the opening, which is the passage the map holds at one tempo and the one a listener judges.
        before_spread, _ = steadiness(beats, 20000, 14000)
        after_spread, _ = steadiness(laid, 20000, 14000)

        err_before = nearest(beats, mapg)
        err_after = nearest(laid, mapg)
        cov_before = 100 * float((nearest(mapg, beats) <= 60).mean())
        cov_after = 100 * float((nearest(mapg, laid) <= 60).mean())

        totals["spread before"].append(before_spread)
        totals["spread after"].append(after_spread)
        totals["err before"].append(float(np.median(err_before)))
        totals["err after"].append(float(np.median(err_after)))
        totals["cov before"].append(cov_before)
        totals["cov after"].append(cov_after)

        print(f"  {track_id:>9} | {beats.size:>6} {laid.size:>6} | {before_spread:>18.0f}ms {after_spread:>8.0f}ms | "
              f"{np.median(err_before):>18.1f}ms {np.median(err_after):>8.1f}ms | "
              f"{cov_before:>7.1f}% -> {cov_after:>5.1f}%")

    print("")
    print("medians over the tracks (gap spread is p90-p10 inside a 14s window the map holds at one tempo):")
    print(f"  gap spread as found {np.nanmedian(totals['spread before']):7.0f}ms   laid out {np.nanmedian(totals['spread after']):7.0f}ms")
    print(f"  median error         {np.nanmedian(totals['err before']):7.1f}ms          {np.nanmedian(totals['err after']):7.1f}ms")
    print(f"  coverage within 60ms {np.nanmedian(totals['cov before']):7.1f}%          {np.nanmedian(totals['cov after']):7.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
