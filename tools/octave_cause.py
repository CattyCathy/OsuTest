"""Which of the two heads sets the metrical level, and what the level error actually is.

An octave error is different in kind from every other error in this reading and it is worth being precise about which
part produces it, because the two parts need opposite fixes. The period head is used for one thing - how far apart two
beats must be before both can be kept - so if the curve is at the right level and the period is at the wrong one, the
fault is a radius and the fix is in the period. If the curve itself peaks halfway between the map's beats, then the
level is baked into the beat classification and no amount of work on the period will move it.

The test separates them without either head's opinion. Take a passage the map holds at one tempo, take the map's own
beats, and ask what the beat curve reads on those beats against what it reads exactly halfway between them - which is
where a curve an octave too fine would be putting its own beats. Then do the same with the grid subdivided, which is
where a curve an octave too coarse would find a beat it had missed. Whichever grid the curve separates best is the
level the classification itself is at, and that is a fact about the model rather than about the period.

Reported per section, with the map's level named, so that a passage where the map is at 300 BPM and the detector is
convincingly at 150 can be told from one where the detector simply cannot separate either.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model  # noqa: E402
from diagnose_level import uninherited  # noqa: E402
from tempo_sections import sections as map_sections  # noqa: E402
from fit_grid import infer, peaks, smooth  # noqa: E402
from refine_choice import centroid  # noqa: E402


def grid(points: list[tuple[float, float]], until: float, scale: float) -> np.ndarray:
    out: list[float] = []

    for i, (time, length) in enumerate(points):
        step = length * scale

        if step <= 5:
            continue

        end = points[i + 1][0] if i + 1 < len(points) else max(until, time)

        while time < end and len(out) < 5_000_000:
            out.append(time)
            time += step

    return np.array(sorted(out))


def at(curve: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.interp(times, np.arange(curve.size) / FPS * 1000.0, curve)


def separation(curve: np.ndarray, beats: np.ndarray) -> float:
    """How much stronger the curve is on a grid than halfway between its points."""
    if beats.size < 8:
        return float("nan")

    halfway = (beats[:-1] + beats[1:]) / 2

    return float(np.median(at(curve, beats)) - np.median(at(curve, halfway)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--ids", default="1533028,67565,2155472,1633250,1335372,1850986")
    parser.add_argument("--long", type=float, default=10.0)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print("Is the beat curve itself at the map's level? Curve strength on a grid minus its strength halfway")
    print("between the grid's points, so a positive number means the curve has a beat there and everything")
    print("below zero means it has one in the gaps instead. The level with the largest number is the one the")
    print("classification is actually at.")
    print("")

    verdicts = {"map level": 0, "half the beat": 0, "double the beat": 0, "neither": 0}

    for track_id in args.ids.split(","):
        cached = args.cache / f"{track_id}.npy"
        archive = next(args.corpus.glob(f"{track_id} *.osz"), None)

        if not cached.exists() or archive is None:
            continue

        mels = np.load(cached).astype(np.float32)
        samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
        until = len(samples) / 22050.0 * 1000.0
        points = uninherited(archive)

        probability, period = infer(model, mels, device)
        period = smooth(period, 1.0) / FPS * 1000.0

        runs = [s for s in map_sections(points, until, 0.002) if s[1] - s[0] >= args.long * 1000.0]

        print(f"=== {track_id}")

        for start, end, length in sorted(runs, key=lambda s: s[1] - s[0], reverse=True)[:args.show]:
            # The map's own grid inside the section, at its level and at the octave either side of it.
            inside = [(t, l) for t, l in points if start - 1 <= t or True]
            section_points = [(max(t, start), l) for t, l in points if t < end]

            if not section_points:
                continue

            scores: dict[str, float] = {}

            for name, scale in (("map level", 1.0), ("half the beat", 0.5), ("double the beat", 2.0)):
                beats = grid(section_points, end, scale)
                beats = beats[(beats >= start) & (beats < end)]
                scores[name] = separation(probability, beats)

            section_period = period[int(start / 1000.0 * FPS):int(end / 1000.0 * FPS)]

            if section_period.size == 0:
                continue

            head = float(np.median(section_period))
            best = max(scores, key=lambda k: scores[k])

            if np.isnan(scores[best]) or scores[best] <= 0:
                verdicts["neither"] += 1
            else:
                verdicts[best] += 1

            print(f"   {start / 1000:7.1f}s..{end / 1000:7.1f}s  map {length:7.2f}ms ({60000 / length:6.1f} BPM)  "
                  + "  ".join(f"{name} {scores[name]:+6.3f}" for name in scores)
                  + f"   | period head {head:7.2f}ms ({60000 / head:6.1f} BPM)"
                  + f"  -> curve is at {best}")

        print("")

    print("sections by where the beat curve's own level is:")
    for name, count in verdicts.items():
        if count:
            print(f"   {name:>16}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
