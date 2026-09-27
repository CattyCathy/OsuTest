"""Asks where the detector loses the beat level, separating the model from the reading.

Three separations, because "the beats are wrong" has three different causes and they need different fixes:

  activation against the map grid   if the curve's own peaks do not sit on the map's beats, the model is wrong and no
                                    amount of post-processing helps.
  activation against its own labels if the curve matches the labels it was trained on, the model learned its target
                                    faithfully and the target is what is wrong.
  labels against the map grid       the export's own error, which is invisible from the player.

Run it on one track id. The audio and labels come from the exported dataset, the grid from the .osz, and the model from
model/detector.pt - the trained one, not Beat This!, so that what is being judged is this project's detector.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model, log_mel, pick_peaks_suppressed  # noqa: E402


def uninherited(osz: Path) -> list[tuple[float, float]]:
    with zipfile.ZipFile(osz) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".osu"))
        text = archive.read(name).decode("utf-8", "replace")

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

    return sorted(points)


def grid(timing: list[tuple[float, float]], until: float, scale: float = 1.0) -> np.ndarray:
    """The map's own beat times, in milliseconds, from its uninherited points."""
    out: list[float] = []

    for i, (time, length) in enumerate(timing):
        step = length * scale

        if step <= 5:
            continue

        end = timing[i + 1][0] if i + 1 < len(timing) else max(until, time)

        while time < end and len(out) < 5_000_000:
            out.append(time)
            time += step

    return np.array(sorted(out))


def nearest(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Distance from each point in source to the closest point in target, in the same units."""
    target = np.sort(target)

    if target.size == 0:
        return np.full(source.shape, np.inf)

    index = np.clip(np.searchsorted(target, source), 1, target.size - 1)

    return np.minimum(np.abs(source - target[index - 1]), np.abs(source - target[index]))


def report(name: str, distance: np.ndarray) -> None:
    if distance.size == 0:
        print(f"  {name}: nothing to measure")
        return

    print(f"  {name}: n {distance.size:5}  median {np.median(distance):6.1f}ms  "
          f"p90 {np.percentile(distance, 90):6.1f}  within 20ms {100 * (distance <= 20).mean():5.1f}%  "
          f"within 60ms {100 * (distance <= 60).mean():5.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="1335372")
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--segment", type=int, default=512)
    parser.add_argument("--share", type=float, default=0.6)
    parser.add_argument("--smoothing", type=float, default=1.0,
                        help="seconds either side for the median period smoothing the C# side applies")
    parser.add_argument("--window", type=float, default=60.0, help="seconds to print in detail, from 0 to this")
    args = parser.parse_args()

    samples = np.fromfile(args.dataset / "audio" / f"{args.id}.f32", dtype=np.float32)
    labels = np.fromfile(args.dataset / "labels" / f"{args.id}.i8", dtype=np.int8)
    osz = next(args.corpus.glob(f"{args.id} *.osz"))

    print(f"track {args.id}: {len(samples) / 22050:.1f}s audio, {len(labels)} label frames "
          f"({len(labels) / FPS:.1f}s), {int((labels > 0).sum())} labelled beats")

    mels = log_mel(samples).astype(np.float32)
    print(f"mel {mels.shape[0]} x {mels.shape[1]}")

    # The .oct file records the octave the exporter chose per frame, which is the thing being questioned.
    octave_path = args.dataset / "labels" / f"{args.id}.oct"
    octaves = np.fromfile(octave_path, dtype=np.int8) if octave_path.exists() else np.zeros(len(labels), dtype=np.int8)

    model = build_model(torch)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    frames = min(mels.shape[0], len(labels))
    probability = np.zeros(frames, dtype=np.float32)
    period_frames = np.zeros(frames, dtype=np.float32)

    hop = args.segment // 2
    starts = list(range(0, max(1, frames - args.segment), hop))
    starts.append(max(0, frames - args.segment))

    with torch.no_grad():
        for start in starts:
            end = min(frames, start + args.segment)
            start = max(0, end - args.segment)
            window = mels[start:end].T[None]

            if window.shape[2] < args.segment:
                window = np.pad(window, ((0, 0), (0, 0), (0, args.segment - window.shape[2])))

            logits, period_logits = model(torch.from_numpy(window))
            keep = end - start

            probability[start:end] = torch.sigmoid(logits)[0, :keep].numpy()
            period_frames[start:end] = torch.exp(period_logits)[0, :keep].numpy()

    period_ms = period_frames / FPS * 1000.0
    print(f"predicted period: median {np.median(period_ms):.0f}ms  "
          f"p10 {np.percentile(period_ms, 10):.0f}  p90 {np.percentile(period_ms, 90):.0f}")

    # Smoothing, as the C# reading does, before the period is used to suppress.
    if args.smoothing > 0:
        half = int(round(args.smoothing * FPS))
        smoothed = np.copy(period_frames)

        for i in range(frames):
            low = max(0, i - half)
            high = min(frames, i + half + 1)
            smoothed[i] = np.median(period_frames[low:high])

        period_frames = smoothed
        print(f"smoothed period:  median {np.median(period_frames) / FPS * 1000:.0f}ms")

    picked = pick_peaks_suppressed(probability, period_frames, share=args.share)
    picked_ms = picked / FPS * 1000.0
    label_ms = np.flatnonzero(labels > 0) / FPS * 1000.0

    until = frames * 1000.0 / FPS
    timing = uninherited(osz)
    map_ms = grid(timing, until)
    map_ms = map_ms[map_ms <= until]

    print("")
    print(f"map has {len(timing)} uninherited points, {len(map_ms)} beats over {until / 1000:.0f}s")

    print("")
    print("separating the causes:")
    report("peaks  -> map grid ", nearest(picked_ms, map_ms))
    report("peaks  -> own labels", nearest(picked_ms, label_ms))
    report("labels -> map grid  ", nearest(label_ms, map_ms))
    report("map    -> labels     ", nearest(map_ms, label_ms))

    print("")
    print("the raw curve, read with no peak picking at all:")
    top = np.argsort(probability)[::-1]
    strong = np.sort(top[: max(1, frames // 50)])
    strong_ms = strong / FPS * 1000.0
    report("strongest 2%     -> map", nearest(strong_ms, map_ms))
    report("strongest 2% -> labels  ", nearest(strong_ms, label_ms))

    # A peak-picking-free statement of level: at the curve's local maxima, how much higher is it on a map beat than
    # halfway between two? If the two are the same the curve genuinely does not know the level.
    at_beat = np.interp(map_ms, np.arange(frames) / FPS * 1000.0, probability)
    half = (map_ms[:-1] + map_ms[1:]) / 2
    at_half = np.interp(half, np.arange(frames) / FPS * 1000.0, probability)
    print("")
    print(f"curve at map beats:  median {np.median(at_beat):.3f}   "
          f"halfway between:  median {np.median(at_half):.3f}   ratio {np.median(at_beat) / max(1e-6, np.median(at_half)):.2f}")

    # Where the octave the exporter chose disagreed with the octave the map's local beat length implies.
    beat_octave = np.zeros(frames, dtype=np.int8)
    for i, (time, length) in enumerate(timing):
        end = timing[i + 1][0] if i + 1 < len(timing) else until
        low = max(0, int(time / 1000 * FPS))
        high = min(frames, int(end / 1000 * FPS))
        beat_octave[low:high] = i

    print("")
    print("the window to look at by hand:")
    seconds = args.window
    print("   time     curve  period     peak?   map beat?  label?")
    for frame in range(0, min(frames, int(seconds * FPS))):
        near_peak = np.abs(picked - frame).min() <= 1 if picked.size else False
        near_map = np.abs(map_ms - frame / FPS * 1000).min() <= 15 if map_ms.size else False
        near_label = labels[frame] > 0
        marker = "".join(["P" if near_peak else ".", "M" if near_map else ".", "L" if near_label else "."])

        if frame % 10 == 0 or near_peak or near_label:
            print(f"  {frame / FPS:7.2f}  {probability[frame]:6.3f}  {period_frames[frame] / FPS * 1000:6.0f}ms   {marker}")

    # The gap sequences side by side, which is the one view that says whether a level is being missed. A run of map
    # gaps twice a run of label gaps is the label sitting an octave below; the reverse is it sitting an octave above.
    print("")
    print("gap sequences, in milliseconds, from the start:")
    map_gaps = np.diff(map_ms)
    label_gaps = np.diff(label_ms)
    picked_gaps = np.diff(picked_ms)

    for name, gaps in (("map   ", map_gaps), ("label ", label_gaps), ("peaks ", picked_gaps)):
        shown = " ".join(f"{g:5.0f}" for g in gaps[:40])
        print(f"  {name}: {shown}")

    # The local ratio of the map's gap to the labels' gap, which is the octave disagreement made visible. One means
    # the labels are at the map's level, two that they are an octave below it.
    print("")
    ratios = []
    for start in range(0, int(until), 20_000):
        window_map = map_gaps[(map_ms[:-1] >= start) & (map_ms[:-1] < start + 20_000)]
        window_label = label_gaps[(label_ms[:-1] >= start) & (label_ms[:-1] < start + 20_000)]

        if window_map.size < 8 or window_label.size < 8:
            continue

        ratio = np.median(window_map) / np.median(window_label)
        ratios.append(ratio)
        print(f"  {start / 1000:6.0f}s  map {np.median(window_map):6.0f}ms  label {np.median(window_label):6.0f}ms  "
              f"map/label {ratio:4.2f}   map beats {window_map.size:4}  label beats {window_label.size:4}")

    if ratios:
        ratios = np.array(ratios)
        print(f"  map/label ratio over the track: median {np.median(ratios):.2f}  "
              f"near 1 ({int(((ratios > 0.8) & (ratios < 1.25)).sum())}/{ratios.size})  "
              f"near 2 ({int(((ratios > 1.6) & (ratios < 2.5)).sum())}/{ratios.size})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

