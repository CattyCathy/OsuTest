"""Reads one track with the trained detector and reports what it says, without any scoring in the way.

Written because the detector reported four times the map's beat count on a track whose labels are unambiguously at the
map's level: the labels for that track have a 660ms median gap and the model's own output came out at 160ms. Either the
model is not generalising to audio it trained on, or the way the audio is presented to it at inference differs from the
way it was presented during training, and those have different fixes.

Prints, for one track:

  the label gaps, which are what the model was asked to reproduce
  the probability curve's own statistics, because a curve that is saturated or flat is a model that is not reading
  the input, and a curve with a spike every 160ms is a model that is reading it and disagreeing
  the gaps between the picked peaks, which is what the detector reports
  the same again under a whole-track pass and under several window sizes, so that a difference between them is a
  windowing artefact rather than a property of the model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, read_track, log_mel, load_manifest, build_model, pick_peaks  # noqa: E402

import torch  # noqa: E402


def gaps_of(frames: np.ndarray) -> np.ndarray:
    return np.diff(np.sort(frames)) / FPS * 1000.0


def describe(name: str, frames: np.ndarray, limit: int = 8) -> None:
    if frames.size < 2:
        print(f"{name}: {frames.size} frames, no gaps")
        return

    gaps = gaps_of(frames)
    print(f"{name}: {frames.size} peaks, median gap {np.median(gaps):.0f}ms")

    buckets: dict[int, int] = {}

    for gap in gaps:
        buckets[int(gap // 40) * 40] = buckets.get(int(gap // 40) * 40, 0) + 1

    for bucket in sorted(buckets)[:limit]:
        print(f"    {bucket:5}ms {'#' * min(50, buckets[bucket])} {buckets[bucket]}")


def run(model, mels: np.ndarray, device, segment: int, step: int) -> np.ndarray:
    """Probabilities for a whole track, windowed into the model's expected input length."""
    frames = mels.shape[0]
    probability = np.zeros(frames, dtype=np.float32)

    if segment >= frames:
        with torch.no_grad():
            x = torch.from_numpy(mels.T.astype(np.float32)).unsqueeze(0).to(device)
            return torch.sigmoid(model(x))[0].cpu().numpy()

    starts = list(range(0, frames, step))
    weight = np.zeros(frames, dtype=np.float32)

    for start in starts:
        end = min(frames, start + segment)
        window = mels[start:end]

        if window.shape[0] < 64:
            break

        with torch.no_grad():
            x = torch.from_numpy(window.T.astype(np.float32)).unsqueeze(0).to(device)
            values = torch.sigmoid(model(x))[0].cpu().numpy()

        # Averaged over the overlap rather than concatenated. Concatenating over counts the shared frames and, worse,
        # puts a frame that one window saw at its edge next to a frame another window saw in its middle as if they came
        # from the same reading.
        probability[start:end] += values
        weight[start:end] += 1

    weight[weight == 0] = 1

    return probability / weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("track", type=str)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    track = next((t for t in load_manifest(args.dataset) if t.id == args.track), None)

    if track is None:
        print(f"no track {args.track}")
        return 1

    samples, labels = read_track(track)
    mels = log_mel(samples)

    print(f"track {track.id}: {labels.size} frames, {samples.size / 22050:.0f}s")
    print(f"  mel {mels.shape}")
    print("")

    describe("labels", np.flatnonzero(labels > 0.5))
    print("")

    for segment, step in ((1024, 960), (2048, 1984), (4096, 4032), (100000, 0)):
        probability = run(model, mels, device, segment, step)

        label = f"window {segment}" if segment < 100000 else "whole track"

        print(f"{label}: p mean {probability.mean():.3f}  max {probability.max():.3f}  "
              f"above 0.5 {100.0 * (probability > 0.5).mean():.1f}%  above 0.9 {100.0 * (probability > 0.9).mean():.1f}%")

        describe(f"  its peaks", pick_peaks(probability, radius=3, threshold=0.5))
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
