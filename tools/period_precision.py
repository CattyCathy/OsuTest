"""How precise the period head actually is, since a constant-tempo section rests entirely on it.

A section of a grid is one number, and the number is a tempo, so the question a section-based reading has to answer is
not where a beat is but how accurately the model can name a tempo. The two failures are different sizes: a beat placed
twenty milliseconds out is twenty milliseconds out and stays there, where a tempo half a per cent out is a whole beat
out by the end of a hundred seconds. So the error is reported as a share of the period as well as in milliseconds, and
beside it the drift that share implies over a section.

Measured only where the map is holding one tempo, because that is the only place a single number is the right shape
for the answer. Over a passage the map itself changes tempo in, the head is being asked for an average and the error
there says nothing about its precision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_detector import FPS, build_model, log_mel  # noqa: E402
from diagnose_level import uninherited  # noqa: E402
from tempo_sections import sections as map_sections  # noqa: E402
from fit_grid import infer, smooth  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--corpus", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\train-corpus"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model\detector.pt"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    parser.add_argument("--ids", default="1335372,1533028,2155472,1633250,1192164,1023679,1850986,67565,594884,875117")
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--long", type=float, default=5.0)
    parser.add_argument("--show", type=int, default=6, help="sections per track to print")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print(f"period head against the map, over sections the map holds at one tempo for {args.long:.0f}s or more")
    print("")

    shares: list[float] = []

    for track_id in args.ids.split(","):
        cached = args.cache / f"{track_id}.npy"

        if not cached.exists():
            print(f"  {track_id}: no cached frontend, skipped")
            continue

        archive = next(args.corpus.glob(f"{track_id} *.osz"), None)

        if archive is None:
            continue

        mels = np.load(cached).astype(np.float32)
        samples = np.fromfile(args.dataset / "audio" / f"{track_id}.f32", dtype=np.float32)
        until = len(samples) / 22050.0 * 1000.0
        points = uninherited(archive)

        # A tight merge, so that only genuinely the same beat length forms a section. A loose one would join passages
        # the map changes tempo across and the head would be scored for not averaging them.
        runs = [s for s in map_sections(points, until, 0.002) if s[1] - s[0] >= args.long * 1000.0]

        if not runs:
            continue

        probability, period = infer(model, mels, device)
        period = smooth(period, args.smoothing) / FPS * 1000.0

        print(f"=== {track_id}: {len(runs)} sections the map holds at one tempo")

        for start, end, length in sorted(runs, key=lambda s: s[1] - s[0], reverse=True)[:args.show]:
            low = int(start / 1000.0 * FPS)
            high = min(period.size, int(end / 1000.0 * FPS))

            if high - low < 50:
                continue

            window = period[low:high]
            middle = float(np.median(window))
            share = (middle - length) / length

            shares.append(share)

            print(f"   {start / 1000:7.1f}s..{end / 1000:7.1f}s ({(end - start) / 1000:6.1f}s)  "
                  f"map {length:8.2f}ms ({60000 / length:6.1f} BPM)  head {middle:8.2f}ms ({60000 / middle:6.1f} BPM)  "
                  f"error {100 * share:+7.3f}%  drift over the section {abs(share) * (end - start):6.0f}ms")

            # The scatter is reported beside the middle because a head that is right on average but wanders frame to
            # frame is a head whose tempo cannot be held constant, however good its average is.
            print(f"      p10 {np.percentile(window, 10):8.2f}ms  p90 {np.percentile(window, 90):8.2f}ms  "
                  f"which is {100 * (np.percentile(window, 90) - np.percentile(window, 10)) / length:6.3f}% of the beat")

    if shares:
        shares_arr = np.abs(np.array(shares))
        print("")
        print(f"over {shares_arr.size} sections:")
        print(f"  middle error          {np.median(shares_arr) * 100:.3f}%   p90 {np.percentile(shares_arr, 90) * 100:.3f}%")
        print(f"  drift over a minute   {np.median(shares_arr) * 60_000:.0f}ms   p90 {np.percentile(shares_arr, 90) * 60_000:.0f}ms")
        print(f"  sections within 0.05% (30ms a minute)  {int((shares_arr <= 0.0005).sum())}/{shares_arr.size}")
        print(f"  sections within 0.10% (60ms a minute)  {int((shares_arr <= 0.001).sum())}/{shares_arr.size}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
