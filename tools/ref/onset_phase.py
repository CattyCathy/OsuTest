"""Which phase is the music actually on where the model and the beatmap disagree?

The model reports a clean 300ms pulse at 33.8-36.2s that sits 131ms away from the map's grid, and the map declares a
single 300ms grid from 409ms to 82008ms. Rather than assume either side is right, measure the onset strength of the
audio at each candidate phase and let the audio decide.

Usage:
    python onset_phase.py <dump-dir> <timingpoints.csv>
"""

import sys
import os
import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "beat_this-1.1.0"))
from beat_this.preprocessing import LogMelSpect

dump_dir, timing_path = sys.argv[1], sys.argv[2]

pcm = np.fromfile(os.path.join(dump_dir, "pcm.f32"), dtype=np.float32)
signal = torch.tensor(pcm, dtype=torch.float32)

spect = LogMelSpect()(signal).numpy()  # (frames, 128)
# spectral flux on the log-mel, half-wave rectified: peaks where energy is added
flux = np.maximum(0.0, np.diff(spect, axis=0)).sum(1)
flux = np.concatenate([[0.0], flux])
flux = flux / max(flux.mean(), 1e-12)

print(f"frames {len(flux)}, mean flux {flux.mean():.4f}")

points = []
for line in open(timing_path, encoding="utf-8-sig"):
    line = line.strip()
    if not line or line.startswith("#") or "," not in line:
        continue
    a, b = line.split(",")[:2]
    points.append((float(a), float(b)))
points.sort()


def grid_between(lo_ms, hi_ms):
    out = []
    for i, (time, length) in enumerate(points):
        if length <= 0:
            continue
        end = points[i + 1][0] if i + 1 < len(points) else hi_ms
        t = time
        while t < end:
            if lo_ms <= t < hi_ms:
                out.append(t)
            t += length
    return np.array(out)


def mean_flux_at(times_ms, radius_ms=20):
    idx = np.round(np.array(times_ms) * 50 / 1000).astype(int)
    r = int(round(radius_ms * 50 / 1000))
    picked = []
    for i in idx:
        if 0 <= i < len(flux):
            picked.append(flux[max(0, i - r): i + r + 1].max())
    return float(np.mean(picked)) if picked else float("nan")


for lo, hi, label in [(2000, 28000, "0-28s (model on the map's grid)"),
                      (30000, 46000, "30-46s (model 131ms off)"),
                      (50000, 60000, "50-60s"),
                      (70000, 80000, "70-80s")]:
    base = grid_between(lo, hi)
    print()
    print(f"=== {label}: {len(base)} grid lines in [{lo/1000:0.0f}s, {hi/1000:0.0f}s), "
          f"mean flux at an arbitrary time {mean_flux_at(np.linspace(lo, hi, 200)):.3f}")
    best = None
    for offset in range(0, 300, 10):
        value = mean_flux_at(base + offset)
        if best is None or value > best[1]:
            best = (offset, value)
        print(f"    grid + {offset:3}ms  ->  {value:.3f}" + ("   <-- best" if False else ""))
    print(f"    best offset {best[0]}ms with {best[1]:.3f}")
