"""Confirm the exact frontend the reference uses: filterbank normalization and STFT scaling."""

import sys
import os
import inspect
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

dump_dir = sys.argv[1]

pcm = np.fromfile(os.path.join(dump_dir, "pcm.f32"), dtype=np.float32)
csharp_mag = np.fromfile(os.path.join(dump_dir, "magnitudes.f32"), dtype=np.float32).reshape(-1, 512)
first = int(open(os.path.join(dump_dir, "magnitudes.txt")).read().strip())
frame = first + 4

signal = torch.tensor(pcm, dtype=torch.float32)

mod = T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=441, f_min=30, f_max=11000,
                       n_mels=128, mel_scale="slaney", normalized="frame_length", power=1)
print("MelSpectrogram default norm =", inspect.signature(T.MelSpectrogram.__init__).parameters["norm"].default)
print("MelSpectrogram default mel_scale =",
      inspect.signature(T.MelSpectrogram.__init__).parameters["mel_scale"].default)
print("built-in fb : peak", mod.mel_scale.fb.max().item(), " rowsum band0", mod.mel_scale.fb[:512, 0].sum().item())

for norm in ["slaney", None]:
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=513, f_min=30.0, f_max=11000.0, n_mels=128, sample_rate=22050,
        norm=norm, mel_scale="slaney",
    ).numpy()
    d = np.abs(fb - mod.mel_scale.fb.numpy()).max()
    print(f"  melscale_fbanks(norm={norm!r}) vs built-in: max|d| {d:.6g}  peak {fb.max():.6g}")

# the whole frontend, rebuilt from the verified C# magnitudes
plain = mod(signal).numpy()
fb_none = torchaudio.functional.melscale_fbanks(
    n_freqs=513, f_min=30.0, f_max=11000.0, n_mels=128, sample_rate=22050,
    norm=None, mel_scale="slaney",
).numpy()[:512].T

print()
print(f"frame {frame}: candidate frontends, band0 / band127")
candidates = {
    "c# current (slaney norm, no 1/32)": (fb_none, None),
    "norm=None, no 1/32": (fb_none, None),
    "norm=None, with 1/32": (fb_none, 32.0),
}
mine_slaney = torchaudio.functional.melscale_fbanks(
    n_freqs=513, f_min=30.0, f_max=11000.0, n_mels=128, sample_rate=22050,
    norm="slaney", mel_scale="slaney",
).numpy()[:512].T

mag = csharp_mag[4].astype(np.float64)
for label, (fb, scale) in [
    ("c# current: slaney-norm fb, no 1/32", (mine_slaney, 1.0)),
    ("norm=None fb, no 1/32", (fb_none, 1.0)),
    ("norm=None fb, / sqrt(1024)", (fb_none, 32.0)),
    ("slaney-norm fb, / sqrt(1024)", (mine_slaney, 32.0)),
]:
    mel = (fb @ mag) / scale
    print(f"  {label:38} band0 {np.log1p(1000 * mel[0]):8.4f}  band127 {np.log1p(1000 * mel[127]):8.4f}")

print(f"  {'torchaudio reference (mod(signal))':38} "
      f"band0 {np.log1p(1000 * plain[0, frame]):8.4f}  band127 {np.log1p(1000 * plain[127, frame]):8.4f}")

# and over every frame of the probe, for the winning candidate
ref = np.log1p(1000 * mod(signal).numpy().T)[first:first + len(csharp_mag)]
best = np.log1p(1000 * ((fb_none @ csharp_mag.T) / 32.0).T)
print()
print(f"  norm=None + 1/32 over all {len(csharp_mag)} probe frames: max|d| {np.abs(best - ref).max():.6g}")

only_norm = np.log1p(1000 * (fb_none @ csharp_mag.T).T)
print(f"  norm=None alone:                                        max|d| {np.abs(only_norm - ref).max():.6g}")
only_scale = np.log1p(1000 * ((mine_slaney @ csharp_mag.T) / 32.0).T)
print(f"  slaney fb + 1/32:                                       max|d| {np.abs(only_scale - ref).max():.6g}")
