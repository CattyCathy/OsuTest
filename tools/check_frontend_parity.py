"""Checks a Python mel frontend against the one ParaTactus uses.

Run this before training anything. A detector is trained on features computed here and run against features computed
in C#, so if the two frontends disagree by a constant tilt, a framing offset or an edge convention the model is being
asked at play time to read something it never saw. Both implementations look correct on their own, which is exactly
why the comparison has to be numerical.

Three candidates are compared, because there is no reason to hand-roll one if a library already agrees:
  1. torchaudio's MelSpectrogram with the arguments the reference frontend documents
  2. numpy, written out longhand
  3. librosa, if it happens to be installed
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
HOP = 441
N_FFT = 1024
N_MELS = 128
F_MIN = 30.0
F_MAX = 11000.0


def read_reference(path: Path):
    raw = path.read_bytes()

    frames, bins = struct.unpack_from("<ii", raw, 0)
    offset = 8
    spectrogram = np.frombuffer(raw, dtype="<f4", count=frames * bins, offset=offset).reshape(frames, bins)
    offset += frames * bins * 4

    sample_count = struct.unpack_from("<i", raw, len(raw) - 4)[0]
    samples = np.frombuffer(raw, dtype="<f4", count=sample_count, offset=offset).astype(np.float64)

    return spectrogram.astype(np.float64), samples


def report(name: str, mine: np.ndarray, reference: np.ndarray) -> bool:
    if mine.shape != reference.shape:
        print(f"{name:20} SHAPE MISMATCH {mine.shape} vs {reference.shape}")
        return False

    difference = np.abs(mine - reference)
    worst = difference.max()
    print(f"{name:20} max {worst:12.6f}   mean {difference.mean():12.8f}   "
          f"range {mine.min():8.3f}..{mine.max():8.3f}")

    return worst < 1e-3


# ------------------------------------------------------------------------------------------------ candidate 1: torchaudio


def with_torchaudio(samples: np.ndarray):
    try:
        import torch
        import torchaudio
    except Exception as error:  # noqa: BLE001
        print(f"torchaudio unavailable: {error}")
        return None

    waveform = torch.from_numpy(samples).float().unsqueeze(0)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP,
        f_min=F_MIN,
        f_max=F_MAX,
        n_mels=N_MELS,
        mel_scale="slaney",
        norm=None,          # raw triangles; "slaney" area-normalises and tilts the high bands
        power=1,            # magnitude, not power
        center=True,
        pad_mode="reflect",
        window_fn=torch.hann_window,
    )

    with torch.no_grad():
        spectrogram = mel(waveform)

    spectrogram = torch.log1p(1000.0 * spectrogram)

    return spectrogram.squeeze(0).T.numpy().astype(np.float64)


# ------------------------------------------------------------------------------------------------ candidate 2: numpy


def slaney_filterbank() -> np.ndarray:
    def hz_to_mel(f):
        f_sp = 200.0 / 3
        min_log_hz, logstep = 1000.0, math.log(6.4) / 27.0
        min_log_mel = min_log_hz / f_sp

        if f >= min_log_hz:
            return min_log_mel + math.log(f / min_log_hz) / logstep

        return f / f_sp

    def mel_to_hz(m):
        f_sp = 200.0 / 3
        min_log_hz, logstep = 1000.0, math.log(6.4) / 27.0
        min_log_mel = min_log_hz / f_sp

        if m >= min_log_mel:
            return min_log_hz * math.exp(logstep * (m - min_log_mel))

        return f_sp * m

    bands = np.array([mel_to_hz(m) for m in np.linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), N_MELS + 2)])
    frequencies = np.arange(N_FFT // 2 + 1) * SAMPLE_RATE / N_FFT

    bank = np.zeros((N_MELS, len(frequencies)))

    for m in range(N_MELS):
        lower, centre, upper = bands[m], bands[m + 1], bands[m + 2]

        for i, f in enumerate(frequencies):
            if lower < f < centre:
                bank[m, i] = (f - lower) / (centre - lower)
            elif centre <= f < upper:
                bank[m, i] = (upper - f) / (upper - centre)

    return bank


def with_numpy(samples: np.ndarray):
    pad = N_FFT // 2
    padded = np.pad(samples, pad, mode="reflect")

    frame_count = 1 + (len(padded) - N_FFT) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP][:frame_count]

    # torchaudio's default hann_window is periodic, which drops the last sample of a symmetric window.
    window = np.hanning(N_FFT + 1)[:-1]

    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) / math.sqrt(N_FFT)
    mels = spectrum @ slaney_filterbank().T

    return np.log1p(1000.0 * mels)


# ------------------------------------------------------------------------------------------------ candidate 3: librosa


def with_librosa(samples: np.ndarray):
    try:
        import librosa
    except Exception:  # noqa: BLE001
        return None

    mels = librosa.feature.melspectrogram(
        y=samples.astype(np.float32),
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=1.0,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        htk=False,
        norm=None,
    )

    return np.log1p(1000.0 * mels).T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()

    reference, samples = read_reference(args.reference)

    print(f"reference: {reference.shape[0]} frames x {reference.shape[1]} bins, {len(samples)} samples "
          f"({len(samples) / SAMPLE_RATE:.1f}s)")
    print(f"reference range: {reference.min():.3f} .. {reference.max():.3f}")
    print("")

    results = {}

    for name, function in (("torchaudio", with_torchaudio), ("numpy", with_numpy), ("librosa", with_librosa)):
        try:
            mine = function(samples)
        except Exception as error:  # noqa: BLE001
            print(f"{name:20} raised {type(error).__name__}: {error}")
            continue

        if mine is None:
            continue

        results[name] = report(name, mine, reference)

    print("")

    if not results:
        print("no candidate could be evaluated")
        return 1

    # The numpy one is the fallback the training script uses, so it is the one that has to agree.
    for name, ok in results.items():
        print(f"{name:20} {'PARITY OK' if ok else 'PARITY FAILED'}")

    return 0 if results.get("numpy") else 1


if __name__ == "__main__":
    raise SystemExit(main())
