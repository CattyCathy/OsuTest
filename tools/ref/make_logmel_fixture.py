"""Generate the frontend regression fixture.

The C# log-mel frontend has to reproduce torchaudio's exactly, and it did not: it applied the Slaney area
normalisation that torchaudio's MelSpectrogram leaves off, and omitted the 1/sqrt(n_fft) scale that
normalized="frame_length" applies. Both were silent - the model still tracked steady music, and only fell apart on
dense music, which is why they survived so long. This writes the reference's own output for a deterministic signal so
the test suite can catch either one coming back without needing torchaudio, or a model, or a track.

Usage:
    python make_logmel_fixture.py <output.bin>
"""

import struct
import sys
import numpy as np
import torch
import torchaudio

HOP = 441
FRAMES = 120
SAMPLE_RATE = 22050

# A deterministic signal with energy at both ends of the spectrum, so a filterbank that is tilted by frequency shows
# up rather than cancelling out.
FREQUENCIES = [(220.0, 0.50), (1310.0, 0.30), (4400.0, 0.20), (9000.0, 0.10)]


def samples(count):
    n = np.arange(count, dtype=np.float64)
    signal = np.zeros(count, dtype=np.float64)

    for frequency, amplitude in FREQUENCIES:
        signal += amplitude * np.sin(2.0 * np.pi * frequency * n / SAMPLE_RATE)

    return signal.astype(np.float32)


def main():
    output = sys.argv[1]

    # Long enough that the last wanted frame's window lies inside the signal, so neither side reflects.
    count = FRAMES * HOP + 512
    signal = samples(count)

    spect = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=HOP, f_min=30, f_max=11000,
        n_mels=128, mel_scale="slaney", normalized="frame_length", power=1,
    )(torch.tensor(signal))

    logmel = torch.log1p(1000 * spect.T).numpy()[:FRAMES].astype(np.float32)

    with open(output, "wb") as handle:
        handle.write(struct.pack("<III", FRAMES, 128, count))
        handle.write(logmel.tobytes())
        handle.write(signal.tobytes())

    print(f"wrote {output}: {FRAMES} frames from {count} samples")
    print(f"  band 0 min/max {logmel[:, 0].min():.4f}/{logmel[:, 0].max():.4f}")
    print(f"  band 127 min/max {logmel[:, 127].min():.4f}/{logmel[:, 127].max():.4f}")


if __name__ == "__main__":
    main()
