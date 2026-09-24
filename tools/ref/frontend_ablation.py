"""Ablation: which frontend normalisation does this ONNX model actually want?

Everything except the frontend is held fixed (same PCM, same reference chunk schedule, same reference postprocessor)
so the only variable is whether the STFT is divided by sqrt(n_fft), which is what torchaudio's
normalized="frame_length" does. Each variant's beats are scored against the beatmap's own timing-point grid.

Usage:
    python frontend_ablation.py <dump-dir> <model.onnx> <timingpoints.csv>
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "beat_this-1.1.0"))

from beat_this.preprocessing import LogMelSpect
from beat_this.model.postprocessor import Postprocessor

import onnxruntime as ort

CHUNK = 1500
BORDER = 6
N_MELS = 128
FPS = 50


def split_piece(spect, chunk_size, border_size=6):
    starts = np.arange(-border_size, len(spect) - border_size, chunk_size - 2 * border_size)
    starts[-1] = len(spect) - (chunk_size - border_size)
    chunks = []
    for start in starts:
        piece = spect[max(start, 0): min(start + chunk_size, len(spect))]
        left = max(0, -start)
        right = max(0, min(border_size, start + chunk_size - len(spect)))
        if left or right:
            piece = torch.nn.functional.pad(piece, (0, 0, left, right), "constant", 0)
        chunks.append(piece)
    return chunks, starts


def aggregate(pred_chunks, starts, full_size, chunk_size, border_size):
    pred_chunks = [
        {"beat": p["beat"][border_size:-border_size], "downbeat": p["downbeat"][border_size:-border_size]}
        for p in pred_chunks
    ]
    beat = torch.full((full_size,), -1000.0)
    for start, pchunk in zip(reversed(list(starts)), reversed(list(pred_chunks))):
        beat[start + border_size: start + chunk_size - border_size] = pchunk["beat"]
    return beat


def frontend(signal, frame_length_normalized):
    """The published LogMelSpect, optionally without the STFT normalisation it asks torchaudio for."""
    spec = LogMelSpect()
    if frame_length_normalized:
        return spec(signal).numpy()

    spect_class = spec.spect_class
    with torch.no_grad():
        mel = spect_class(signal) * np.sqrt(1024.0)
    return torch.log1p(1000 * mel.T).numpy()


def beats_for(session, spect):
    chunks, starts = split_piece(torch.from_numpy(spect), CHUNK, BORDER)
    pred_chunks = []
    for chunk in chunks:
        out = session.run(None, {"spectrogram": chunk[None, :, :].numpy().astype(np.float32)})
        names = [o.name for o in session.get_outputs()]
        pred_chunks.append({
            "beat": torch.from_numpy(out[names.index("beat")][0]),
            "downbeat": torch.from_numpy(out[names.index("downbeat")][0]),
        })
    logits = aggregate(pred_chunks, starts, spect.shape[0], CHUNK, BORDER)
    beats, _ = Postprocessor(type="minimal")(logits, logits)
    return np.asarray(beats) * 1000.0, logits.numpy()


def load_timing_points(path):
    points = []
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if not line or line.startswith("#") or "," not in line:
            continue
        a, b = line.split(",")[:2]
        points.append((float(a), float(b)))
    return sorted(points)


def build_grid(points, until_ms=1_000_000):
    grid = []
    for i, (time, length) in enumerate(points):
        if length <= 0:
            continue
        end = points[i + 1][0] if i + 1 < len(points) else until_ms
        t = time
        while t < end and len(grid) < 500_000:
            grid.append(t)
            t += length
    return np.sort(np.array(grid))


def residuals(beats, grid):
    idx = np.searchsorted(grid, beats)
    out = []
    for i, t in enumerate(beats):
        lo = max(0, idx[i] - 2)
        hi = min(len(grid), idx[i] + 3)
        out.append(np.min(np.abs(grid[lo:hi] - t)))
    return np.array(out)


def report(name, beats, grid):
    res = residuals(beats, grid)
    print()
    print(f"=== {name}: {len(beats)} beats, median residual {np.median(res):.0f}ms, "
          f"over 60ms {int((res > 60).sum())} of {len(res)} ({100 * (res > 60).mean():.0f}%)")
    print("   from      n  median     p90     max  over60   implied BPM")
    for lo in range(0, int(beats[-1]), 10_000):
        mask = (beats >= lo) & (beats < lo + 10_000)
        if not mask.any():
            continue
        b = np.sort(res[mask])
        inside = beats[mask]
        gaps = np.diff(inside)
        bpm = 60000 / np.median(gaps) if len(gaps) else 0
        print(f"{lo / 1000:6.0f}s {len(b):6} {b[len(b) // 2]:7.0f} {b[int(len(b) * 0.9)]:7.0f} "
              f"{b[-1]:7.0f} {int((b > 60).sum()):7} {bpm:12.1f}")
    return res


def main():
    dump_dir, model_path, timing_path = sys.argv[1], sys.argv[2], sys.argv[3]

    pcm = np.fromfile(os.path.join(dump_dir, "pcm.f32"), dtype=np.float32)
    grid = build_grid(load_timing_points(timing_path))
    signal = torch.tensor(pcm, dtype=torch.float32)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    for label, normalized in [("official: normalized=\"frame_length\" (STFT / sqrt(n_fft))", True),
                              ("no frame_length normalisation", False)]:
        spect = frontend(signal, normalized)
        beats, logits = beats_for(session, spect)
        report(label, beats, grid)
        np.save(os.path.join(dump_dir, f"beats_{'norm' if normalized else 'plain'}.npy"), beats)


if __name__ == "__main__":
    main()
