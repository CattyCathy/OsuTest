"""Compare the C# frontend and model against the published beat_this Python implementation.

Reads the dumps written by ReferenceParityDump so both sides see exactly the same decoded PCM, which removes the
decoder as a variable and leaves the frontend, the chunking and the postprocessing to be checked one at a time.

Usage:
    python parity.py <dump-dir> <model.onnx>
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "beat_this-1.1.0"))

from beat_this.preprocessing import LogMelSpect
from beat_this.model.postprocessor import Postprocessor

import torch.nn.functional as F
import onnxruntime as ort


def zeropad(spect, left=0, right=0):
    if left == 0 and right == 0:
        return spect
    return F.pad(spect, (0, 0, left, right), "constant", 0)


def split_piece(spect, chunk_size, border_size=6, avoid_short_end=True):
    """Transcribed verbatim from beat_this/inference.py; inference.py itself pulls in the model class."""
    starts = np.arange(-border_size, len(spect) - border_size, chunk_size - 2 * border_size)
    if avoid_short_end and len(spect) > chunk_size - 2 * border_size:
        starts[-1] = len(spect) - (chunk_size - border_size)
    chunks = [
        zeropad(
            spect[max(start, 0): min(start + chunk_size, len(spect))],
            left=max(0, -start),
            right=max(0, min(border_size, start + chunk_size - len(spect))),
        )
        for start in starts
    ]
    return chunks, starts


def aggregate_prediction(pred_chunks, starts, full_size, chunk_size, border_size, overlap_mode, device):
    """Transcribed verbatim from beat_this/inference.py."""
    if border_size > 0:
        pred_chunks = [
            {
                "beat": pchunk["beat"][border_size:-border_size],
                "downbeat": pchunk["downbeat"][border_size:-border_size],
            }
            for pchunk in pred_chunks
        ]
    piece_beat = torch.full((full_size,), -1000.0, device=device)
    piece_downbeat = torch.full((full_size,), -1000.0, device=device)
    if overlap_mode == "keep_first":
        pred_chunks = reversed(list(pred_chunks))
        starts = reversed(list(starts))
    for start, pchunk in zip(starts, pred_chunks):
        piece_beat[start + border_size: start + chunk_size - border_size] = pchunk["beat"]
        piece_downbeat[start + border_size: start + chunk_size - border_size] = pchunk["downbeat"]
    return piece_beat, piece_downbeat

FPS = 50
CHUNK = 1500
BORDER = 6
N_MELS = 128


def load(path, dtype=np.float32):
    return np.fromfile(path, dtype=dtype)


def run_onnx(session, spect_chunk):
    """spect_chunk: (T, 128) float32 -> (beat, downbeat) each (T,)."""
    inp = spect_chunk[None, :, :].astype(np.float32)
    out = session.run(None, {"spectrogram": inp})
    names = [o.name for o in session.get_outputs()]
    beat = out[names.index("beat")][0]
    downbeat = out[names.index("downbeat")][0]
    return beat, downbeat


def reference_logits(session, spect):
    """The published chunk schedule: starts at -border, last chunk forced to end at the piece end."""
    chunks, starts = split_piece(spect, CHUNK, border_size=BORDER, avoid_short_end=True)
    pred_chunks = []

    for chunk in chunks:
        beat, downbeat = run_onnx(session, chunk.numpy())
        pred_chunks.append({"beat": torch.from_numpy(beat), "downbeat": torch.from_numpy(downbeat)})

    beat, downbeat = aggregate_prediction(
        pred_chunks, starts, spect.shape[0], CHUNK, BORDER, "keep_first", "cpu"
    )
    return beat.numpy(), downbeat.numpy(), starts


def describe(name, csharp, reference):
    delta = np.abs(csharp - reference)
    denom = np.maximum(np.abs(reference), 1e-6)
    rel = delta / denom
    print(f"  {name:12} max|d| {delta.max():12.6g}   mean|d| {delta.mean():12.6g}   "
          f"max rel {rel.max():10.4g}   mean rel {rel.mean():10.4g}")
    return rel.max()


def main():
    dump_dir, model_path = sys.argv[1], sys.argv[2]

    pcm = load(os.path.join(dump_dir, "pcm.f32"))
    csharp_spect = load(os.path.join(dump_dir, "spect.f32"))
    csharp_logits = load(os.path.join(dump_dir, "logits.f32"))

    print(f"pcm {pcm.shape}  csharp spect {csharp_spect.shape}  csharp logits {csharp_logits.shape}")

    signal = torch.tensor(pcm, dtype=torch.float32)
    frontend = LogMelSpect()
    reference_spect = frontend(signal).numpy()

    print(f"reference spect {reference_spect.shape}")

    frames = min(csharp_spect.shape[0] // N_MELS, reference_spect.shape[0])

    print()
    print("frontend (log-mel spectrogram), same PCM on both sides:")
    worst = describe("complete", csharp_spect[: frames * N_MELS], reference_spect[:frames].reshape(-1))

    if worst > 1e-3:
        a = csharp_spect[: frames * N_MELS].reshape(frames, N_MELS)
        b = reference_spect[:frames]
        print()
        print("  difference structure (first 3 frames, first 8 bins):")
        print("    c#   ", np.array2string(a[0, :8], precision=4))
        print("    ref  ", np.array2string(b[0, :8], precision=4))
        print("    c#   ", np.array2string(a[1, :8], precision=4))
        print("    ref  ", np.array2string(b[1, :8], precision=4))

        ratio = np.expm1(b) / np.maximum(np.expm1(a), 1e-12)
        print(f"  implied mel ratio (ref/c#, ignoring the log1p): median {np.median(ratio):.6g}  "
              f"mean {ratio.mean():.6g}")
        delta = b - a
        print(f"  additive delta in log space: median {np.median(delta):.6g}  mean {delta.mean():.6g}")

    print()
    print("frontend per mel band, mean relative difference:")
    a = csharp_spect[: frames * N_MELS].reshape(frames, N_MELS)
    b = reference_spect[:frames]
    per_band = (np.abs(a - b) / np.maximum(np.abs(b), 1e-6)).mean(axis=0)
    print("  " + np.array2string(per_band, precision=3, max_line_width=200))

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print()
    print(f"onnx outputs: {[o.name for o in session.get_outputs()]}")

    reference_beat, _, starts = reference_logits(session, torch.from_numpy(reference_spect))
    print(f"reference chunk starts: {starts[:6].tolist()} ... (n={len(starts)})")

    print()
    print("model logits (same chunk schedule is NOT shared - this compares reference chunks vs the c# ones):")
    describe("beat logits", csharp_logits, reference_beat)

    reference_beats, _ = Postprocessor(type="minimal")(torch.from_numpy(reference_beat), torch.from_numpy(reference_beat))
    reference_beats = np.asarray(reference_beats) * 1000.0

    print()
    print(f"reference postprocessor found {len(reference_beats)} beats over {reference_beats[-1] / 1000:.1f}s")
    print("beats in the problem window:")
    for lo, hi in [(0, 10000), (28000, 30000), (36000, 48000)]:
        inside = reference_beats[(reference_beats >= lo) & (reference_beats < hi)]
        print(f"  [{lo / 1000:5.0f}s,{hi / 1000:5.0f}s) n={len(inside):3}  "
              + ", ".join(f"{t / 1000:.2f}" for t in inside[:24]))

    np.save(os.path.join(dump_dir, "reference_beats.npy"), reference_beats)
    np.save(os.path.join(dump_dir, "reference_logits.npy"), reference_beat)


if __name__ == "__main__":
    main()
