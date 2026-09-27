"""Trains a beat detector that reads a log-mel spectrogram and reports a beat on each frame.

The detector exists because peak picking has no notion of tempo. It takes each local maximum on its own merits with a
local rule about gaps, so nothing in it can say "a beat was 300ms ago and this one cannot be 600ms away", and the two
failures that follow are the whole of the problem this project has: on a passage where every other beat is lost the
peaks come out at half the density and there is nothing in the positions to say so, and on a passage at half density
they look exactly the same. A network with ten seconds of context can tell those apart, because it can see which one
the surrounding music is doing.

The frontend matches the one ParaTactus already uses, so the features the model is trained on are the features it will
be given at play time. That frontend is not torchaudio's default and the differences are all silent:

    magnitude, not power          torchaudio defaults to power=2
    Slaney mel filters, not area  torchaudio's norm=None passes raw triangles; norm="slaney" tilts them by up to 25x
    frame-length STFT scaling     normalized="frame_length" divides by sqrt(n_fft)
    log1p(1000 * mel)             not a plain log

Use --check-parity against a dump from the C# side before trusting any of it.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
HOP = 441
N_FFT = 1024
N_MELS = 128
FPS = 50


# -------------------------------------------------------------------------------------------------------------- frontend


def mel_filterbank() -> np.ndarray:
    """Slaney-scale triangular mel filters, not area normalised, matching the C# implementation."""
    def hz_to_mel(f: float) -> float:
        f_sp = 200.0 / 3
        mel = f / f_sp
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0

        if f >= min_log_hz:
            mel = min_log_mel + math.log(f / min_log_hz) / logstep

        return mel

    def mel_to_hz(m: float) -> float:
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0

        if m >= min_log_mel:
            return min_log_hz * math.exp(logstep * (m - min_log_mel))

        return f_sp * m

    f_min, f_max = 30.0, 11000.0
    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), N_MELS + 2)
    freqs = np.array([mel_to_hz(m) for m in mels])
    bins = np.arange(N_FFT // 2 + 1)
    bin_freqs = bins * SAMPLE_RATE / N_FFT

    bank = np.zeros((N_MELS, len(bins)), dtype=np.float64)

    for m in range(N_MELS):
        lower, centre, upper = freqs[m], freqs[m + 1], freqs[m + 2]

        for i, f in enumerate(bin_freqs):
            if lower <= f <= centre and centre > lower:
                bank[m, i] = (f - lower) / (centre - lower)
            elif centre <= f <= upper and upper > centre:
                bank[m, i] = (upper - f) / (upper - centre)

    return bank


def log_mel(samples: np.ndarray) -> np.ndarray:
    """log1p(1000 * mel) of the magnitude spectrogram, shaped (frames, mels), centred like torchaudio."""
    bank = mel_filterbank()

    # torchaudio's center=True reflects about both edges by n_fft//2 before framing.
    pad = N_FFT // 2
    padded = np.pad(samples, pad, mode="reflect")

    frame_count = 1 + (len(padded) - N_FFT) // HOP
    window = np.hanning(N_FFT + 1)[:-1] if False else np.hanning(N_FFT)

    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP][:frame_count]

    # Magnitude, one-sided, scaled by 1/sqrt(n_fft) as frame_length normalisation does.
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) / math.sqrt(N_FFT)

    mels = spectrum @ bank.T

    return np.log1p(1000.0 * mels).astype(np.float32)


# -------------------------------------------------------------------------------------------------------------- dataset


@dataclass
class Track:
    id: str
    audio: Path
    labels: Path
    frames: int
    usable: bool
    label_period: float = 0.0


def load_manifest(dataset: Path) -> list[Track]:
    manifest = dataset / "manifest.tsv"
    tracks: list[Track] = []

    with manifest.open() as handle:
        header = handle.readline()

        for line in handle:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 9:
                continue

            track_id, _set, audio, labels, frames, _beats, period, _lag, _octave = parts[:9]
            usable = parts[9] == "1" if len(parts) > 9 else True

            tracks.append(Track(
                id=track_id,
                audio=dataset / "audio" / audio,
                labels=dataset / "labels" / labels,
                frames=int(frames),
                usable=usable,
                label_period=float(period) if period else 0.0,
            ))

    return [t for t in tracks if t.audio.exists() and t.labels.exists()]


def read_track(track: Track) -> tuple[np.ndarray, np.ndarray]:
    samples = np.fromfile(track.audio, dtype=np.float32)

    # The label path is carried rather than derived. Path.with_suffix would replace ".i8" with nothing useful, and the
    # audio and label names differ only in their extension, which is exactly the kind of thing that a derivation gets
    # subtly wrong on one file and then reads a truncated label set without complaining.
    labels = np.fromfile(track.labels, dtype=np.int8).astype(np.float32)

    # The exporter writes a downbeat as class two and every other beat as class one. A beat detector is being asked
    # whether a frame is a beat, not which beat of the bar it is, so the two are folded together here - and folded
    # rather than clamped, because the one place the distinction could quietly change an answer is the period, which is
    # measured between beats and would be measured between downbeats only if this were left unmapped.
    labels = (labels > 0).astype(np.float32)

    return samples, labels


# -------------------------------------------------------------------------------------------------------------- model


def build_model(torch, context_frames: int = 512, wide_period: bool = True):
    """A stack of dilated convolutions over mel frames, with a small frequency-reducing stem.

    The receptive field is what matters and it is set by the dilations: a decision about whether a run of gaps is a
    lost beat or a real tempo change needs several beats either side, which at 50 frames a second and a two-second beat
    is a few hundred frames. The stem collapses frequency so the temporal stack is cheap.

    `wide_period` chooses the shape of the period head, and it exists only so that models trained before the head was
    widened can still be loaded for comparison. The two are not interchangeable - their weights have different shapes -
    so a caller that reads an old checkpoint has to ask for the old shape.
    """
    nn = torch.nn

    class Detector(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.stem = nn.Sequential(
                nn.Conv2d(1, 24, kernel_size=(5, 5), padding=(2, 2)),
                nn.BatchNorm2d(24),
                nn.GELU(),
                nn.MaxPool2d((4, 1)),
                nn.Conv2d(24, 48, kernel_size=(5, 5), padding=(2, 2)),
                nn.BatchNorm2d(48),
                nn.GELU(),
                nn.MaxPool2d((4, 1)),
            )

            # The stem's two pools take 128 mel bins to 29 and then to 8, so the frequency axis is not collapsed to one
            # and the temporal stack has to say what it does with the rest. Collapsing it with a convolution of kernel
            # 8 - the whole remaining frequency extent - is what turns the spectrogram into a per-frame vector, and
            # doing it here rather than by pooling keeps the learned part of the reduction.
            self.collapse = nn.Sequential(
                nn.Conv2d(48, 64, kernel_size=(5, 5), padding=(2, 2)),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=(8, 1)),
                nn.BatchNorm2d(64),
                nn.GELU(),
            )

            self.temporal = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, padding=2, dilation=1),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=5, padding=4, dilation=2),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=5, padding=8, dilation=4),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=5, padding=16, dilation=8),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=5, padding=32, dilation=16),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=5, padding=64, dilation=32),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=5, padding=128, dilation=64),
                nn.BatchNorm1d(128), nn.GELU(),
            )

            self.head = nn.Conv1d(128, 1, kernel_size=1)

            # A second output: the beat period, in frames, as a logarithm so that being a frame out at 300ms costs the
            # same as being a frame out at 60. Predicting it is what gives the model somewhere to put the tempo, which a
            # binary per-frame target leaves nowhere for - and without it the only reading of the curve is to take the
            # peaks, which fires on every onset the music has.
            #
            # Wider than two pointwise layers, which is what it was, because it turned out to be the thing the reading
            # depends on and it was not learning it. The period is what sets how far apart two beats have to be before
            # both are kept, so its error is the reading's: measured over twenty-four tracks the head's middle ratio to
            # the true period is 0.98, which is right, while six tracks are more than a tenth short and three are short
            # by a third or a half - and on those three the reading reports 2.16, 1.98 and 1.74 times the beats there
            # are, because a radius of half the truth keeps a beat every half a beat. Substituting the labels' own gaps
            # for the head's output takes the same three to 1.07, 1.08 and 1.03.
            #
            # What is ruled out as the cause: the target, which is the middle of the label gaps and measures 21 frames
            # where the labels are 21; and the loss weight, which was swept to 3, 10 and 30 and left the head predicting
            # 14 frames against a 21-frame target at every one of them. A regression an order of magnitude out that does
            # not move when its loss is multiplied by thirty is not being under-weighted, it is being asked of two
            # pointwise layers. The temporal convolutions do the work here.
            self.period = nn.Sequential(*(
                [
                    nn.Conv1d(128, 128, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(128, 64, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(64, 1, kernel_size=1),
                ]
                if wide_period
                else [
                    nn.Conv1d(128, 64, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(64, 1, kernel_size=1),
                ]
            ))

        def forward(self, x):
            # x: (batch, mels, frames)
            x = x.unsqueeze(1)
            x = self.stem(x)
            x = self.collapse(x)

            # The frequency axis is one by now and the channel axis is what the temporal stack wants. Written as a
            # reshape rather than a squeeze on a position, because squeeze takes every axis of size one and would remove
            # the frame axis too whenever a caller passed a single frame.
            batch, channels, _, frames = x.shape
            x = x.reshape(batch, channels, frames)

            features = self.temporal(x)

            return self.head(features).reshape(batch, frames), self.period(features).reshape(batch, frames)

    return Detector()


def period_targets(labels: np.ndarray, minimum: int = 8, maximum: int = 200) -> np.ndarray:
    """The beat period in frames, per frame, from the labelled beats.

    A second thing for the model to predict, and the one that was missing. A per-frame classifier with a binary target
    has no way to know how far apart beats are: the label says "beat here" and "not a beat there", and the cheapest
    function that satisfies that on music with an onset every sixteenth is to fire on the sixteenths, which is what the
    first two runs did - four times the map's beat count on a track whose labels are unambiguously 660ms apart.

    What the period is measured from matters as much as the head existing. Taking the median of every gap in a track
    whose labels mix levels gives half the beat: one track's labels are 36 gaps of 320ms among 119 of 640 and 680, and
    the median over all of them came out at 362ms, which is why that track's model predicted 362ms against a map at
    667. The gaps a period is taken from are therefore the ones between beats that are not themselves subdivisions,
    which is decided by whether a beat is closer to its neighbour than the level around it.
    """
    frames = len(labels)
    target = np.zeros(frames, dtype=np.float32)

    beats = np.flatnonzero(labels > 0.5)

    if beats.size < 3:
        return target

    keep = level_beats(beats)

    if keep.size < 2:
        keep = beats

    # The period at a frame is the median of the gaps between the kept beats near it, and a frame in a gap inherits the
    # tempo leading into it rather than being left undefined - what is being asked is not where the beats are but what
    # the tempo is, and that question has an answer in a passage with no beats at all.
    for position in range(frames):
        index = int(np.searchsorted(keep, position, side="right")) - 1
        gaps = []

        for offset in range(max(0, index - 2), min(keep.size - 1, index + 3)):
            gaps.append(keep[offset + 1] - keep[offset])

        if not gaps:
            gaps = [int(keep[-1] - keep[0]) // max(1, keep.size - 1)]

        target[position] = float(np.median(gaps))

    return np.clip(target, minimum, maximum)


def level_beats(beats: np.ndarray) -> np.ndarray:
    """The beats that are not subdivisions of the level around them.

    A beat counts as a subdivision when both gaps to its neighbours are short against the local middle gap. Testing both
    rather than either is what keeps a real change of tempo: at a transition the gap on one side belongs to the old
    tempo and the gap on the other to the new, and a beat that is genuinely twice as fast as its predecessor still has
    a long gap behind it.

    Run to a fixed point, because removing one subdivision shortens the gaps its neighbours are judged by and exposes
    the next.
    """
    keep = np.ones(beats.size, dtype=bool)

    for _ in range(6):
        indices = np.flatnonzero(keep)

        if indices.size < 3:
            break

        gaps = np.diff(beats[indices])
        changed = False

        for position in range(1, indices.size - 1):
            local = np.median(gaps[max(0, position - 3):position + 4])
            before = gaps[position - 1]
            after = gaps[position]

            # Both sides short: this beat sits inside a run of tighter spacing than the level it is in. A beat half the
            # period from one neighbour and a full period from the other is a real beat at a faster tempo.
            if before < local * 0.7 and after < local * 0.7:
                keep[indices[position]] = False
                changed = True

        if not changed:
            break

    return beats[keep]


def soft_targets(labels, torch, sigma: float = 0.7):
    """Gaussian-blurred beat targets, which give a gradient for being nearly right.

    Narrow, at seven tenths of a frame. A beat is an instant and the model's job at a frame next to one is to be nearly
    sure, not to be half sure: a wide target makes a run of frames all worth predicting and the cheapest way to satisfy
    that is to predict a high value everywhere, which is what the first run did - a recall of one and a precision of a
    tenth, which is a model that says yes to everything.
    """
    kernel = torch.arange(-4, 5, dtype=torch.float32, device=labels.device)
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel = kernel / kernel.sum()

    shape = labels.shape
    flat = labels.reshape(-1, 1, shape[-1])
    blurred = torch.nn.functional.conv1d(flat, kernel.view(1, 1, -1), padding=4)

    return blurred.reshape(shape)


# -------------------------------------------------------------------------------------------------------------- parity


def check_parity(reference: Path) -> int:
    """Compares this frontend with a dump from the C# one."""
    raw = reference.read_bytes()

    frames, bins = struct.unpack_from("<ii", raw, 0)
    offset = 8
    spectrogram = np.frombuffer(raw, dtype="<f4", count=frames * bins, offset=offset).reshape(frames, bins)
    offset += frames * bins * 4
    sample_count = struct.unpack_from("<i", raw, offset + 0)[0] if False else None

    # The sample count is written last; read it from the tail.
    sample_count = struct.unpack_from("<i", raw, len(raw) - 4)[0]
    samples = np.frombuffer(raw, dtype="<f4", count=sample_count, offset=offset).copy()

    print(f"C# spectrogram: {frames} x {bins}, {sample_count} samples")

    mine = log_mel(samples)

    print(f"python spectrogram: {mine.shape[0]} x {mine.shape[1]}")

    if mine.shape[0] != frames:
        print(f"FRAME COUNT MISMATCH: python {mine.shape[0]} vs C# {frames}")
        return 1

    difference = np.abs(mine - spectrogram)
    print(f"max abs difference: {difference.max():.6f}")
    print(f"mean abs difference: {difference.mean():.6f}")
    print(f"C# range:     {spectrogram.min():.4f} .. {spectrogram.max():.4f}")
    print(f"python range: {mine.min():.4f} .. {mine.max():.4f}")

    if difference.max() < 1e-3:
        print("PARITY OK")
        return 0

    print("PARITY FAILED - the two frontends disagree")

    # Where they disagree is more useful than how much: a constant tilt is a filterbank difference, a growing error is
    # a framing difference, and a difference in the first frames only is an edge-padding difference.
    per_frame = difference.mean(axis=1)
    per_bin = difference.mean(axis=0)

    print(f"  worst frames: {np.argsort(per_frame)[-5:]}")
    print(f"  worst bins:   {np.argsort(per_bin)[-5:]}")

    return 1


# -------------------------------------------------------------------------------------------------------------- training


def pick_peaks_suppressed(probability: np.ndarray, period: np.ndarray, share: float = 0.6,
                          threshold: float = 0.5) -> np.ndarray:
    """Peaks chosen strongest first, with everything within a share of the predicted period suppressed.

    This is what the period head is for. Taking the local maxima of the curve reports every onset the music has, which
    on the first two runs was two to four times the map's beat count; a frame cannot be a beat if the model itself says
    the beats are half a second apart and there is a louder frame a tenth of a second away.

    Strongest first rather than left to right, because the strongest frame in a neighbourhood is the one the model is
    surest about and a left-to-right walk would take whichever came first. A share below one leaves room for a tempo
    slightly faster than the prediction without letting a subdivision through.
    """
    candidates = [(float(probability[i]), int(i)) for i in np.flatnonzero(probability >= threshold)]
    candidates.sort(reverse=True)

    chosen: list[int] = []

    for value, index in candidates:
        minimum = max(1.0, share * float(period[index]))

        if all(abs(index - other) >= minimum for other in chosen):
            chosen.append(index)

    return np.array(sorted(chosen), dtype=np.int64)


def pick_peaks(probability: np.ndarray, radius: int = 3, threshold: float = 0.5) -> np.ndarray:
    """The frames that would be chosen as beats from a probability curve.

    The same rule the post-processor in ParaTactus uses: a frame at least as large as its neighbours within the radius,
    above the threshold. Applying it here is what makes the validation numbers the ones the player would produce
    rather than a frame-by-frame agreement that no part of the player asks for.
    """
    candidates = np.flatnonzero(probability >= threshold)
    peaks = []

    for index in candidates:
        low = max(0, index - radius)
        high = min(len(probability), index + radius + 1)

        if probability[index] >= probability[low:high].max():
            peaks.append(index)

    return np.array(peaks, dtype=np.int64)


def inspect(dataset: Path) -> int:
    """Reads the manifest and one track, to check the dataset and the frontend fit together."""
    tracks = load_manifest(dataset)
    usable = [t for t in tracks if t.usable]

    print(f"{len(tracks)} tracks in the manifest, {len(usable)} usable")

    if not tracks:
        return 1

    total_frames = sum(t.frames for t in tracks)
    print(f"{total_frames} frames, {total_frames / FPS / 60:.1f} minutes of audio")

    periods = sorted(t.label_period for t in tracks if t.label_period > 0)
    if periods:
        print(f"label periods: min {periods[0]}ms  median {periods[len(periods)//2]}ms  max {periods[-1]}ms")
        print(f"  which is {60000/periods[-1]:.0f} to {60000/periods[0]:.0f} BPM")

    track = tracks[0]
    samples, labels = read_track(track)
    mels = log_mel(samples)

    print("")
    print(f"first track {track.id}: {len(samples)} samples ({len(samples)/SAMPLE_RATE:.1f}s)")
    print(f"  labels {len(labels)} frames, {int(labels.sum())} beats, density {labels.mean():.4f}")
    print(f"  mel    {mels.shape[0]} x {mels.shape[1]}, range {mels.min():.3f} .. {mels.max():.3f}")
    print(f"  frames agree: {mels.shape[0] == len(labels) == track.frames}")

    if mels.shape[0] != len(labels):
        print("  MISMATCH between feature frames and label frames")
        return 1

    # How far apart the labelled beats are, which is what the model is being asked to reproduce and the one sanity
    # check that catches a label set built at the wrong level.
    beats = np.flatnonzero(labels > 0.5)
    if beats.size > 2:
        gaps = np.diff(beats) / FPS * 1000
        print(f"  gaps between labelled beats: median {np.median(gaps):.0f}ms, "
              f"p10 {np.percentile(gaps, 10):.0f}ms, p90 {np.percentile(gaps, 90):.0f}ms")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\dataset"))
    parser.add_argument("--check-parity", type=Path, default=None)
    parser.add_argument("--inspect", action="store_true", help="read the manifest and one track's features, then stop")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--segments-per-track", type=int, default=12)
    parser.add_argument("--segment-frames", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--period-weight", type=float, default=1.0,
                        help="weight of the beat-period regression against the beat classification")
    parser.add_argument("--positive-weight", type=float, default=0.0,
                        help="weight on the positive class; zero or less measures it from the labels")
    parser.add_argument("--out", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\model"))
    parser.add_argument("--cache", type=Path, default=Path(r"D:\Linux\Proj\OsuTest\mel-cache"))
    args = parser.parse_args()

    if args.check_parity is not None:
        return check_parity(args.check_parity)

    if args.inspect:
        return inspect(args.dataset)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} on {device}")

    tracks = load_manifest(args.dataset)
    print(f"{len(tracks)} tracks in the manifest, {sum(1 for t in tracks if t.usable)} usable")

    if not tracks:
        print("nothing to train on")
        return 1

    # Split by track, so no part of a track is in both halves. A random frame split would put half of one performance
    # in each and report a validation loss that says nothing about a track the model has not seen.
    random.Random(1234).shuffle(tracks)
    cut = max(1, int(len(tracks) * 0.85))
    train_tracks, val_tracks = tracks[:cut], tracks[cut:]

    print(f"train {len(train_tracks)} tracks, validate {len(val_tracks)}")

    args.cache.mkdir(parents=True, exist_ok=True)

    # Built once and then cached on disk, and read back per track rather than all at once. A corpus of five hours holds
    # about a gigabyte of mel at float16, and holding every track's copy in memory would put the training run into swap
    # before the first epoch finished. What is held in memory is a bounded most-recently-used set, so that consecutive
    # segments of the same track do not each go back to disk.
    mel_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
    cache_limit = 16

    def features(track: Track) -> np.ndarray:
        if track.id in mel_cache:
            mel_cache.move_to_end(track.id)
            return mel_cache[track.id]

        cached = args.cache / f"{track.id}.npy"

        if cached.exists():
            mels = np.load(cached)
        else:
            samples, _ = read_track(track)
            mels = log_mel(samples).astype(np.float16)
            np.save(cached, mels)

        mel_cache[track.id] = mels

        while len(mel_cache) > cache_limit:
            mel_cache.popitem(last=False)

        return mels

    for i, track in enumerate(train_tracks + val_tracks, 1):
        if not (args.cache / f"{track.id}.npy").exists():
            print(f"  frontend {i}/{len(tracks)}: {track.id}", flush=True)

        mels = features(track)

        if mels.shape[0] != track.frames:
            print(f"  WARNING {track.id}: {mels.shape[0]} feature frames against {track.frames} labels")

    mel_cache.clear()

    # Measured from the labels rather than assumed, because it is what the positive weight is derived from and a
    # weight derived from a guessed rate is a guess. Counted over the training half only, so that nothing about the
    # validation tracks leaks into a constant of the loss.
    def measure_positive_rate(tracks: list[Track]) -> float:
        positive = 0
        total = 0

        for track in tracks:
            _, labels = read_track(track)
            positive += int((labels > 0.5).sum())
            total += labels.size

        return positive / max(1, total)

    model = build_model(torch).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model: {parameters/1e3:.1f}k parameters")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine decay rather than a one-cycle schedule. A one-cycle's warm-up and annealing are sized by the total step
    # count, and at this corpus that is under two hundred steps, so the schedule spends most of the run with the
    # learning rate still rising and the loss never settles - which is what the first run showed, at a loss that moved
    # from 1.38 to 1.22 over thirty epochs while the precision wandered.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, args.epochs))

    def batch(tracks: list[Track], rng: random.Random):
        xs, ys, ps = [], [], []

        for track in tracks:
            mels = features(track)
            _, labels = read_track(track)

            length = min(len(labels), mels.shape[0])

            if length <= args.segment_frames + 10:
                continue

            start = rng.randrange(0, length - args.segment_frames)

            window_labels = labels[start:start + args.segment_frames].copy()

            xs.append(mels[start:start + args.segment_frames].T.astype(np.float32))
            ys.append(window_labels)

            # The period target has to be computed on the window rather than sliced from a whole-track calculation,
            # because a window's first frame has no earlier beat inside the array to measure a gap against and would
            # otherwise be given the whole track's median, which is a different tempo from the window's.
            ps.append(period_targets(window_labels))

        if not xs:
            return None

        return (torch.from_numpy(np.stack(xs)).to(device),
                torch.from_numpy(np.stack(ys)).to(device),
                torch.from_numpy(np.stack(ps)).to(device))

    # The positive rate is a few per cent of frames, so an unweighted loss is minimised by reporting nothing at all, and
    # the weight is what stops that. It is not a free parameter and it is not a constant of the corpus: the weight that
    # makes the two classes worth the same is the negative rate over the positive rate, and it moves whenever the labels
    # do. Sixty was measured against labels that carried three fifths of the beats the maps have, and when the labels
    # were rebuilt to carry all of them the same sixty was three times too much - which showed up exactly as an
    # overweighted positive class does, as a detector reporting 1.15 times the beats and precision down four points.
    positive_rate = measure_positive_rate(train_tracks)
    positive_weight = args.positive_weight if args.positive_weight > 0 else (1.0 - positive_rate) / max(1e-9, positive_rate)
    print(f"labels are {100 * positive_rate:.2f}% positive frames, so the balanced positive weight is {positive_weight:.1f}")
    rng = random.Random(7)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_tracks[:]
        rng.shuffle(order)

        total, seen = 0.0, 0
        steps = 0

        for i in range(0, len(order) - args.batch + 1, args.batch):
            prepared = batch(order[i:i + args.batch], rng)

            if prepared is None:
                continue

            x, y, p = prepared
            logits, period_logits = model(x)
            target = soft_targets(y, torch)

            beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=torch.tensor(positive_weight, device=device))

            # The period is trained where the labels carry one at all; a track with no usable beats anywhere in the
            # window has no tempo to report and is not asked for one.
            defined = p > 0

            if defined.any():
                period_loss = torch.nn.functional.huber_loss(
                    period_logits[defined], torch.log(p[defined]), delta=0.5)
            else:
                period_loss = torch.zeros((), device=device)

            loss = beat_loss + args.period_weight * period_loss

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            total += float(beat_loss)
            steps += 1
            seen += len(x)

        schedule.step()

        # Validation by peaks rather than by frames, because peaks are what a player uses. A frame-level comparison
        # rewards a model for a blurry plateau over a beat - every frame of it counts as a hit - and punishes one for a
        # sharp spike that lands a frame early. What matters is how many of the labels' beats have a chosen beat near
        # them, how many of the chosen beats are real, and how far the middle one is - and the last of those is the one
        # a rhythm game feels, because thirty milliseconds is a visible error to a player and a frame is twenty.
        model.eval()
        hits, chosen, actual = 0, 0, 0
        frame_positive, frame_total = 0, 0
        distances: list[float] = []

        with torch.no_grad():
            for track in val_tracks[:20]:
                prepared = batch([track], rng)

                if prepared is None:
                    continue

                x, y, p = prepared
                logits, period_logits = model(x)
                probability = torch.sigmoid(logits)[0].cpu().numpy()
                period = torch.exp(period_logits)[0].cpu().numpy()
                label = y[0].cpu().numpy()

                frame_positive += int((probability > 0.5).sum())
                frame_total += probability.size

                beat_frames = np.flatnonzero(label > 0.5)

                # Peaks suppressed by the period the model itself predicts, which is the reading the period head
                # exists to make possible: a frame cannot be a beat if the model says the beats are half a second
                # apart and there is a louder frame a tenth of a second away.
                picked_frames = pick_peaks_suppressed(probability, period)

                chosen += len(picked_frames)
                actual += len(beat_frames)

                if beat_frames.size == 0 or picked_frames.size == 0:
                    continue

                # Every chosen beat against its nearest label, rather than every label against the nearest chosen one.
                # Both directions matter but they are not the same number: this one is how far a beat the model did
                # report is from where it belongs, which is what a player hears, and counting it is what makes a
                # spurious beat visible instead of merely lowering coverage.
                order = np.sort(beat_frames)
                index = np.clip(np.searchsorted(order, picked_frames), 1, order.size - 1)
                nearest = np.minimum(np.abs(picked_frames - order[index - 1]),
                                     np.abs(picked_frames - order[index]))

                hits += int((nearest <= 3).sum())
                distances.append(float(np.median(nearest)))

        precision = hits / max(1, chosen)
        recall = hits / max(1, actual)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        middle = float(np.median(distances)) * 1000.0 / FPS if distances else float("nan")

        print(f"epoch {epoch:3}/{args.epochs}  loss {total/max(1,steps):.4f}  "
              f"recall {recall:.3f}  precision {precision:.3f}  f1 {f1:.3f}  "
              f"chosen {chosen/max(1,actual):.2f}x the beats  middle error {middle:5.1f}ms")

    args.out.mkdir(parents=True, exist_ok=True)
    weights = args.out / "detector.pt"
    torch.save(model.state_dict(), weights)
    print(f"saved {weights}")

    # Exported for the player, which runs ONNX Runtime in C# rather than torch. Both heads are exported: the player
    # needs the period to suppress the peaks, and a graph that returned only the beat logits would leave it with a
    # curve it can only threshold.
    onnx_path = args.out / "detector.onnx"
    model.eval().cpu()
    dummy = torch.zeros(1, N_MELS, args.segment_frames)

    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, mel):
            logits, period = self.inner(mel)

            return logits, period

    torch.onnx.export(
        Export(model), dummy, onnx_path,
        input_names=["mel"], output_names=["beat_logit", "period_logit"],
        dynamic_axes={
            "mel": {0: "batch", 2: "frames"},
            "beat_logit": {0: "batch", 1: "frames"},
            "period_logit": {0: "batch", 1: "frames"},
        },
        opset_version=17)

    print(f"saved {onnx_path}")

    # A quick reading of whether the two heads agree with what the labels say, on one held-out track, before anything
    # downstream is pointed at the export. The model has been moved to the processor for the export above, so the batch
    # is rebuilt there rather than on the accelerator it was trained on.
    if val_tracks:
        prepared = batch([val_tracks[0]], rng)

        if prepared is not None:
            x, y, _ = prepared
            x = x.cpu()
            y = y.cpu()

            with torch.no_grad():
                logits, period_logits = model(x)
                probability = torch.sigmoid(logits)[0].cpu().numpy()
                period = torch.exp(period_logits)[0].cpu().numpy()

            label = y[0].cpu().numpy()
            beats = np.flatnonzero(label > 0.5)

            if beats.size > 2:
                print("")
                print(f"  held-out {val_tracks[0].id}: labels median gap "
                      f"{np.median(np.diff(beats)):.0f} frames, model predicts {np.median(period):.0f} frames")

            picked = pick_peaks_suppressed(probability, period)
            print(f"  peaks without suppression {len(pick_peaks(probability))}, with it {len(picked)}, "
                  f"labels {beats.size}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
