# Reference tooling for the Beat This! frontend and postprocessor

These scripts exist to hold the C# beat tracker against the implementation the model was actually published with,
`beat_this` 1.1.0. They are development tools, not part of the build, and nothing in `OsuTest.Game` depends on them.

The reason they exist: the frontend was wrong for a long time in a way nothing local could detect. It applied the
Slaney *area* normalisation to the mel filters, which `torchaudio.transforms.MelSpectrogram` only does for
`norm="slaney"` and not for the `mel_scale="slaney"` the reference asks for, and it omitted the `1/sqrt(n_fft)` scale
that `normalized="frame_length"` applies. Together those tilted the spectrogram by up to 25x across the bands and
shifted it by 32x overall. The model still tracked steady music well enough to look correct, and only produced
unusable beats on dense music, so no test written against synthetic fixtures or against a second copy of the same
analyser could have found it. Comparing against the real reference is what found it, twice.

## Setup

`src/` holds the extracted `beat_this-1.1.0` source and `venv/` a virtualenv with a matching CPU `torch` 2.2.2 and
`torchaudio` 2.2.2 (`numpy<2`). To rebuild it:

```
python -m venv --system-site-packages venv
venv\Scripts\python.exe -m pip install "numpy<2" "torchaudio==2.2.2" ^
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

The versions have to match the installed `torch`; a mismatched `torchaudio` fails to load its extension DLL. The
sdist can be refreshed with `pip download beat-this --no-deps --no-binary :all: -d .`.

## The scripts

Every script that reads a dump needs one first. `ReferenceParityDump.TestDumpForReferenceParity` writes the C# side's
decoded PCM, log-mel spectrogram, model logits and mel filterbank to `OSUTEST_DUMP_DIR`, and the scripts read that
rather than decoding the audio themselves. That is deliberate: feeding both sides the same PCM removes the decoder as
a variable and leaves the frontend, the chunking and the postprocessing to be checked one at a time.

| script | what it answers |
| --- | --- |
| `parity.py` | Do the frontend and the model logits match the published implementation? This is the headline check. |
| `frontend_confirm.py` | Which exact combination of filter normalisation and STFT scaling does the reference use? |
| `frontend_ablation.py` | Which frontend produces better beats, scored against a beatmap's own timing-point grid? |
| `onset_phase.py` | Where are the audio's onsets actually, when the model and a beatmap's grid disagree about phase? |
| `make_logmel_fixture.py` | Regenerates `OsuTest.Game.Tests/Analysis/LogMelFixture.bin`, the committed fixture the standing `LogMelFrontendTest` checks against. |

`parity.py` and `frontend_ablation.py` take the timing-point CSV from
`OsuTest.Game.Tests/Analysis/DesignantTimingPoints.csv` and the ONNX model.

## What the measurements concluded

- The frontend must use **un-normalised** triangular filters (peak 1.0) on the **Slaney frequency scale**, with the
  STFT divided by `sqrt(n_fft)`. `frontend_confirm.py` shows this reproduces torchaudio to 7e-7, while either
  alternative is off by 3.5 in log space.
- The first chunk starts at a **negative** frame and the last chunk's start is **moved left** to end on the piece's last
  frame. `parity.py` prints the resulting schedule.
- Where the model's beats sit ~130ms off this beatmap's grid for 30-46s, `onset_phase.py` shows the audio's onsets are
  at grid+130ms (flux 7.10 against 2.29 at an arbitrary time) - the model is right there and the map's phase is not.
  This is why the residual-against-beatmap test cannot be read as a model error rate.

## Known limits of the model, so they are not re-investigated as bugs

These are not defects in this port. Each was measured, and each is a boundary of the tool rather than of the code
around it. They were found the hard way, one at a time, and re-deriving them costs hours.

### The model cannot represent a tempo above about 215 BPM

The publication's own alternative postprocessor says so outright. `beat_this/model/postprocessor.py` builds its DBN as

```python
DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=50, transition_lambda=100)
```

so even the reference's "better" postprocessor is bounded at 215. Above that the model settles onto a slower pulse
instead of failing visibly, which is the dangerous part: the output looks confident and is wrong.

Measured on **Designant** `<music folder>\DJ - Designant. (Inst.).mp3`, whose beatmap ramps from 150 BPM at
107s to **400 BPM** at 137s:

| time | beatmap | model | ratio |
| --- | --- | --- | --- |
| 120s | 170 | 166.7 | 0.98 |
| 124s | 205 | 176.5 | 0.86 |
| 130s | 260 | 166.7 | 0.64 |
| 136s | 350 | 166.7 | 0.48 |
| 140s | 400 | 176.5 | 0.44 |

`TestRampAgreement` prints this table. The model tracks the gentle part of the ramp (150 to 170 over 107-120s) and then
stops, holding a **flat** interval while the map goes to 400:

```
130.28 360   130.64 360   130.98 340   131.34 360   131.68 340   132.00 320
132.44 320   132.80 360   133.16 360   133.52 360   133.88 360   134.26 380
```

Those are consecutive inter-beat gaps in milliseconds from `TestBeatPlacementDump`, with activation 1.1 to 2.8 - as
confident as anywhere in the track. It is not struggling; it believes the music is at a steady 167 BPM.

**Nothing in the postprocessing can recover this, and this was checked rather than assumed.** The model's activation
halfway between its own beats is *negative* in this passage
(`TestActivationAtBeatMidpoints`, 130-141s: mean -0.748 at midpoints against +1.391 at beats, only 6 of 32 midpoints
above zero), so doubling the beats would invent beats the model does not believe in. And the ratio is not a clean
octave anyway - 360ms against the map's 150-231ms is 1.6x to 2.4x, so it is not a metrical-level error that dropping or
inserting every other beat would fix. Disabling `Prune` entirely changes nothing here, which rules out the one part of
this pipeline that is not the published algorithm.

Recovering such a passage needs a separate tempo-curve tracker driven by the onset envelope rather than by the model's
activation. That is a different tool, not a parameter of this one.

### Two shorter passages on Stage 5 read as brief bursts at double the surrounding tempo

`Lv.4 - Stage 5.mp3`, around 284-299s and 316-330s, holds readings of 500 BPM for up to nine seconds where the music is
at about 200 either side. Unlike the case above these are dense bursts of individually weak peaks, and no
neighbourhood-based period estimate can call them spurious, because a burst long enough to be the majority of a window
is also long enough to be the majority of a wider one. Widening the period window to a span of time rather than a count
of gaps was tried and made it *worse* - the 500 held for nine seconds instead of six - so it was reverted.

`TestLocalTempoStability` prints the readings and how long each one holds, which is what distinguishes a real section
change from a fragment.

## The measurement that matches the actual goal

Tempo agreement is not the property the visuals need. What they need is that **every beat the UI pulses on coincides
with a real beat in the music**; whether the reported tempo is the music's own or half of it does not matter, because
pulsing on every other beat still hits every beat it pulses on. A grid can be perfectly stable in tempo and still be
useless if its beats sit between the music's.

`TestBeatHitRate` measures that directly, against neither the beatmap nor the model but the audio: it builds an onset
envelope from the samples, samples it at each beat, and reports the ratio to the envelope's own mean plus the share of
beats above that mean. Typical values: **0.5-1.1 with under 60% above the mean means the beats are not on onsets**;
1.3-3.0 with 80-100% means they are.

This is what found the phase errors that no tempo measurement could see:

| track | section | model | beatmap's own grid |
| --- | --- | --- | --- |
| Designant | 150s | 1.00 / 48% | 1.14 / 76% |
| Designant | 160s | 1.09 / 63% | 1.21 / 91% |
| Stage 5 | 40s | 0.56 / 4% | 0.74 / 24% |
| Stage 5 | 50s | 0.61 / 8% | 0.89 / 45% |

### Do not use this metric as an objective function. It was tried and it backfired.

Those rows say the model's beats sit between the onsets, and the obvious next step is to move them onto the onsets. That
was implemented as `BeatGridRefiner` and it was **wrong**. It improved this metric on every track and every section, and
it made the beats worse.

The reason is that this metric cannot judge a change that optimises it. The strongest onsets in most music are not on
the beat - and where the beatmap is written at double the music's tempo, the off-beats are onsets too - so a grid
allowed to move by half a period simply walks onto the off-beats and scores better for it. Measured against the
beatmap, the refined grid's signed offset had a **median of -152ms**, which at the map's 300ms beat is half a beat
exactly, and the share of beats more than 60ms from a map grid line went from **161 in 453 to 229**.

Constraining the search to a quarter period did not rescue it either: the onset gain mostly disappeared (overall 1.30x
down to 1.23x, and the 40s improvement gone entirely) while the grid still sat a median of 102ms off the map.

So the metric is a good *diagnostic* and a bad *target*. Anything that changes beat placement has to be checked
against a second opinion - the beatmap, or the audio by a method that is not the one being optimised - because agreeing
with yourself proves nothing. This is the same trap the previous analyser fell into, in a different costume.

`BeatGridRefiner.cs` was deleted. Phase correction on this model is an open problem, not a solved one.

## Three tracks, current state

| track | model hit rate | beatmap's grid | notes |
| --- | --- | --- | --- |
| `Junk - Turning POINT.mp3` | 1.36x | 0.65x | clean throughout |
| `DJ - Designant. (Inst.).mp3` | 1.19x | 1.04x | 122-141s unrecoverable, see above |
| `Lv.4 - Stage 5.mp3` | 1.48x | 0.97x | two burst passages, see above |

The beatmap's own grid is not ground truth: it is usually written at double the music's tempo, so half its lines are
not onsets at all, and in places its phase does not match the audio. That is why the conclusions above are drawn from
the audio. But it is the only independent opinion available on which side of the beat a grid should sit, which is
exactly what the section above is about.

## The cache will silently undo every fix, and did

`BeatGridCache` stores analysed grids on disk. Its key was originally a hash of the audio file and the model file, and
that is not enough: it cannot distinguish a grid produced by one version of the analysis from a grid produced by
another. The consequence was that a whole session of corrections - the mel frontend, the chunk schedule, the pruning
and the phase refinement - changed nothing at all in the player for any track that had already been played once. The
code was fixed and the tests went green, because the tests call the analysis directly and never touch the cache, while
the application read a grid written before any of it.

It surfaced as "Designant is completely wrong" while the measurements said Designant was fine. The key for that track
without the version was `eeb2c691aa60ebc6af36177c7ea2ac3b`, which named a file in `%LOCALAPPDATA%\OsuTest\beatgrids`
written hours earlier and holding 405 beats where the current analysis produces 453.

The key now includes the build identity of the analysis assembly, so it changes exactly when the code that produces
beats changes and not otherwise. **If beats ever look stale again, delete `%LOCALAPPDATA%\OsuTest\beatgrids` before
investigating anything else** - it costs one re-analysis per track and rules out the entire class.

