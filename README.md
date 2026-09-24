# OsuTest

A worked example of driving animation from music with [ParaTactus](https://github.com/CattyCathy/ParaTactus): the beat
tracker analyses a track, and a player pulses on the beats it reports — including where the tempo changes, which is the
whole point of the exercise.

It is a **sample, not a product**. Nothing here is maintained as software to deploy. The point is that the animation
follows the music, and that what the tracker does when it is wrong can be seen rather than argued about.

## Running it

```powershell
dotnet run --project OsuTest/OsuTest.Desktop
```

The library is consumed as packages rather than as a project, because it lives in its own repository. Until it is
published, the two packages are committed in `packages-local/` and `nuget.config` points at that folder, so a clean
clone restores and builds without publishing anything first.

## The model

A model is the one thing you have to fetch. Download `beat-this-final0-int8.onnx` (20.9 MB) from the releases of
[`beat-this-onnx`](https://github.com/CattyCathy/beat-this-onnx) and put it in your Downloads folder, or set
`OSUTEST_MODEL` to its full path. The float build works the same way; the quantised one is what this application has
actually been run against, and the difference between the two is measured in that repository's README.

The app resolves the path in this order, and if it finds nothing it says what it looked for:

1. `OSUTEST_MODEL`, when it is set — and it is an error, not a fallback, if that file does not exist;
2. `%USERPROFILE%\Downloads\beat-this-final0-int8.onnx` (`$HOME` elsewhere);
3. the same folder's `beat-this-final0.onnx`.

To the library the model is a **path**, never a bundled resource — which is why it is not in this repository:

```csharp
var provider = new BeatGridProvider(modelPath, cacheDirectory, BassAudioDecoder.Default);

BeatGrid grid = provider.Get(audioPath);        // analysed once, then cached on disk
double bpm = grid.BpmAt(timeInMilliseconds);    // the tempo at a time, following real tempo changes
IReadOnlyList<double> beats = grid.Beats;       // the beat instants, in milliseconds
```

Any ONNX graph with the input and output names documented in
[`ParaTactus/docs/model.md`](https://github.com/CattyCathy/ParaTactus/blob/main/docs/model.md) will run; the file is
opaque to the library. The tests read `OSUTEST_MODEL` as well, but default to the float build rather than the quantised
one.

## What is where

| | |
| --- | --- |
| `OsuTest.Desktop/` | the entry point: window, configuration, game host |
| `OsuTest.Game/` | the scene — the player, the beat-synced visuals, and the demo that ties them to the tracker |
| `OsuTest.Game.Tests/` | the tests, and the ground truth they are checked against |
| `OsuTest.Resources/` | fonts and textures, taken from osu-resources — see the licence below |
| `tools/ref/` | the reference scripts that held this port against the original implementation, and the notes on what they measured |

## Tests

```powershell
dotnet test OsuTest/OsuTest.Game.Tests
```

Some of them need material that cannot be committed: set `OSUTEST_AUDIO` to a track and `OSUTEST_MODEL` to an ONNX file,
and they measure the tracker against the beatmap's own timing points. The rest run without either.

`OsuTest.Game.Tests/Analysis/DesignantTimingPoints.csv` is the uninherited timing-point list of the beatmap
*Designant.* by Murumoo (beatmap 4870830, set 2283870), committed as the one external fact the suite is checked
against. The tracker's measured limits — the tactus ceiling, the low-passed tempo ramps, the unrecoverable phase —
came out of that comparison, and are written up in the library.

## Licence and credits

**Licensed in parts, and the parts differ.** Read [`LICENSE.md`](LICENSE.md) before using anything here.

- The **code** is MIT.
- The **bundled osu! assets** (`OsuTest.Resources` fonts and texture) are CC BY-NC 4.0 and **non-commercial only**,
  credited to ppy Pty Ltd and the osu! contributors.
- Nothing here grants any right to the **osu!** or **ppy** names. This is a third-party project, not affiliated with or
  endorsed by ppy.

[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) covers everything the build pulls in, including BASS, which is
proprietary and free for non-commercial use only.
