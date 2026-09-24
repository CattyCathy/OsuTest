# Third-party notices

Licensing for this repository's own code and bundled assets is in `LICENSE.md`. This file covers what the project
depends on.

## Packages

`OsuTest.Game` references `ppy.osu.Framework` 2026.921.0 and `Microsoft.ML.OnnxRuntime` 1.30.0 directly for playback
and inference, and consumes `ParaTactus.NET` and `ParaTactus.Bass` — the beat analysis and its optional BASS decoder — as
packages from `packages-local/` until they are published. Everything below is the restored closure of that; the
identifier is the one each package declares in its own `.nuspec`.

### Permissive

| Package | Licence |
| --- | --- |
| `ppy.osu.Framework`, `ppy.osu.Framework.SourceGeneration` | MIT |
| `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.Managed` | MIT |
| `Microsoft.Diagnostics.NETCore.Client`, `Microsoft.Diagnostics.Runtime` | MIT |
| `Microsoft.Extensions.DependencyInjection`, `.Abstractions`, `Microsoft.Extensions.ObjectPool` | MIT |
| `Newtonsoft.Json`, `NUnit`, `JetBrains.Annotations`, `PolySharp` | MIT |
| `SharpGen.Runtime`, `SharpGen.Runtime.COM`, `SharpFNT` | MIT |
| `Vortice.D3DCompiler`, `Vortice.Direct3D11`, `Vortice.DirectX`, `Vortice.DXGI`, `Vortice.Mathematics` | MIT |
| `System.ComponentModel.Annotations`, `System.Numerics.Tensors` | MIT |
| `ppy.SDL3-CS`, `ppy.Veldrid`, `ppy.Veldrid.MetalBindings`, `ppy.Veldrid.OpenGLBindings`, `ppy.Veldrid.SPIRV` | MIT |
| `Markdig` | BSD-2-Clause |
| `StbiSharp` | BSD-3-Clause |
| `HidSharpCore`, `ppy.osuTK.NS20`, `ppy.SDL2-CS`, `ppy.Vk`, `ppy.managed-midi` | no identifier declared in the package; MIT or Apache-2.0 upstream, verify before relying on it |

### Copyleft

| Package | Licence | Note |
| --- | --- | --- |
| `OpenTabletDriver`, `OpenTabletDriver.Configurations`, `OpenTabletDriver.Native`, `OpenTabletDriver.Plugin` | LGPL-3.0-or-later | tablet input, via `osu.Framework` |
| `FFmpeg.AutoGen` | LGPL-3.0 | the .NET bindings; FFmpeg itself is LGPL-2.1-or-later or GPL depending on the build |

A self-contained distribution of an LGPL-3.0 library has to let the recipient relink against a modified version: ship
the licence texts, keep the assemblies separate rather than merged, and offer the corresponding source.

### Non-open-source

| Package | Licence | Note |
| --- | --- | --- |
| BASS (`libbass.so`, `bass.dll`, via `ppy.osu.Framework.NativeLibs`) | proprietary, un4seen developments | free for non-commercial use only; a commercial product needs a licence from un4seen (shareware EUR 125 / single commercial EUR 950 / unlimited EUR 3,450). The NuGet package declares MIT, which is the licence ppy distributes under and does not reach the third-party binaries inside it; the package's maintainers state that it is meant for osu!'s own consumption and that consumers must do their own diligence. |
| `SixLabors.ImageSharp` 3.1.11 | Six Labors Split License 1.0 | free for open-source projects and small organisations, paid above that; used by `osu.Framework` for texture loading |

**This application decodes audio with BASS, so it is a non-commercial product as distributed.** Making it commercial
means buying a BASS licence or replacing the decoder — see the note below.

## The decode path

The analysis — `ParaTactus.NET`: `BeatThisBeatTracker`, `StreamingBeatTracker`, the FFT, the grid and the regulariser —
depends only on `Microsoft.ML.OnnxRuntime`, three MIT packages in total, and takes mono samples as a `float[]`. The
decode is a separate package, `ParaTactus.Bass`, and it is the only thing in this project that is not open source: it binds
BASS through `osu.Framework`, for the file callbacks used to decode a stream that has no path.

That is deliberate rather than incidental. Decoding used to happen inside the analysis, which meant the analysis could
not honestly be published as MIT; it now sits behind `IAudioDecoder` in a package a consumer has to choose.

To take BASS out of the build, supply your own `IAudioDecoder` to `BeatGridProvider` and drop the `ParaTactus.Bass`
reference. The application still needs `osu.Framework` for playback either way, so its own position on BASS is set by
`BassAudioSource` and not by this library.

## Model

The network is not in this repository. It is **Beat This!** by Francesco Foscarin, Jan Schlüter and Gerhard Widmer, MIT
licensed with a specific copyright holder that has to be reproduced:

```
MIT License

Copyright (c) 2024 Institute of Computational Perception, JKU Linz, Austria
```

- Paper: *Beat This! Accurate Beat Tracking Without DBN Postprocessing*, ISMIR 2024 — https://arxiv.org/abs/2407.21658
- Code and checkpoints: https://github.com/CPJKU/beat_this

The `final0` checkpoint is the one every measurement in this project was taken with, exported to ONNX and quantised to
int8. A quantised build is a derivative of the checkpoint and carries the same notice. The library keeps the licence text
and the SHA-256 of both builds in `OsuTest/ParaTactus/MODEL-LICENCE.txt`.

## Vendored reference source

`tools/ref/src/beat_this-1.1.0/` and `tools/ref/beat_this-1.1.0.tar.gz` are the unmodified `beat_this` 1.1.0 source
distribution, kept so the C# port can be held against the implementation the model was published with. It is MIT
licensed and its copyright notice is retained at `tools/ref/src/beat_this-1.1.0/LICENSE`. The compiled `.pyc` files
committed alongside it are build artifacts of that source and carry the same licence.
