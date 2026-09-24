# Licensing

> **Reference only.** This repository is published so that the work can be read, checked and cited. It is not a
> starting point for another project: please do not clone it to build on, and do not redistribute it or a build of it.
> What actually binds you is the licensing below — read it before you use anything here.

This repository is licensed **in parts**, and the parts carry different terms. Read this before you use or redistribute
anything here.

**The repository as a whole — and any binary built from it — may be used for non-commercial purposes only**, because it
bundles artwork that is licensed non-commercially. That constraint comes from the assets, not from the code.

## 1. Source code — MIT

Everything written for this project, in `OsuTest/OsuTest.Game`, `OsuTest/OsuTest.Desktop`,
`OsuTest/OsuTest.Game.Tests` and `tools/`, is MIT licensed:

```
MIT License

Copyright (c) 2026 CattyCathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The beat analysis is not in this repository. It is a separate repository — `ParaTactus.NET` for the analysis, and
`ParaTactus.Bass` for its optional decoder — consumed here as packages, packed into `packages-local/` until they are
published. Both are MIT as source; `ParaTactus.Bass` depends on the proprietary BASS library, which is why the two are
separate packages rather than one, and why the analysis itself has no proprietary dependency at all.

## 2. Bundled assets — CC BY-NC 4.0, non-commercial

`OsuTest/OsuTest.Resources/Fonts/*` (the pre-rendered Noto font atlases) and
`OsuTest/OsuTest.Resources/Textures/logo.png` come from [ppy/osu-resources](https://github.com/ppy/osu-resources), whose
README states:

> The majority of content in this repository is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
> …you can use it in a non-commercial manner.
>
> Some fonts have separate licencing; please ensure to check their local licence files before distributing them.

So: **you may not use these assets, or a build that contains them, commercially.** The upstream Noto fonts are under the
SIL Open Font License 1.1, but what is committed here is osu!framework's rendered font stores taken from osu-resources,
so treat them as CC BY-NC 4.0 unless you have checked the upstream licence files yourself.

Attribution: ppy Pty Ltd and the osu! contributors, CC BY-NC 4.0.

## 3. Trademarks

Nothing here grants any right to the **osu!** or **ppy** names or branding, which are protected by trademark. The same
upstream README says so explicitly. This is a third-party project and is not affiliated with or endorsed by ppy.

## 4. Test data derived from a beatmap

`OsuTest/OsuTest.Game.Tests/Analysis/DesignantTimingPoints.csv` is the uninherited timing-point list extracted from the
osu! beatmap *Designant.* by **Murumoo** (beatmap 4870830, set 2283870), committed as a ground truth for the tempo tests
and credited here. It is a list of tempo values and offsets, not the beatmap or its audio. If you would rather not carry
third-party content, it can be replaced by a synthetic timing grid; the tests read it through one helper.

## 5. Dependencies

Third-party packages and their terms — including BASS, which is proprietary and only free for non-commercial use, and is
reached through `osu.Framework` by the audio-decoding path — are listed in `THIRD-PARTY-NOTICES.md`.

## Why not GPL-3.0

"Stricter" cannot be expressed by switching this repository to GPL. CC BY-NC 4.0's non-commercial restriction is an
*additional* restriction, and GPL-3.0 section 7 forbids adding restrictions on top of it, so content under CC BY-NC 4.0
and code under GPL-3.0 cannot be distributed together at all. Copyleft and non-commercial are different axes; the assets
here are non-commercial, and the code is permissive, and the two coexist only because the permissive side does not add
conditions the non-commercial side forbids.

If the intent is for the *application code itself* to be non-commercial, the compatible choice is a purpose-built
non-commercial software licence such as PolyForm Noncommercial 1.0.0, replacing section 1 above.
