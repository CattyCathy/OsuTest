#!/usr/bin/env python3
"""Correct the licence wording in both repositories after the split, and after ppy's reply.

Two things to fix.

**The claim about the NativeLibs package.** It was written as "a known packaging error" and "wrong", which says ppy
mislabeled something. Their maintainers' position is that the package is intended for osu!'s own consumption and that
its MIT declaration is the licence *they* distribute under, with consumers expected to do their own diligence. That is
a fair description of the declaration's scope, and the accurate statement is narrower than the one here was: a licence
can only be granted by whoever holds the rights, so a declaration by the package's publisher cannot extend to the
third-party binaries inside it. The warning does not change - BASS is proprietary and the consumer's obligations are
un4seen's - but the accusation does, and an inaccurate accusation in a licence file is worth less than an accurate one.

**Paths and structure.** The application no longer contains the library: it consumes it as a package, and the licence
file still describes the old layout.
"""

import os
import sys

APP = r"D:\Linux\Proj\OsuTest"
REPO = r"D:\Linux\Proj\Tactus"

BASS_ADAPTER_OLD = """The native libraries arrive in `ppy.osu.Framework.NativeLibs`, whose package metadata declares MIT. **That declaration is
wrong for its payload** — a known packaging error, reported as ppy/osu-framework#6388 — and the package ships no licence
text of its own to correct it. Treat the BASS binaries as proprietary no matter what the NuGet page says."""

BASS_ADAPTER_NEW = """The native libraries arrive in `ppy.osu.Framework.NativeLibs`, whose NuGet metadata declares MIT. That declaration is
about the package ppy distributes, and it cannot reach the third-party binaries inside it: a licence can only be granted
by whoever holds the rights, and what governs BASS is un4seen's terms, not a field in someone else's `.nuspec`. The
package's maintainers have said as much — that it is intended for osu!'s own consumption, and that anyone consuming it
has to do their own due diligence. This file is that diligence, done once and written down. Treat the BASS binaries as
proprietary whatever the NuGet page says."""

BASS_APP_OLD = """The NuGet package declares MIT for its payload, which is a known packaging error (ppy/osu-framework#6388). |"""

BASS_APP_NEW = """The NuGet package declares MIT, which is the licence ppy distributes under and does not reach the third-party binaries inside it; the package's maintainers state that it is meant for osu!'s own consumption and that consumers must do their own diligence. |"""

README_OLD = """BASS is free for non-commercial use and licensed per product otherwise, and its terms require a licensed product to be
an end-user product rather than a component of another product. Read `THIRD-PARTY-NOTICES.md` in this directory — the
NuGet metadata for the package that carries the native BASS binaries says MIT, and that is wrong."""

README_NEW = """BASS is free for non-commercial use and licensed per product otherwise, and its terms require a licensed product to be
an end-user product rather than a component of another product. Read `THIRD-PARTY-NOTICES.md` in this directory: the
NuGet metadata on the package that carries the native BASS binaries declares MIT, which is the licence its publisher
distributes under and does not cover the binaries themselves."""

NAMESPACE_NOTE = """The package is `Tactus.Bass`; the namespace is `Tactus.Decoding`. A namespace ending in the same word as a type shadows it
for everything declared inside it, so `Tactus.Bass` present in scope would make `Bass.Init` resolve to the namespace
rather than to `ManagedBass.Bass`. Only the namespace avoids that.

"""

LICENSE_OLD = """Everything written for this project, in `OsuTest/Tactus`, `OsuTest/Tactus.Bass`, `OsuTest/OsuTest.Game`,
`OsuTest/OsuTest.Desktop`, `OsuTest/OsuTest.Game.Tests` and `tools/`, is MIT licensed:"""

LICENSE_NEW = """Everything written for this project, in `OsuTest/OsuTest.Game`, `OsuTest/OsuTest.Desktop`,
`OsuTest/OsuTest.Game.Tests` and `tools/`, is MIT licensed:"""

REPO_NOTE_OLD = """`OsuTest/Tactus` is also published as its own repository and NuGet package, `Tactus.NET`, under MIT, together with
`Tactus.Bass` — the optional decoder package, whose own source is MIT but which depends on the proprietary BASS library.
The copies here are the working copies. The analysis inside `Tactus.NET` depends on nothing but three MIT packages; the
proprietary dependency belongs to `Tactus.Bass` alone, which is why the two are separate packages rather than one."""

REPO_NOTE_NEW = """The beat analysis is not in this repository. It is a separate repository — `Tactus.NET` for the analysis, and
`Tactus.Bass` for its optional decoder — consumed here as packages, packed into `packages-local/` until they are
published. Both are MIT as source; `Tactus.Bass` depends on the proprietary BASS library, which is why the two are
separate packages rather than one, and why the analysis itself has no proprietary dependency at all."""

APP_NOTICES_OLD = """`OsuTest.Game` references two projects of its own — `Tactus.NET`, the beat analysis, and `Tactus.Bass`, its optional
BASS decoder — and references `ppy.osu.Framework` 2026.921.0 and `Microsoft.ML.OnnxRuntime` 1.30.0 directly for
playback and inference. The full restored closure is below; the identifier is the one each package declares in its own
`.nuspec`."""

APP_NOTICES_NEW = """`OsuTest.Game` references `ppy.osu.Framework` 2026.921.0 and `Microsoft.ML.OnnxRuntime` 1.30.0 directly for playback
and inference, and consumes `Tactus.NET` and `Tactus.Bass` — the beat analysis and its optional BASS decoder — as
packages from `packages-local/` until they are published. Everything below is the restored closure of that; the
identifier is the one each package declares in its own `.nuspec`."""

REPLACEMENTS = [
    (os.path.join(REPO, "Tactus.Bass", "THIRD-PARTY-NOTICES.md"), BASS_ADAPTER_OLD, BASS_ADAPTER_NEW),
    (os.path.join(REPO, "Tactus.Bass", "README.md"), README_OLD, README_NEW),
    (os.path.join(APP, "THIRD-PARTY-NOTICES.md"), BASS_APP_OLD, BASS_APP_NEW),
    (os.path.join(APP, "THIRD-PARTY-NOTICES.md"), APP_NOTICES_OLD, APP_NOTICES_NEW),
    (os.path.join(APP, "LICENSE.md"), LICENSE_OLD, LICENSE_NEW),
    (os.path.join(APP, "LICENSE.md"), REPO_NOTE_OLD, REPO_NOTE_NEW),
]


def read(path):
    with open(path, "rb") as handle:
        data = handle.read()

    encoding = "utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig" if data[:3] == b"\xef\xbb\xbf" else "utf-8"

    return data.decode(encoding), encoding


def main():
    # The namespace note goes near the top of the adapter's readme, after the first paragraph.
    path = os.path.join(REPO, "Tactus.Bass", "README.md")
    text, encoding = read(path)

    if "the namespace is `Tactus.Decoding`" not in text:
        marker = "The analysis is defined on samples."
        text = text.replace(marker, NAMESPACE_NOTE + marker, 1)

        with open(path, "w", encoding=encoding, newline="") as handle:
            handle.write(text)

        print(f"added the namespace note to {os.path.relpath(path, REPO)}")

    for path, old, new in REPLACEMENTS:
        text, encoding = read(path)

        # Matched on the last line, which differs between the old and new wording. The first line does not: both
        # versions open with the same sentence, which is what made an earlier version of this script skip a file it
        # had not touched.
        if new.strip().split("\n")[-1].strip() in text:
            print(f"already updated: {os.path.relpath(path)}")
            continue

        if old not in text:
            sys.exit(f"could not find the text to replace in {path}")

        with open(path, "w", encoding=encoding, newline="") as handle:
            handle.write(text.replace(old, new))

        print(f"updated {os.path.relpath(path)}")

    print("done")


if __name__ == "__main__":
    main()
