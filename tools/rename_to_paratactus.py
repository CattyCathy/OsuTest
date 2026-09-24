#!/usr/bin/env python3
"""Rename Tactus to ParaTactus, in both repositories.

The name Tactus is the music-theory term for the beat a listener taps, which is what made it a good description of the
library and a bad package identifier: publishing under it takes the ordinary word out of circulation for anyone else
searching for it. ParaTactus keeps the meaning and stops competing with the term, and it matches the naming already used
for the other projects here (Para, Para Osiris).

What that means for each identifier, which is not the same thing as renaming the word everywhere:

    repository        Tactus              -> ParaTactus
    directories       Tactus/ Tactus.Bass/ Tactus.Tests/  -> ParaTactus/ ParaTactus.Bass/ ParaTactus.Tests/
    package   id      Tactus.NET          -> ParaTactus.NET
    package   id      Tactus.Bass         -> ParaTactus.Bass
    assembly+namespace Tactus             -> ParaTactus

The adapter's namespace stays `Decoding` rather than becoming `ParaTactus.Bass`, for the reason recorded in
`ParaTactus.Bass/BassAudioDecoder.cs`: a namespace ending in the same word as a type shadows it, and `Bass.CreateStream`
would resolve to the namespace instead of to `ManagedBass.Bass`.

The replacement is anchored with a negative lookbehind so that running this twice cannot produce `ParaParaTactus`.

`tools/` in the application repository is left alone. It holds the record of the migrations that actually happened,
including the one that introduced the name Tactus, and rewriting it would falsify that record.
"""

import os
import re
import shutil
import subprocess
import sys

OLD_REPO = r"D:\Linux\Proj\Tactus"
NEW_REPO = r"D:\Linux\Proj\ParaTactus"
APP = r"D:\Linux\Proj\OsuTest"
LOCAL_FEED = os.path.join(APP, "packages-local")

RENAMES = {"Tactus": "ParaTactus", "Tactus.Bass": "ParaTactus.Bass", "Tactus.Tests": "ParaTactus.Tests"}
PACKAGES = {"ParaTactus": "ParaTactus.NET", "ParaTactus.Bass": "ParaTactus.Bass"}
STALE_PACKAGES = ["Tactus.NET.1.0.0.nupkg", "Tactus.Bass.1.0.0.nupkg"]

TOKEN = re.compile(r"(?<!Para)Tactus")
EXTENSIONS = {".cs", ".csproj", ".slnx", ".md", ".txt", ".props", ".targets", ".config", ".gitignore"}
SKIP_DIRS = {".git", "bin", "obj", "__pycache__", "venv", "node_modules"}


def read(path):
    with open(path, "rb") as handle:
        data = handle.read()

    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encoding = "utf-16"
    elif data[:3] == b"\xef\xbb\xbf":
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"

    return data.decode(encoding), encoding


def rewrite(path):
    text, encoding = read(path)
    renamed = TOKEN.sub("ParaTactus", text)

    if renamed == text:
        return False

    with open(path, "w", encoding=encoding, newline="") as handle:
        handle.write(renamed)

    return True


def rename_within_repo():
    if os.path.isdir(NEW_REPO):
        sys.exit(f"{NEW_REPO} already exists")

    if not os.path.isdir(OLD_REPO):
        sys.exit(f"{OLD_REPO} does not exist")

    os.rename(OLD_REPO, NEW_REPO)
    print(f"renamed {OLD_REPO} -> {NEW_REPO}")

    for directory, replacement in RENAMES.items():
        source = os.path.join(NEW_REPO, directory)
        target = os.path.join(NEW_REPO, replacement)

        if os.path.isdir(source):
            os.rename(source, target)

    for base, dirs, files in os.walk(NEW_REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for name in files:
            path = os.path.join(base, name)

            for old, new in RENAMES.items():
                if name == old + os.path.splitext(name)[1] and name.startswith(old + "."):
                    renamed = os.path.join(base, new + os.path.splitext(name)[1])
                    os.rename(path, renamed)
                    path = renamed
                    name = os.path.basename(renamed)
                    break

            if os.path.splitext(name)[1].lower() in EXTENSIONS or name == ".gitignore":
                rewrite(path)

    print("rewrote the namespaces, package identifiers and documents inside the repository")


def rewrite_app():
    changed = 0

    for base, dirs, files in os.walk(os.path.join(APP, "OsuTest")):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for name in files:
            if os.path.splitext(name)[1].lower() in EXTENSIONS and rewrite(os.path.join(base, name)):
                changed += 1

    for name in ("LICENSE.md", "THIRD-PARTY-NOTICES.md"):
        path = os.path.join(APP, name)

        if os.path.isfile(path) and rewrite(path):
            changed += 1

    print(f"rewrote {changed} files in the application repository")


def repack():
    for stale in STALE_PACKAGES:
        path = os.path.join(LOCAL_FEED, stale)

        if os.path.isfile(path):
            os.remove(path)
            print(f"removed the stale {stale}")

    for project, package in PACKAGES.items():
        subprocess.run(
            ["dotnet", "pack", project, "-c", "Release", "-o", LOCAL_FEED],
            cwd=NEW_REPO,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    print(f"packed {' and '.join(PACKAGES.values())} into {os.path.relpath(LOCAL_FEED, APP)}")


def restage():
    subprocess.run(["git", "add", "-A"], cwd=NEW_REPO, check=True)
    print("staged the renamed files in the library repository (nothing committed)")


if __name__ == "__main__":
    rename_within_repo()
    rewrite_app()
    repack()
    restage()
