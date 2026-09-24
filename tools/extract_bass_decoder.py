#!/usr/bin/env python3
"""Point the application and its tests at the extracted Tactus.Bass decoder.

The decoder moved out of the analysis so that the analysis has no proprietary dependency; this updates everything that
was reaching for it. Three kinds of change:

* `AudioDecoder` becomes `BassAudioDecoder`, which now lives in `Tactus.Bass`. The lookbehind keeps it from renaming
  the replacement it just made.
* `BeatThisBeatTracker.BeatTimes(audio, model)` and `new BeatGridProvider(...)` took a path and decoded it themselves.
  They now take a decoder, and get `BassAudioDecoder.Default`.
* Both projects gain a reference to the new project, and the solution filters list it beside the analysis.

Idempotent: every replacement is a no-op the second time.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSUMERS = ["OsuTest.Game", "OsuTest.Game.Tests", "OsuTest.Desktop", "OsuTest.iOS"]
USING = "using Tactus.Bass;"

SKIP_DIRS = {"bin", "obj", ".git", "headless", "venv", "node_modules"}

PROVIDER_CALLS = [
    ("new BeatGridProvider(model, directory)", "new BeatGridProvider(model, directory, BassAudioDecoder.Default)"),
    ("new BeatGridProvider(model, cache)", "new BeatGridProvider(model, cache, BassAudioDecoder.Default)"),
    (
        "new BeatGridProvider(defaultModelPath(), defaultCacheDirectory())",
        "new BeatGridProvider(defaultModelPath(), defaultCacheDirectory(), BassAudioDecoder.Default)",
    ),
]

PROJECT_REFERENCES = [
    ("OsuTest.Game", r"..\Tactus\Tactus.csproj"),
    ("OsuTest.Game.Tests", r"..\Tactus\Tactus.csproj"),
]


def read_text(path):
    with open(path, "rb") as handle:
        data = handle.read()

    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16"), "utf-16"

    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"

    return data.decode("utf-8"), "utf-8"


def write_text(path, text, encoding):
    with open(path, "w", encoding=encoding, newline="") as handle:
        handle.write(text)


def add_using(text):
    """Insert the using for the decoder after the last using that precedes the namespace."""
    if USING in text:
        return text

    lines = text.split("\n")
    last = -1
    namespace = len(lines)

    for index, line in enumerate(lines):
        if line.startswith("namespace "):
            namespace = index
            break

        if line.startswith("using "):
            last = index

    if last < 0:
        lines.insert(namespace, USING)
    else:
        lines.insert(last + 1, USING)

    return "\n".join(lines)


def update_source(path):
    text, encoding = read_text(path)
    original = text

    text = re.sub(r"(?<!Bass)AudioDecoder\.", "BassAudioDecoder.", text)
    text = text.replace(
        "BeatThisBeatTracker.BeatTimes(audio, model)",
        "BeatThisBeatTracker.BeatTimes(audio, model, BassAudioDecoder.Default)",
    )

    for old, new in PROVIDER_CALLS:
        text = text.replace(old, new)

    if "BassAudioDecoder" in text:
        text = add_using(text)

    if text == original:
        return False

    write_text(path, text, encoding)

    return True


def update_project_references():
    for project, reference in PROJECT_REFERENCES:
        path = os.path.join(ROOT, "OsuTest", project, project + ".csproj")
        text, encoding = read_text(path)

        if "Tactus.Bass" in text:
            continue

        marker = f'    <ProjectReference Include="{reference}" />'
        addition = marker + f'\n    <ProjectReference Include="..\\Tactus.Bass\\Tactus.Bass.csproj" />'

        if marker not in text:
            sys.exit(f"could not find the analysis reference in {path}")

        write_text(path, text.replace(marker, addition), encoding)
        print(f"added the decoder reference to {project}.csproj")


def update_solution_filters():
    for name in os.listdir(os.path.join(ROOT, "OsuTest")):
        if not name.endswith(".slnf"):
            continue

        path = os.path.join(ROOT, "OsuTest", name)
        text, encoding = read_text(path)
        marker = '"Tactus\\\\Tactus.csproj",'

        if "Tactus.Bass" in text or marker not in text:
            continue

        write_text(path, text.replace(marker, marker + '\n      "Tactus.Bass\\\\Tactus.Bass.csproj",'), encoding)
        print(f"added the decoder project to {name}")


if __name__ == "__main__":
    changed = 0

    for consumer in CONSUMERS:
        base = os.path.join(ROOT, "OsuTest", consumer)

        for directory, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for file in files:
                if file.endswith(".cs") and update_source(os.path.join(directory, file)):
                    changed += 1

    print(f"updated {changed} source files")
    update_project_references()
    update_solution_filters()
