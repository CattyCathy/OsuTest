"""Restores the application namespace in the test project after the split.

The split renamed `using OsuTest.Game.Analysis;` to point at the new library, which is right for the library's types
but wrong for the ones that stayed behind - the decoder, the provider and the file entry point. Rather than work out
per file which of the two it needs, both are added everywhere; an unused using costs nothing.

Run from the repository root:
    python tools/fix_test_usings.py
"""

import io
import os
import glob

TESTS = r"OsuTest\OsuTest.Game.Tests"

WANTED = [
    "using OsuTest.BeatAnalysis;",
    "using OsuTest.Game.Analysis;",
]

changed = []

for path in glob.glob(os.path.join(TESTS, "**", "*.cs"), recursive=True):
    with io.open(path, encoding="utf-8-sig") as handle:
        text = handle.read()

    # Generated assembly attributes are not hand edited.
    if path.endswith("AssemblyInfo.cs"):
        continue

    missing = [u for u in WANTED if u not in text]

    if not missing:
        continue

    lines = text.split("\n")
    last_using = -1

    for i, line in enumerate(lines):
        if line.startswith("using "):
            last_using = i

    if last_using < 0:
        continue

    lines[last_using + 1:last_using + 1] = missing

    with io.open(path, "w", encoding="utf-8", newline="\r\n") as handle:
        handle.write("\n".join(lines))

    changed.append(os.path.relpath(path, TESTS))

print(f"{len(changed)} test files updated")
