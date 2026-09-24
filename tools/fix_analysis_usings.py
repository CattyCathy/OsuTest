"""Adds the namespaces the split left missing in the application.

Run from the repository root after extract_analysis_library.py:
    python tools/fix_analysis_usings.py
"""

import io
import os
import glob

APP = r"OsuTest\OsuTest.Game"
LIB_USING = "using OsuTest.BeatAnalysis;"
APP_USING = "using OsuTest.Game.Analysis;"

added = []

for path in glob.glob(os.path.join(APP, "**", "*.cs"), recursive=True):
    with io.open(path, encoding="utf-8-sig") as handle:
        text = handle.read()

    # Files that live in the analysis namespace itself need no using for their own types; everything else needs both,
    # because the types are now split across two namespaces and a caller does not care which is which.
    in_analysis = os.path.dirname(path).endswith("Analysis")

    wanted = [LIB_USING] if in_analysis else [LIB_USING, APP_USING]
    missing = [u for u in wanted if u not in text]

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

    added.append(f"{os.path.basename(path)}: {', '.join(missing)}")

print("usings added:")
for entry in added:
    print("  " + entry)
