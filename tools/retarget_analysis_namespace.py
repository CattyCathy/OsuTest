"""Points every remaining reference at the one namespace after the analysis moved out of the application.

Run from the repository root:
    python tools/retarget_analysis_namespace.py
"""

import io
import glob

LIB_USING = "using OsuTest.BeatAnalysis;"
OLD_USING = "using OsuTest.Game.Analysis;"

renamed = 0
retargeted = 0

for path in glob.glob(r"OsuTest\**\*.cs", recursive=True):
    with io.open(path, encoding="utf-8-sig") as handle:
        text = handle.read()

    original = text

    # The moved files declare their own namespace; the tests have one of their own and are left alone.
    if "namespace OsuTest.Game.Analysis" in text:
        text = text.replace("namespace OsuTest.Game.Analysis", "namespace OsuTest.BeatAnalysis")
        renamed += 1

    # That namespace does not exist any more, so a using of it will not compile.
    if OLD_USING in text:
        text = text.replace(OLD_USING + "\r\n", "").replace(OLD_USING + "\n", "")
        retargeted += 1

    if LIB_USING not in text and ("OsuTest.BeatAnalysis" in text or OLD_USING not in original):
        # A file that mentions the library at all should be able to see it.
        if text.rstrip().endswith("}") and "namespace " in text:
            lines = text.split("\n")
            last_using = -1

            for i, line in enumerate(lines):
                if line.startswith("using "):
                    last_using = i

            if last_using >= 0:
                lines[last_using + 1:last_using + 1] = [LIB_USING]
                text = "\n".join(lines)

    if text != original:
        with io.open(path, "w", encoding="utf-8", newline="\r\n") as handle:
            handle.write(text)

print(f"namespaces renamed: {renamed}, stale usings removed: {retargeted}")
