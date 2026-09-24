"""Removes the half-added seek slider so the file builds again.

Run with the repository root as the working directory:
    python tools/remove_seek_bar.py
"""

import io

PATH = r"OsuTest\OsuTest.Game\Graphics\BeatSyncDemo.cs"

with io.open(PATH, encoding="utf-8") as handle:
    text = handle.read()

before = len(text)

pieces = [
    (
        "field",
        "        /// <summary>Scrubs the track, so a section can be reached without waiting for it to play.</summary>\n"
        "        private SliderBar<double> seekBar = null!;\n"
        "\n"
        "        private bool seekBarReady;\n",
    ),
    (
        "slider",
        "\n"
        "                        // A scrubber, because the failures being chased are in specific sections of specific tracks and\n"
        "                        // waiting for a three hundred second track to reach 284 seconds is not a way to work.\n"
        "                        seekBar = new SliderBar<double>\n"
        "                        {\n"
        "                            Width = 300,\n"
        "                            MinValue = 0,\n"
        "                            MaxValue = 1000,\n"
        "                            Current = { Value = 0 },\n"
        "                            TooltipText = \"\u62d6\u52a8\u5b9a\u4f4d\",\n"
        "                        },\n",
    ),
]

for label, piece in pieces:
    if piece not in text:
        raise SystemExit(f"{label} block not found verbatim; refusing to guess")

    text = text.replace(piece, "", 1)

for leftover in ("seekBar", "SliderBar"):
    if leftover in text:
        raise SystemExit(f"{leftover} still present after removal; refusing to leave a half state")

with io.open(PATH, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"removed the seek slider, {before - len(text)} characters")
