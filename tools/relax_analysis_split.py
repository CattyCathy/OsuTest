"""Relaxes the split to what was actually wanted: independent of this project, not of the framework.

Decoding only ever depended on osu.Framework and BASS, never on the application, so it belongs with the analysis. Moving
it in removes the seam that existed only to keep those two out, and lets the provider move in unchanged.

Run from the repository root:
    python tools/relax_analysis_split.py
"""

import io
import glob
import os

LIB = r"OsuTest\OsuTest.BeatAnalysis"


def read(path):
    with io.open(path, encoding="utf-8-sig") as handle:
        return handle.read()


def write(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\r\n") as handle:
        handle.write(text)


def replace_in(path, old, new):
    text = read(path)

    if old not in text:
        raise SystemExit(f"{path}: expected text not found:\n{old[:160]}")

    write(path, text.replace(old, new))


# 1. The sample-based entry point goes back to taking a path, now that the decoder lives here too.
replace_in(os.path.join(LIB, "BeatThisBeatTracker.cs"),
           """        /// <summary>
        /// Model beat times in milliseconds, for mono audio at <see cref="AnalysisAudio.SampleRate"/>.
        /// </summary>
        /// <remarks>
        /// Samples rather than a path, so that nothing here has to know how audio is decoded. Decoding is the one part
        /// of the pipeline that cannot be shared: it needs a platform audio library, and dragging one in would make
        /// this project unusable anywhere that does not want it.
        /// </remarks>
        public static double[] BeatTimes(float[] samples, string modelPath)
        {
            var spectrogram = LogMelSpectrogram(samples);""",
           """        /// <summary>Model beat times in milliseconds, for mono audio at <see cref="AnalysisAudio.SampleRate"/>.</summary>
        /// <remarks>
        /// Also accepts samples directly, which is what callers that have already decoded a track - or that are feeding
        /// audio as it plays - should use.
        /// </remarks>
        public static double[] BeatTimes(float[] samples, string modelPath)
        {
            var spectrogram = LogMelSpectrogram(samples);""")

# An overload that takes a path, beside the one that takes samples. Decoding is in this project now, so neither the
# frontend nor any caller has to care which form it wants.
text = read(os.path.join(LIB, "BeatThisBeatTracker.cs"))

anchor = """        /// <summary>Model beat times in milliseconds, for mono audio at <see cref="AnalysisAudio.SampleRate"/>.</summary>"""

addition = """        /// <summary>Model beat times in milliseconds, taken from a file.</summary>
        public static double[] BeatTimes(string audioPath, string modelPath)
        {
            return BeatTimes(AudioDecoder.DecodeMono(audioPath, AnalysisAudio.SampleRate), modelPath);
        }

"""

if "BeatTimes(string audioPath" not in text:
    write(os.path.join(LIB, "BeatThisBeatTracker.cs"), text.replace(anchor, addition + anchor))

# 2. The application-side shim has nothing left to do.
shim = os.path.join(r"OsuTest\OsuTest.Game\Analysis", "FileBeats.cs")

if os.path.exists(shim):
    os.remove(shim)

# 3. Callers go back to the single entry point.
for path in glob.glob(r"OsuTest\**\*.cs", recursive=True):
    text = read(path)

    if "FileBeats.BeatTimes(" in text:
        write(path, text.replace("FileBeats.BeatTimes(", "BeatThisBeatTracker.BeatTimes("))

print("split relaxed")
