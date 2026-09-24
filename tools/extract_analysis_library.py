"""Splits the beat analysis out of the application into its own project.

Moves the namespace, replaces the one place the library reached for the application's decoder, and points the callers
that have a file path at a thin application-side helper instead.

Run from the repository root:
    python tools/extract_analysis_library.py
"""

import io
import os
import re
import glob

LIB = r"OsuTest\OsuTest.BeatAnalysis"
APP = r"OsuTest\OsuTest.Game"

MOVED = [
    "Fft.cs", "BeatGrid.cs", "MetricalLevel.cs", "BeatTimeMap.cs", "BeatThisBeatTracker.cs",
    "StreamingBeatTracker.cs", "BeatTrainRegulariser.cs", "OnsetEnvelope.cs", "BeatGridCache.cs",
]


def read(path):
    with io.open(path, encoding="utf-8-sig") as handle:
        return handle.read()


def write(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\r\n") as handle:
        handle.write(text)


def replace_in(path, old, new, required=True):
    text = read(path)

    if old not in text:
        if required:
            raise SystemExit(f"{path}: expected text not found:\n{old[:200]}")
        return False

    write(path, text.replace(old, new))
    return True


# 1. The moved files get the library namespace.
for name in MOVED:
    replace_in(os.path.join(LIB, name),
               "namespace OsuTest.Game.Analysis",
               "namespace OsuTest.BeatAnalysis")

# 2. Every reference to that namespace anywhere follows it.
for path in glob.glob(r"OsuTest\**\*.cs", recursive=True):
    text = read(path)

    if "using OsuTest.Game.Analysis;" in text:
        write(path, text.replace("using OsuTest.Game.Analysis;", "using OsuTest.BeatAnalysis;"))

# 3. The library stops knowing about the decoder. It takes samples; the sample rate is its own constant now, because
#    the frontend's frame grid is defined in terms of it.
with io.open(os.path.join(LIB, "AnalysisAudio.cs"), "w", encoding="utf-8", newline="\r\n") as handle:
    handle.write("""namespace OsuTest.BeatAnalysis
{
    /// <summary>
    /// The sample rate everything here works at.
    /// </summary>
    /// <remarks>
    /// Fixed rather than configurable, because the model's frame grid is defined in terms of it: the frontend hops 441
    /// samples at 22050Hz, which is exactly 50 frames per second, and that is the rate the model's output is in. Any
    /// other rate would need the whole frontend re-derived.
    /// </remarks>
    public static class AnalysisAudio
    {
        /// <summary>The rate the analysis runs at, in hertz.</summary>
        public const int SampleRate = 22050;
    }
}
""")

replace_in(os.path.join(LIB, "OnsetEnvelope.cs"),
           "int hop = AudioDecoder.ANALYSIS_SAMPLE_RATE / BeatThisBeatTracker.FramesPerSecond;",
           "int hop = AnalysisAudio.SampleRate / BeatThisBeatTracker.FramesPerSecond;")

replace_in(os.path.join(LIB, "BeatThisBeatTracker.cs"),
           """        /// <summary>Model beat times in milliseconds, for a file.</summary>
        public static double[] BeatTimes(string audioPath, string modelPath)
        {
            var samples = AudioDecoder.DecodeMono(audioPath, sample_rate);
            var spectrogram = LogMelSpectrogram(samples);""",
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
            var spectrogram = LogMelSpectrogram(samples);""")

# 4. The application keeps a path-taking entry point next to its decoder.
with io.open(os.path.join(APP, "Analysis", "FileBeats.cs"), "w", encoding="utf-8", newline="\r\n") as handle:
    handle.write("""using OsuTest.BeatAnalysis;

namespace OsuTest.Game.Analysis
{
    /// <summary>
    /// The beats of a file, for callers that have a path rather than samples.
    /// </summary>
    /// <remarks>
    /// The one thing the analysis library cannot do for itself is turn a file into PCM, so that half lives here beside
    /// the decoder and the analysis is called with the result. Callers that already have samples should call the
    /// library directly.
    /// </remarks>
    public static class FileBeats
    {
        /// <summary>Decodes a file and returns the model's beat times in milliseconds.</summary>
        public static double[] BeatTimes(string audioPath, string modelPath)
        {
            return BeatThisBeatTracker.BeatTimes(
                AudioDecoder.DecodeMono(audioPath, AnalysisAudio.SampleRate), modelPath);
        }
    }
}
""")

# 5. The decoder's constant now points at the library's, so the application has one definition to follow.
audio_decoder = os.path.join(APP, "Analysis", "AudioDecoder.cs")
text = read(audio_decoder)

match = re.search(r"public const int ANALYSIS_SAMPLE_RATE = \d+;", text)

if not match:
    raise SystemExit("AudioDecoder: the sample rate constant was not found")

text = text.replace(match.group(0),
                    "public const int ANALYSIS_SAMPLE_RATE = AnalysisAudio.SampleRate;")

if "using OsuTest.BeatAnalysis;" not in text:
    text = "using OsuTest.BeatAnalysis;\n" + text

write(audio_decoder, text)

# 6. Callers that passed a path now go through the application helper.
for path in glob.glob(r"OsuTest\OsuTest.Game.Tests\**\*.cs", recursive=True) + \
            glob.glob(r"OsuTest\OsuTest.Game\**\*.cs", recursive=True):
    text = read(path)

    if "BeatThisBeatTracker.BeatTimes(audio, model)" in text:
        write(path, text.replace("BeatThisBeatTracker.BeatTimes(audio, model)",
                                 "FileBeats.BeatTimes(audio, model)"))

print("extraction applied")
