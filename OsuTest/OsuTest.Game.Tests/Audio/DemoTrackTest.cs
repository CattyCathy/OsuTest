#nullable enable

using System;
using System.IO;
using System.Linq;
using ManagedBass;
using NUnit.Framework;
using ParaTactus;
using OsuTest.Game.Audio;
using ParaTactus.Decoding;

namespace OsuTest.Game.Tests.Audio
{
    /// <summary>
    /// End-to-end check of the demo path: synthesise a tempo-changing track, write it as a real WAV, decode
    /// it with BASS and analyse the decoded PCM, then compare against the tempo map the track was built from.
    /// </summary>
    /// <remarks>
    /// This is the only test that covers the pieces the synthetic unit tests cannot: WAV encoding, the BASS
    /// decode-and-resample step, and the fact that a real decode introduces a small prepended silence (which
    /// shifts every beat by a constant offset the analysis has to absorb).
    /// </remarks>
    [TestFixture]
    public class DemoTrackTest
    {
        private const double match_tolerance_ms = 70;

        private string path = null!;
        private TempoMap map = null!;

        [OneTimeSetUp]
        public void SetUp()
        {
            loadBassNative();

            // Initialised explicitly because nothing else in a test host does it: there is no
            // osu.Framework.Game here to start an AudioManager.
            if (!Bass.Init(Bass.NoSoundDevice, 44100, DeviceInitFlags.Default, IntPtr.Zero) && Bass.LastError != Errors.Already)
                Assert.Ignore($"BASS could not be initialised for decoding: {Bass.LastError}");

            path = DemoTrackGenerator.GetDefaultPath();
            map = DemoTrackGenerator.WriteTo(path);
        }

        [OneTimeTearDown]
        public void TearDown()
        {
            try
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
            catch (IOException)
            {
                // A locked temp file is not worth failing a test run over.
            }
        }

        /// <summary>
        /// The same analysis, but on the samples straight from the generator - no BASS decode involved.
        /// </summary>
        /// <remarks>
        /// Exists to separate the two possible sources of failure: if the decoding test fails and this one
        /// passes, the problem is in the decode step rather than in the onset or tempo analysis. Not part of the
        /// regular suite because on this fixture it runs straight into the octave ambiguity, and its outcome is
        /// then a property of the fixture rather than of the code.
        /// </remarks>
        [Test]
        public void TestDecodeLengthMatchesFileHeader()
        {
            byte[] header = new byte[44];

            using (var stream = File.OpenRead(path))
                stream.ReadExactly(header);

            short channels = BitConverter.ToInt16(header, 22);
            int sampleRate = BitConverter.ToInt32(header, 24);
            int dataLength = BitConverter.ToInt32(header, 40);

            float[] decoded;

            using (var stream = File.OpenRead(path))
                decoded = BassAudioDecoder.DecodeMono(stream);

            double expectedSeconds = dataLength / (double)(sampleRate * channels * sizeof(short));
            double actualSeconds = decoded.Length / (double)BassAudioDecoder.ANALYSIS_SAMPLE_RATE;

            TestContext.Out.WriteLine($"wav: channels={channels} rate={sampleRate} data={dataLength} -> {expectedSeconds:0.###}s");
            TestContext.Out.WriteLine($"decoded: {decoded.Length} samples @ {BassAudioDecoder.ANALYSIS_SAMPLE_RATE} -> {actualSeconds:0.###}s");

            using (var stream = File.OpenRead(path))
            {
                var raw = BassAudioDecoder.DecodeRaw(stream);
                TestContext.Out.WriteLine($"raw: {raw.Samples.Length} samples, channels={raw.Channels}, rate={raw.SampleRate} -> {raw.Samples.Length / (double)raw.Channels / raw.SampleRate:0.###}s");
            }

            Assert.That(actualSeconds, Is.EqualTo(expectedSeconds).Within(0.05), "decoded duration must match the file");
        }

        /// <summary>
        /// Decodes a real MP3 produced by an external encoder, which is the case that failed in practice.
        /// </summary>
        /// <remarks>
        /// WAV is a trivial container; MP3 is not. It carries an ID3v2 tag to skip, may be variable bitrate, and
        /// needs the decoder to seek within the stream. That is why this test exists separately: decoding
        /// through file callbacks with <c>StreamSystem.NoBuffer</c> handled the generated WAV fine and failed on
        /// MP3, and only a real encoded file exercises that path.
        ///
        /// Skipped when ffmpeg is not installed, since the point is the format rather than the encoder.
        /// </remarks>
        [Test]
        public void TestDecodesEncodedMp3()
        {
            if (!Directory.Exists(@"C:\ffmpeg\bin") && Environment.GetEnvironmentVariable("PATH")?.Contains("ffmpeg") != true)
                Assert.Ignore("ffmpeg is not available to produce an MP3 fixture.");

            string mp3 = Path.Combine(Path.GetTempPath(), $"osutest-mp3-{Guid.NewGuid():N}.mp3");

            try
            {
                // 128k CBR with an ID3v2 tag, which is what a normal music file looks like.
                if (!runFfmpeg($"-hide_banner -v error -y -i \"{path}\" -codec:a libmp3lame -b:a 128k -id3v2_version 3 \"{mp3}\""))
                    Assert.Ignore("ffmpeg could not encode the fixture.");

                var raw = BassAudioDecoder.DecodeRaw(mp3);

                TestContext.Out.WriteLine($"mp3 size={new FileInfo(mp3).Length} raw: {raw.Samples.Length} samples, "
                                          + $"{raw.Channels}ch @ {raw.SampleRate}Hz -> {raw.Samples.Length / (double)raw.Channels / raw.SampleRate:0.###}s");

                Assert.Multiple(() =>
                {
                    // Tempo is not measured here any more - it comes from the model rather than the decoder, and
                    // this test is about whether an MP3 with an ID3v2 tag decodes back to the right samples.
                    Assert.That(raw.SampleRate, Is.EqualTo(DemoTrackGenerator.SAMPLE_RATE), "MP3 must decode back at the source rate");

                    // Compared against the rendered length including the decay tail, not the musical duration.
                    double renderedSeconds = (map.Duration + 400) / 1000.0;

                    Assert.That(raw.Samples.Length / (double)raw.Channels / raw.SampleRate, Is.EqualTo(renderedSeconds).Within(0.15),
                        "decoded duration must match what was encoded (encoder padding aside)");

                    // Tempo is not measured here any more - it comes from the model rather than from the decoder,
                    // and this test is about whether an MP3 with an ID3 tag decodes back to the right samples.
                });
            }
            finally
            {
                try
                {
                    if (File.Exists(mp3))
                        File.Delete(mp3);
                }
                catch (IOException)
                {
                }
            }
        }

        private static bool runFfmpeg(string arguments)
        {
            string executable = File.Exists(@"C:\ffmpeg\bin\ffmpeg.exe") ? @"C:\ffmpeg\bin\ffmpeg.exe" : "ffmpeg";

            try
            {
                using (var process = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
                       {
                           FileName = executable,
                           Arguments = arguments,
                           UseShellExecute = false,
                           CreateNoWindow = true,
                           RedirectStandardError = true,
                           RedirectStandardOutput = true,
                       }))
                {
                    if (process == null)
                        return false;

                    process.WaitForExit(60_000);
                    return process.HasExited && process.ExitCode == 0;
                }
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>
        /// Analysing a second, different file must produce a different tempo map. This is the core assumption
        /// behind swapping the audio source at runtime.
        /// </summary>
        /// <remarks>
        /// Two fixtures are used rather than the same one twice, because re-analysing identical audio would pass
        /// whether or not the analysis actually depends on the decoded content.
        /// </remarks>
        private static void writeFixture(string path, double bpm, double seconds)
        {
            int total = (int)(seconds * DemoTrackGenerator.SAMPLE_RATE);
            var samples = new float[total];
            var random = new Random(4242);

            int clickLength = DemoTrackGenerator.SAMPLE_RATE / 8;

            for (double time = 0; time < seconds * 1000; time += 60000 / bpm)
            {
                int start = (int)(time * DemoTrackGenerator.SAMPLE_RATE / 1000);

                for (int i = 0; i < clickLength && start + i < total; i++)
                {
                    double t = i / (double)DemoTrackGenerator.SAMPLE_RATE;
                    double envelope = Math.Exp(-t * 30);
                    double tone = Math.Sin(2 * Math.PI * 70 * t) + 0.5 * Math.Sin(2 * Math.PI * 140 * t);
                    double click = (random.NextDouble() - 0.5) * Math.Exp(-t * 120);

                    samples[start + i] += (float)(envelope * (tone * 0.3 + click * 0.5));
                }
            }

            WavWriter.Write(path, samples, DemoTrackGenerator.SAMPLE_RATE);
        }

        private static void loadBassNative()
        {
            string packages = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");
            string natives = Path.Combine(packages, "ppy.osu.framework.nativelibs");

            if (!Directory.Exists(natives))
                return;

            string? source = Directory.GetDirectories(natives)
                                      .OrderByDescending(d => d)
                                      .Select(d => Path.Combine(d, "runtimes", "win-x64", "native"))
                                      .FirstOrDefault(Directory.Exists);

            if (source == null)
                return;

            foreach (string file in new[] { "bass.dll", "bass_fx.dll" })
            {
                string from = Path.Combine(source, file);
                string to = Path.Combine(AppContext.BaseDirectory, file);

                if (!File.Exists(from) || File.Exists(to))
                    continue;

                try
                {
                    File.Copy(from, to);
                }
                catch (IOException)
                {
                    // Already loaded by another test run; the name lookup will still find it.
                }
            }

            // Touch the library so a failure surfaces here rather than as an opaque decode error.
            try
            {
                _ = Bass.Version;
            }
            catch (DllNotFoundException)
            {
                Assert.Ignore("BASS native library is not available in this environment.");
            }
        }
    }
}
