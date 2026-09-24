#nullable enable

using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using ManagedBass;
using NUnit.Framework;
using ParaTactus;
using OsuTest.Game.Audio;

namespace OsuTest.Game.Tests.Audio
{
    /// <summary>
    /// Tests the direct-BASS playback path used when the framework's track store cannot open a file.
    /// </summary>
    /// <remarks>
    /// This path only exists because the framework offers no public way to hand its track store a directory, so
    /// files outside the game's own resources cannot be played through it. Everything the visuals depend on -
    /// length, position, transport, rate - is therefore verified here, since nothing else in the framework covers
    /// it.
    /// </remarks>
    [TestFixture]
    public class BassAudioSourceTest
    {
        private string path = null!;

        [OneTimeSetUp]
        public void SetUp()
        {
            loadBassNative();

            if (!Bass.Init(Bass.NoSoundDevice, 44100, DeviceInitFlags.Default, IntPtr.Zero) && Bass.LastError != Errors.Already)
                Assert.Ignore($"BASS could not be initialised: {Bass.LastError}");

            path = DemoTrackGenerator.GetDefaultPath();

            if (!File.Exists(path))
                DemoTrackGenerator.WriteTo(path);
        }

        [Test]
        public void OpensAndReportsLength()
        {
            using (var source = BassAudioSource.TryOpen(path, out string? failure))
            {
                Assert.Multiple(() =>
                {
                    Assert.That(failure, Is.Null);
                    Assert.That(source, Is.Not.Null, "a plain audio file must open for direct playback");
                    Assert.That(source!.Length, Is.GreaterThan(20_000), "length must be reported in milliseconds");
                    Assert.That(source.IsRunning, Is.False, "a freshly opened source must not be playing");
                });
            }
        }

        [Test]
        public void TransportControlsPosition()
        {
            using (var source = BassAudioSource.TryOpen(path, out _))
            {
                Assert.That(source, Is.Not.Null);

                source!.Start();
                Assert.That(source.IsRunning, Is.True, "start must report running");

                // Seek is verified while playing: a push stream produces nothing while paused, so its position is
                // only meaningful once samples are flowing.
                source.Seek(5000);
                System.Threading.Thread.Sleep(120);

                TestContext.Out.WriteLine($"position after seek to 5000ms: {source.CurrentTime:0}ms");
                Assert.That(source.CurrentTime, Is.EqualTo(5000).Within(300), "seek must move the position");

                source.Stop();
                Assert.That(source.IsRunning, Is.False, "stop must report stopped");

                // A stopped source must not keep advancing, since its position is what drives the visuals.
                double stopped = source.CurrentTime;
                System.Threading.Thread.Sleep(150);
                Assert.That(source.CurrentTime, Is.EqualTo(stopped).Within(30), "a stopped source must not advance");
            }
        }

        /// <summary>
        /// The BASS path is created with the decode flag so it works without an output device, but playback is
        /// still expected to advance the position. This is what the music clock is read from.
        /// </summary>
        [Test]
        public void PlaybackAdvancesThePositionWithoutAnOutputDevice()
        {
            using (var source = BassAudioSource.TryOpen(path, out _))
            {
                Assert.That(source, Is.Not.Null);
                source!.Seek(0);
                source.Start();

                double start = source.CurrentTime;
                System.Threading.Thread.Sleep(300);
                double advanced = source.CurrentTime - start;

                TestContext.Out.WriteLine($"position advanced {advanced:0}ms in ~300ms of wall clock");

                Assert.That(advanced, Is.GreaterThan(150), "position must advance while playing, even with no output device");
            }
        }

        [Test]
        public void ReportsFailureForAMissingFile()
        {
            var source = BassAudioSource.TryOpen(Path.Combine(Path.GetTempPath(), "osutest-not-here.mp3"), out string? failure);

            Assert.Multiple(() =>
            {
                Assert.That(source, Is.Null);
                Assert.That(failure, Is.Not.Null.And.Not.Empty, "the reason must be reported rather than swallowed");
            });
        }

        /// <summary>
        /// The case that matters in practice: a user-picked MP3.
        /// </summary>
        /// <remarks>
        /// The framework's track store cannot open files outside its own resources, so this path - a real encoded
        /// file played through a BASS push stream - is what a user actually hits. Verified against an MP3 produced
        /// by an external encoder rather than the generated WAV, because container handling is exactly what the
        /// earlier decode bug was about.
        /// </remarks>
        [Test]
        public void PlaysAnEncodedMp3()
        {
            if (!File.Exists(@"C:\ffmpeg\bin\ffmpeg.exe"))
                Assert.Ignore("ffmpeg is not available to produce an MP3 fixture.");

            string mp3 = Path.Combine(Path.GetTempPath(), $"osutest-playback-{Guid.NewGuid():N}.mp3");

            try
            {
                if (!runFfmpeg($"-hide_banner -v error -y -i \"{path}\" -codec:a libmp3lame -b:a 128k -id3v2_version 3 \"{mp3}\""))
                    Assert.Ignore("ffmpeg could not encode the fixture.");

                using (var source = BassAudioSource.TryOpen(mp3, out string? failure))
                {
                    Assert.Multiple(() =>
                    {
                        Assert.That(failure, Is.Null, $"opening a real MP3 must succeed (reason: {failure})");
                        Assert.That(source, Is.Not.Null);
                        Assert.That(source!.Length, Is.GreaterThan(20_000), "length must be read from the MP3");
                    });

                    source!.Seek(3000);
                    source.Start();

                    double start = source.CurrentTime;
                    System.Threading.Thread.Sleep(250);
                    double advanced = source.CurrentTime - start;

                    TestContext.Out.WriteLine($"MP3 playback position: {start:0}ms -> {source.CurrentTime:0}ms (advanced {advanced:0}ms)");

                    Assert.That(advanced, Is.GreaterThan(120), "an MP3 must play and advance the clock");
                }
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

        /// <summary>
        /// The visualiser has to get a usable spectrum from this playback path, which was reported as not working.
        /// </summary>
        /// <remarks>
        /// A BASS decode channel produces no amplitude data, so the spectrum is computed from a second decode
        /// handle that follows the playback position. This measures what that actually returns - a non-empty array
        /// with a sensible spread of values - rather than assuming the code path produces something measurable.
        /// </remarks>
        [Test]
        public void ProvidesSpectrumForTheVisualiser()
        {
            using (var source = BassAudioSource.TryOpen(path, out _))
            {
                Assert.That(source, Is.Not.Null);

                source!.Seek(4000);
                source.Start();
                System.Threading.Thread.Sleep(120);

                // Called repeatedly, because that is how a visualiser uses it: the returned values are smoothed
                // across frames, so a single call after construction would read near zero even when the path
                // works. Verifying one call would have been the wrong test.
                float[] amplitudes = Array.Empty<float>();

                for (int frame = 0; frame < 6; frame++)
                {
                    amplitudes = source.GetFrequencyAmplitudes().ToArray();
                    System.Threading.Thread.Sleep(20);
                }

                double max = 0;
                double sum = 0;
                int nonZero = 0;

                foreach (float value in amplitudes)
                {
                    max = Math.Max(max, value);
                    sum += value;

                    if (value > 0.001f)
                        nonZero++;
                }

                TestContext.Out.WriteLine($"spectrum after 6 frames: {amplitudes.Length} bins, non-zero={nonZero}, max={max:0.####}, mean={sum / Math.Max(1, amplitudes.Length):0.####}");

                Assert.Multiple(() =>
                {
                    Assert.That(amplitudes.Length, Is.GreaterThan(0), "a spectrum must be produced at all");
                    Assert.That(nonZero, Is.GreaterThan(amplitudes.Length / 4), "most bins must carry signal, not silence");
                    Assert.That(max, Is.GreaterThan(0.001), "the spectrum must have a measurable peak, or the bars cannot move");
                });
            }
        }

        private static bool runFfmpeg(string arguments)
        {
            try
            {
                using (var process = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
                       {
                           FileName = @"C:\ffmpeg\bin\ffmpeg.exe",
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
        /// Analyses an audio file from an environment variable and reports why its tempo might be unstable.
        /// </summary>
        /// <remarks>
        /// Exists so a reported problem with a specific track can be investigated against the real audio instead of
        /// against a synthetic approximation. Guessing at parameters from a description of the symptom failed
        /// repeatedly; this produces the per-window estimates, the confidence behind them, and the keyframes that
        /// survived, all of which distinguish "the estimate is jumping", "the confidence has collapsed" and "the
        /// stabilisation is wrong" - three different causes with three different fixes.
        ///
        /// Set <c>OSUTEST_AUDIO</c> to the file path. Optionally set <c>OSUTEST_RANGE</c> to "from,to" in seconds to
        /// print the detailed rows only for that stretch.
        /// </remarks>
        /// <summary>
        /// Makes the BASS native library loadable without a game host to initialise audio.
        /// </summary>
        private static void loadBassNative()
        {
            string packages = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");
            string natives = Path.Combine(packages, "ppy.osu.framework.nativelibs");

            if (!Directory.Exists(natives))
                return;

            string? source = null;

            foreach (string version in Directory.GetDirectories(natives))
            {
                string candidate = Path.Combine(version, "runtimes", "win-x64", "native");

                if (Directory.Exists(candidate))
                {
                    source = candidate;
                    break;
                }
            }

            if (source == null)
                return;

            foreach (string file in new[] { "bass.dll", "bass_fx.dll" })
            {
                string from = Path.Combine(source, file);
                string to = Path.Combine(AppContext.BaseDirectory, file);

                if (File.Exists(from) && !File.Exists(to))
                {
                    try
                    {
                        File.Copy(from, to);
                    }
                    catch (IOException)
                    {
                    }
                }
            }

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
