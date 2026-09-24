using System;
using System.IO;
using ManagedBass;
using NUnit.Framework;
using ParaTactus;
using OsuTest.Game.Audio;
using ParaTactus.Decoding;

namespace OsuTest.Game.Tests.Audio
{
    /// <summary>
    /// Reports what the audio environment actually looks like, and whether the device-selection mechanism the
    /// demo exposes can restore playback.
    /// </summary>
    /// <remarks>
    /// Written after a real failure: files decoded and analysed fine but no playback stream could be opened,
    /// because an empty <c>AudioDevice</c> in framework.ini left BASS initialised against the "No sound" device.
    /// The distinction that matters - decoding needs no output device, playback does - is asserted here rather
    /// than assumed, and the device switch is verified at the BASS level because that is exactly what setting
    /// <c>AudioManager.AudioDevice</c> ends up doing.
    ///
    /// Ignored by default: it initialises BASS globally, which affects other audio tests sharing the process.
    /// </remarks>
    [TestFixture]
    [Explicit("diagnostic; run manually when investigating audio output problems")]
    public class AudioEnvironmentDiagnosticTest
    {
        [Test]
        public void ReportAudioEnvironment()
        {
            string path = DemoTrackGenerator.GetDefaultPath();

            if (!File.Exists(path))
                DemoTrackGenerator.WriteTo(path);

            // ---- 1. decoding, with no output device at all ----
            bool noSound = Bass.Init(Bass.NoSoundDevice, 44100, DeviceInitFlags.Default, IntPtr.Zero) || Bass.LastError == Errors.Already;
            logDevices($"no sound device (init={noSound})");

            var raw = BassAudioDecoder.DecodeRaw(path);
            TestContext.Out.WriteLine($"decode with no output device => ok, {raw.Samples.Length} samples, {raw.Channels}ch @ {raw.SampleRate}Hz");

            int decodePlayback = Bass.CreateStream(path, 0, 0, BassFlags.Float);
            TestContext.Out.WriteLine($"playback stream with no output device => handle={decodePlayback} (unsigned={(uint)decodePlayback}), LastError={Bass.LastError}");
            describeHandle("no-sound-device playback", decodePlayback, path);

            if (decodePlayback != 0)
                Bass.StreamFree(decodePlayback);

            Bass.Free();

            // ---- 2. the same thing against a real output device ----
            // This is what selecting a device in the demo produces, so if playback works here the fix works.
            var info = findDefaultDevice();

            if (info.Index < 0)
            {
                Assert.Ignore("No usable output device on this machine, so the device-switch path cannot be verified.");
                return;
            }

            bool started = Bass.Init(info.Index, 44100, DeviceInitFlags.Default, IntPtr.Zero) || Bass.LastError == Errors.Already;
            logDevices($"real device '{info.Info.Name}' (index {info.Index}, init={started})");

            int realPlayback = Bass.CreateStream(path, 0, 0, BassFlags.Float);
            TestContext.Out.WriteLine($"playback stream with real device => handle={realPlayback} (unsigned={(uint)realPlayback}), LastError={Bass.LastError}");
            int realLength = describeHandle("real-device playback", realPlayback, path);

            Assert.Multiple(() =>
            {
                Assert.That(started, Is.True, "BASS must initialise against a real output device");
                Assert.That(realPlayback, Is.Not.EqualTo(0), "a playback stream must be creatable once a real device is selected");
                Assert.That(realLength, Is.GreaterThan(0),
                    "the handle must be usable - a handle whose length cannot be queried is the real failure, "
                    + "and a negative-looking int is not by itself proof of one because BASS handles are large enough "
                    + "to read as negative when interpreted as signed");
            });

            if (realPlayback != 0)
                Bass.StreamFree(realPlayback);

            Bass.Free();
        }

        /// <summary>
        /// Reports whether a failed-looking handle is actually usable, and returns its length in bytes.
        /// </summary>
        /// <remarks>
        /// BASS handles are 64-bit values; the lower 32 bits can exceed <see cref="int.MaxValue"/> and appear
        /// negative. Treating that as an error would be wrong, so usability is checked by querying the channel
        /// rather than by comparing the handle to zero.
        /// </remarks>
        private static int describeHandle(string context, int handle, string path)
        {
            if (handle == 0)
            {
                TestContext.Out.WriteLine($"  {context}: handle 0 (creation failed), LastError={Bass.LastError}");
                return 0;
            }

            try
            {
                var info = Bass.ChannelGetInfo(handle);
                long lengthBytes = Bass.ChannelGetLength(handle);
                double seconds = Bass.ChannelBytes2Seconds(handle, lengthBytes);

                TestContext.Out.WriteLine($"  {context}: usable - {info.Channels}ch @ {info.Frequency}Hz, {seconds:0.###}s, playbackState={Bass.ChannelIsActive(handle)}");
                return (int)lengthBytes;
            }
            catch (Exception e)
            {
                TestContext.Out.WriteLine($"  {context}: handle {handle} is NOT usable - {e.GetType().Name}: {e.Message}");
                return 0;
            }
        }

        private static (int Index, DeviceInfo Info) findDefaultDevice()
        {
            for (int i = 0; i < Bass.DeviceCount; i++)
            {
                var candidate = Bass.GetDeviceInfo(i);

                if (candidate.IsEnabled && candidate.IsDefault && !candidate.Name.Contains("No sound", StringComparison.OrdinalIgnoreCase))
                    return (i, candidate);
            }

            for (int i = 0; i < Bass.DeviceCount; i++)
            {
                var candidate = Bass.GetDeviceInfo(i);

                if (candidate.IsEnabled && !candidate.Name.Contains("No sound", StringComparison.OrdinalIgnoreCase))
                    return (i, candidate);
            }

            return (-1, default);
        }

        private static void logDevices(string context)
        {
            TestContext.Out.WriteLine($"--- devices ({context}), BASS {Bass.Version} ---");

            for (int i = 0; i < Bass.DeviceCount; i++)
            {
                var device = Bass.GetDeviceInfo(i);
                TestContext.Out.WriteLine($"  [{i}] '{device.Name}' enabled={device.IsEnabled} default={device.IsDefault} initialised={device.IsInitialized}");
            }

            TestContext.Out.WriteLine($"  current device index = {Bass.CurrentDevice}");
        }
    }
}
