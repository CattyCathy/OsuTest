using System;
using System.IO;
using ParaTactus;

namespace OsuTest.Game.Audio
{
    /// <summary>
    /// Synthesises a short piece of music whose tempo changes partway through, and writes it to disk.
    /// </summary>
    /// <remarks>
    /// The project ships no audio, and a tempo-changing track is exactly the case worth demonstrating, so
    /// the fixture is generated rather than imported. Doing it through a real WAV file (instead of a
    /// virtual in-memory track) means the demo exercises the same path a real beatmap would: decode with
    /// BASS, analyse the decoded PCM, then drive visuals from the resulting tempo map.
    ///
    /// The arrangement is deliberately conventional - kick on 1 and 3, snare on 2 and 4, hats on the
    /// off-beats, and a sustained bass note per bar - because the point is to be a fair test of the
    /// analyser: a drum machine pattern with no harmonic content would be far easier than real music.
    /// </remarks>
    public static class DemoTrackGenerator
    {
        public const int SAMPLE_RATE = 44100;

        /// <summary>
        /// The tempo map of the generated track: a 120 BPM intro followed by a 160 BPM section.
        /// </summary>
        public static TempoMap CreateTempoMap()
            => new TempoMap()
               .AddSection(120, 8)
               .AddSection(160, 8);

        /// <summary>
        /// Writes the demo track to <paramref name="path"/> and returns its tempo map.
        /// </summary>
        public static TempoMap WriteTo(string path)
        {
            var map = CreateTempoMap();
            float[] samples = Render(map);
            WavWriter.Write(path, samples, SAMPLE_RATE);

            return map;
        }

        /// <summary>
        /// Renders the track described by <paramref name="map"/> to mono float samples.
        /// </summary>
        public static float[] Render(TempoMap map)
        {
            // A short tail so the final note decays instead of being cut off mid-cycle.
            int totalSamples = (int)((map.Duration + 400) * SAMPLE_RATE / 1000);
            var samples = new float[totalSamples];
            var random = new Random(20260923);

            double[] beats = map.BuildBeatTimes();

            for (int i = 0; i < beats.Length; i++)
            {
                int beatInBar = i % 4;
                double beatLength = map.NearestBeatLength(beats[i]);
                double intensity = i % 8 < 4 ? 0.85 : 1.0;

                if (beatInBar == 0 || beatInBar == 2)
                    renderKick(samples, beats[i], 0.9 * intensity);
                else
                    renderSnare(samples, beats[i], 0.55 * intensity, random);

                if (beatInBar == 0)
                    renderBass(samples, beats[i], beatLength * 4, 0.32);

                // Off-beat hats, quieter than the pulse - the same trap the analyser's suppression window
                // and tempo prior exist to deal with.
                renderHat(samples, beats[i] + beatLength / 2, 0.1, random);
            }

            addNoiseFloor(samples, random);
            return samples;
        }

        /// <summary>Sustained low note with a couple of harmonics, giving the analyser tonal content to ignore.</summary>
        private static void renderBass(float[] samples, double timeMs, double durationMs, double amplitude)
        {
            int start = (int)(timeMs * SAMPLE_RATE / 1000);
            int length = (int)(durationMs * SAMPLE_RATE / 1000);
            const double frequency = 55; // A1

            for (int i = 0; i < length; i++)
            {
                int index = start + i;

                if (index < 0 || index >= samples.Length)
                    continue;

                double t = i / (double)SAMPLE_RATE;
                double attack = Math.Min(1, t * 40);
                double release = Math.Min(1, (length - i) / (0.15 * SAMPLE_RATE));
                double envelope = attack * release;

                double tone = Math.Sin(2 * Math.PI * frequency * t)
                              + 0.3 * Math.Sin(2 * Math.PI * frequency * 2 * t);

                samples[index] += (float)(amplitude * envelope * tone * 0.5);
            }
        }

        private static void renderKick(float[] samples, double timeMs, double amplitude)
        {
            int start = (int)(timeMs * SAMPLE_RATE / 1000);
            int length = (int)(0.16 * SAMPLE_RATE);

            for (int i = 0; i < length; i++)
            {
                int index = start + i;

                if (index < 0 || index >= samples.Length)
                    continue;

                double t = i / (double)SAMPLE_RATE;
                double envelope = Math.Exp(-t * 28);

                // Slight downward pitch sweep, which is what makes a kick read as a kick rather than as a
                // steady tone in the low band.
                double frequency = 80 * Math.Exp(-t * 9) + 45;
                double tone = Math.Sin(2 * Math.PI * frequency * t);

                samples[index] += (float)(amplitude * envelope * tone * 0.7);
            }
        }

        private static void renderSnare(float[] samples, double timeMs, double amplitude, Random random)
        {
            int start = (int)(timeMs * SAMPLE_RATE / 1000);
            int length = (int)(0.14 * SAMPLE_RATE);

            for (int i = 0; i < length; i++)
            {
                int index = start + i;

                if (index < 0 || index >= samples.Length)
                    continue;

                double t = i / (double)SAMPLE_RATE;
                double envelope = Math.Exp(-t * 26);
                double noise = random.NextDouble() * 2 - 1;
                double body = Math.Sin(2 * Math.PI * 190 * t);

                samples[index] += (float)(amplitude * envelope * (noise * 0.75 + body * 0.35));
            }
        }

        private static void renderHat(float[] samples, double timeMs, double amplitude, Random random)
        {
            int start = (int)(timeMs * SAMPLE_RATE / 1000);
            int length = (int)(0.04 * SAMPLE_RATE);

            for (int i = 0; i < length; i++)
            {
                int index = start + i;

                if (index < 0 || index >= samples.Length)
                    continue;

                double envelope = Math.Exp(-i / (double)SAMPLE_RATE * 160);
                double noise = random.NextDouble() * 2 - 1;

                samples[index] += (float)(amplitude * envelope * noise);
            }
        }

        /// <summary>
        /// A faint constant hiss. Real recordings are never digitally silent, and an analyser that only
        /// works on a noise-free signal is not worth having.
        /// </summary>
        private static void addNoiseFloor(float[] samples, Random random)
        {
            for (int i = 0; i < samples.Length; i++)
                samples[i] += (float)((random.NextDouble() - 0.5) * 0.004);
        }

        /// <summary>
        /// A stable path for the generated track. Uses the temp directory so nothing generated ends up
        /// committed to the repository.
        /// </summary>
        public static string GetDefaultPath()
            => Path.Combine(Path.GetTempPath(), "osutest-demo-track.wav");
    }
}
