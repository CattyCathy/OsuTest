using System;
using System.IO;
using ParaTactus;

namespace OsuTest.Game.Audio
{
    /// <summary>
    /// Minimal 16-bit PCM WAV writer, so audio fixtures can be produced without shipping binary assets.
    /// </summary>
    public static class WavWriter
    {
        /// <summary>
        /// Writes mono float samples (range -1..1) as a canonical 44-byte-header PCM WAV. This is the
        /// least surprising thing to hand to a decoder, which matters because the whole point of writing a
        /// file at all is to exercise the real decode path rather than a synthetic in-memory track.
        /// </summary>
        public static void Write(string path, float[] samples, int sampleRate)
        {
            using (var stream = File.Create(path))
            using (var writer = new BinaryWriter(stream))
            {
                const short channels = 1;
                const short bitsPerSample = 16;
                int dataLength = samples.Length * sizeof(short);

                writer.Write(new[] { 'R', 'I', 'F', 'F' });
                writer.Write(36 + dataLength);
                writer.Write(new[] { 'W', 'A', 'V', 'E' });

                writer.Write(new[] { 'f', 'm', 't', ' ' });
                writer.Write(16);                       // PCM chunk size
                writer.Write((short)1);                 // PCM format
                writer.Write(channels);
                writer.Write(sampleRate);
                writer.Write(sampleRate * channels * bitsPerSample / 8);   // byte rate
                writer.Write((short)(channels * bitsPerSample / 8));       // block align
                writer.Write(bitsPerSample);

                writer.Write(new[] { 'd', 'a', 't', 'a' });
                writer.Write(dataLength);

                foreach (float sample in samples)
                {
                    // Hard clip rather than normalise: the synth is designed to peak below 1, so clipping
                    // only ever catches a mistake, and normalising would hide it.
                    double clamped = Math.Clamp(sample, -1, 1);
                    writer.Write((short)(clamped * short.MaxValue));
                }
            }
        }
    }
}
