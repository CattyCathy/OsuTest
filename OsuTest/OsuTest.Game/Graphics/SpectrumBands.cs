using System;
using ParaTactus;

namespace OsuTest.Game.Graphics
{
    /// <summary>
    /// Which bins of a spectrum each bar of the display reads.
    /// </summary>
    /// <remarks>
    /// Lifted out of the visualiser so it can be checked without a screen. The mapping is the part that decides what the
    /// display says, and a display that is wrong is indistinguishable from audio that is silent: a band whose bins are
    /// all zero looks exactly like a band the code never reads. Keeping the arithmetic here makes it possible to prove
    /// the first rather than assume it.
    ///
    /// The bands are spaced logarithmically in hertz, because equal-width slices of a linear spectrum give every band
    /// the same bandwidth and music puts almost all of its energy in the bottom few hundred hertz - which pins the
    /// lowest bars to the ceiling and leaves everything from the middle up on the floor.
    /// </remarks>
    public static class SpectrumBands
    {
        /// <summary>The lowest frequency shown.</summary>
        /// <remarks>
        /// Not the bottom of the spectrum. Below this is rumble and the tail of any DC offset: loud, unchanging, and not
        /// music.
        /// </remarks>
        public const double BottomHz = 40;

        /// <summary>The highest frequency shown, before the audio's own bandwidth pulls it down.</summary>
        public const double TopHz = 16000;

        /// <summary>
        /// The bin range each band covers, lowest band first.
        /// </summary>
        /// <param name="binCount">How many bins the spectrum has.</param>
        /// <param name="bands">How many bands to divide it into.</param>
        /// <param name="binHz">How many hertz one bin spans.</param>
        /// <param name="topHz">The highest frequency to show.</param>
        /// <remarks>
        /// Bin zero is never used. It is the DC term: not audio, but whatever constant offset the decoder left in the
        /// samples, and both large and completely unchanging. At the osu.Framework source's 78Hz per bin the lowest band
        /// starts below bin one, so it used to take the DC term in and sit pinned at the ceiling for a whole track.
        ///
        /// Bands never share a bin and never come out empty. Equal ratios give the lowest bands barely more than one bin
        /// wide, so rounding sent several adjacent bars to the same bin - and bars reading the same number move together
        /// and sit at the same height, which reads as a clump stuck in place rather than as a spectrum.
        /// </remarks>
        public static (int First, int Last)[] Ranges(int binCount, int bands, double binHz, double topHz)
        {
            if (binCount <= 1 || bands <= 0 || binHz <= 0 || topHz <= BottomHz)
                return Array.Empty<(int, int)>();

            var ranges = new (int First, int Last)[bands];

            int cursor = 1;

            for (int b = 0; b < bands; b++)
            {
                double lowHz = BottomHz * Math.Pow(topHz / BottomHz, (double)b / bands);
                double highHz = BottomHz * Math.Pow(topHz / BottomHz, (double)(b + 1) / bands);

                int first = Math.Clamp(Math.Max(cursor, (int)Math.Floor(lowHz / binHz)), 1, binCount - 1);
                int last = Math.Min(binCount, Math.Max(first + 1, (int)Math.Ceiling(highHz / binHz)));

                if (last - first < 2 && last < binCount)
                    last++;

                cursor = Math.Max(cursor, last);
                ranges[b] = (first, last);
            }

            return ranges;
        }

        /// <summary>
        /// The highest frequency that actually carries content, quantised so a display using it does not twitch.
        /// </summary>
        /// <remarks>
        /// Plenty of tracks are lowpass filtered: a 128kbps mp3 usually has nothing above 16kHz and some rips nothing
        /// above 11. Every bin past the cut is exactly zero for the whole track, so a fixed top spends the highest bands
        /// on that range and they never move - which reads as a display that is broken rather than one reporting
        /// silence.
        /// </remarks>
        public static double AudibleTop(float[] amplitudes, double binHz, double topHz)
        {
            if (amplitudes == null || amplitudes.Length <= 1)
                return topHz;

            float peak = 0;

            foreach (float value in amplitudes)
                peak = Math.Max(peak, value);

            double audible = peak * 0.01;

            for (int bin = amplitudes.Length - 1; bin >= 1; bin--)
            {
                if (amplitudes[bin] >= audible)
                    return Math.Max(BottomHz * 8, Math.Min(topHz, Math.Ceiling(bin * binHz / 500.0) * 500.0));
            }

            return topHz;
        }
    }
}
