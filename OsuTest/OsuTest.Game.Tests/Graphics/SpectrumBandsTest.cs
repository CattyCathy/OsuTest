using System;
using NUnit.Framework;
using OsuTest.Game.Graphics;
using ParaTactus;

namespace OsuTest.Game.Tests.Graphics
{
    /// <summary>
    /// The spectrum display's band mapping has to be provably right, because a band it never reads looks exactly like a
    /// band whose audio is silent.
    /// </summary>
    /// <remarks>
    /// A dead bar in the display can mean two entirely different things and there is no way to tell them apart by
    /// looking: either the bins that bar covers are all zero in the audio, or the mapping never reads them. The mapping
    /// is arithmetic and can be checked; the audio cannot be checked from a screenshot. These tests settle which half is
    /// which, and they exist because a report of "some bars never move" was investigated twice by changing the mapping
    /// when the mapping was not what was wrong.
    /// </remarks>
    [TestFixture]
    public class SpectrumBandsTest
    {
        /// <summary>osu.Framework's amplitude source: 256 bins over 0-20kHz.</summary>
        private const int frameworkBins = 256;
        private const double frameworkBinHz = 20000.0 / 256;
        private const int bands = 24;

        [Test]
        public void TestEveryBandCoversADistinctNonEmptyRunOfBins()
        {
            var ranges = SpectrumBands.Ranges(frameworkBins, bands, frameworkBinHz, SpectrumBands.TopHz);

            Assert.That(ranges.Length, Is.EqualTo(bands));

            int previousLast = 1;

            foreach (var (first, last) in ranges)
            {
                Assert.That(first, Is.GreaterThanOrEqualTo(1), "bin zero is the DC term and is never audio");
                Assert.That(last, Is.GreaterThan(first), $"band from {first} to {last} is empty");
                Assert.That(last, Is.LessThanOrEqualTo(frameworkBins));
                Assert.That(first, Is.GreaterThanOrEqualTo(previousLast), $"band at {first} overlaps the one below");

                previousLast = last;
            }
        }

        [Test]
        public void TestTheBandsRiseInFrequency()
        {
            var ranges = SpectrumBands.Ranges(frameworkBins, bands, frameworkBinHz, SpectrumBands.TopHz);

            int previousMiddle = 0;

            for (int b = 0; b < ranges.Length; b++)
            {
                int middle = (ranges[b].First + ranges[b].Last) / 2;

                Assert.That(middle, Is.GreaterThan(previousMiddle),
                    $"band {b} is not above band {b - 1}: a display whose bands are out of order is wrong regardless of the audio");

                previousMiddle = middle;
            }
        }

        /// <summary>
        /// Prints what each band reads, so a bar that never moves can be traced to a frequency rather than guessed at.
        /// </summary>
        [Test]
        public void TestBandTable()
        {
            var ranges = SpectrumBands.Ranges(frameworkBins, bands, frameworkBinHz, SpectrumBands.TopHz);

            TestContext.Out.WriteLine($"osu.Framework source: {frameworkBins} bins, {frameworkBinHz:0.##}Hz per bin");
            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("  band   bin range       frequency range");

            for (int b = 0; b < ranges.Length; b++)
            {
                var (first, last) = ranges[b];

                TestContext.Out.WriteLine($"{b,6} {first,5}..{last - 1,-5} "
                                          + $"{first * frameworkBinHz,8:0} - {last * frameworkBinHz,7:0} Hz");

                Assert.That(last * frameworkBinHz, Is.LessThanOrEqualTo(SpectrumBands.TopHz + frameworkBinHz));
            }

            // The mirror pairs the display draws, so a reported bar index can be read straight back to a band.
            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("bar (from either edge) -> band:");
            TestContext.Out.WriteLine("  " + string.Join(", ", Array.ConvertAll(Enumerable_Range(bands), b => $"{b + 1}->{bands - 1 - b}")));
        }

        private static int[] Enumerable_Range(int count)
        {
            var values = new int[count];

            for (int i = 0; i < count; i++)
                values[i] = i;

            return values;
        }

        [Test]
        public void TestTheTopFollowsTheAudio()
        {
            // A signal with content only in the bottom quarter of the spectrum, as a lowpass filtered track has.
            var amplitudes = new float[frameworkBins];

            for (int bin = 1; bin < frameworkBins / 4; bin++)
                amplitudes[bin] = 10;

            double top = SpectrumBands.AudibleTop(amplitudes, frameworkBinHz, SpectrumBands.TopHz);

            Assert.That(top, Is.LessThan(SpectrumBands.TopHz),
                "a track with no content above a quarter of the spectrum must not have its bands spread over the empty part");
            Assert.That(top, Is.GreaterThanOrEqualTo(SpectrumBands.BottomHz * 8));
            Assert.That(top, Is.LessThan(frameworkBins / 4 * frameworkBinHz + 600),
                "the cut has to land just above the content, not far above it");

            // And a full-band signal keeps the full range.
            for (int bin = 1; bin < frameworkBins; bin++)
                amplitudes[bin] = 10;

            Assert.That(SpectrumBands.AudibleTop(amplitudes, frameworkBinHz, SpectrumBands.TopHz),
                Is.EqualTo(SpectrumBands.TopHz).Within(1));
        }
    }
}
