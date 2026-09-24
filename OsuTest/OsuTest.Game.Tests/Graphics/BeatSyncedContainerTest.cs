using System.Collections.Generic;
using NUnit.Framework;
using osu.Framework.Timing;
using ParaTactus;
using OsuTest.Game.Graphics;

namespace OsuTest.Game.Tests.Graphics
{
    /// <summary>
    /// Drives <see cref="BeatSyncedContainer"/> from a real analysed tempo map on a manual clock, without
    /// audio hardware or a draw hierarchy.
    /// </summary>
    /// <remarks>
    /// The point is to test the animation layer's contract rather than the analyser: that beats are counted
    /// from the tempo map, that the beat length handed to animations is the analysed one and changes with the
    /// map, and that a paused clock stops the beats - which is the mechanism that freezes animations when the
    /// music is paused.
    /// </remarks>
    [TestFixture]
    public class BeatSyncedContainerTest
    {
        [Test]
        public void TestBeatsFollowTheTempoMap()
        {
            var source = new ManualClock();
            var clock = new FramedClock(source);

            var timeMap = new BeatTimeMap();
            timeMap.Add(0, 120);
            timeMap.Add(10_000, 160);

            var container = new Probe { Clock = clock, TimeMap = timeMap };

            for (double time = 0; time <= 20_000; time += 5)
            {
                source.CurrentTime = time;
                clock.ProcessFrame();
                container.Tick();
            }

            TestContext.Out.WriteLine($"beats={container.BeatCount} lengths={string.Join(",", container.BeatLengths)}");

            Assert.Multiple(() =>
            {
                // 10s at 120 BPM is 20 beats and 10s at 160 BPM is ~26, so the count has to be in that region -
                // a container that ignored the map and used a constant tempo could not land there.
                Assert.That(container.BeatCount, Is.InRange(40, 50), "beat count across both tempo sections");

                Assert.That(container.BeatLengths, Has.Some.EqualTo(500).Within(1), "120 BPM section must produce 500ms beats");
                Assert.That(container.BeatLengths, Has.Some.EqualTo(375).Within(1), "160 BPM section must produce 375ms beats");

                // The animations are driven by this value, so it has to have changed when the tempo did.
                Assert.That(container.BeatLengths[0], Is.GreaterThan(container.BeatLengths[^1]), "beats must get shorter as tempo rises");
            });
        }

        [Test]
        public void TestPausedClockStopsBeats()
        {
            var source = new ManualClock();
            var clock = new FramedClock(source);

            var timeMap = new BeatTimeMap();
            timeMap.Add(0, 120);

            var container = new Probe { Clock = clock, TimeMap = timeMap };

            for (double time = 0; time <= 5_000; time += 5)
            {
                source.CurrentTime = time;
                clock.ProcessFrame();
                container.Tick();
            }

            int afterPlaying = container.BeatCount;

            // Hold the music clock still: this is what pausing the track does. Animations inside a container on
            // the track's clock must therefore stop advancing, which is exactly what osu! relies on.
            for (int i = 0; i < 200; i++)
            {
                clock.ProcessFrame();
                container.Tick();
            }

            Assert.That(container.BeatCount, Is.EqualTo(afterPlaying), "a stopped music clock must not produce further beats");
            Assert.That(container.BeatCount, Is.GreaterThan(5), "beats must have been produced while playing");
        }

        /// <summary>
        /// Replacing the tempo map at runtime must take effect immediately, and clearing it must not stop the
        /// visuals. This is what allows the audio source to be swapped while the screen is running.
        /// </summary>
        [Test]
        public void TestTempoMapCanBeReplacedWhileRunning()
        {
            var source = new ManualClock();
            var clock = new FramedClock(source);

            var slow = new BeatTimeMap();
            slow.Add(0, 90);

            var fast = new BeatTimeMap();
            fast.Add(0, 180);

            var container = new Probe { Clock = clock, TimeMap = slow };

            run(container, source, clock, 4_000);

            Assert.That(container.BeatLengths[^1], Is.EqualTo(666.67).Within(1), "90 BPM beat length");

            int beforeSwap = container.BeatCount;
            container.BeatLengths.Clear();
            container.TimeMap = fast;

            run(container, source, clock, 8_000);

            Assert.Multiple(() =>
            {
                Assert.That(container.BeatLengths[^1], Is.EqualTo(333.33).Within(1), "180 BPM beat length after the swap");
                Assert.That(container.BeatCount, Is.GreaterThan(beforeSwap), "beats must continue after the swap");
            });

            // Clearing the map is the state during a source change: the container must keep animating on its
            // fallback rather than going silent.
            container.TimeMap = null;
            container.BeatLengths.Clear();

            run(container, source, clock, 12_000);

            Assert.Multiple(() =>
            {
                Assert.That(container.AnalysedBeatLength, Is.Null, "no analysed tempo is available");
                Assert.That(container.BeatLengths, Is.All.EqualTo(BeatSyncedContainer.DEFAULT_BEAT_LENGTH), "falls back to the default tempo");
            });
        }

        /// <summary>
        /// Advancing the manual clock in small steps, ticking the container as the draw loop would.
        /// </summary>
        private static void run(Probe container, ManualClock source, FramedClock clock, double untilMs)
        {
            double from = source.CurrentTime;

            for (double time = from; time <= untilMs; time += 5)
            {
                source.CurrentTime = time;
                clock.ProcessFrame();
                container.Tick();
            }
        }

        /// <summary>
        /// A grid whose beats are not evenly spaced must be followed beat for beat.
        /// </summary>
        /// <remarks>
        /// This is the case the tempo map cannot serve. Counting beats from a map means dividing the clock by a beat
        /// length, which is only arithmetic that means anything while the beat length is constant; on a grid that
        /// changes tempo the division drifts, drops beats at the change and calls some twice. A grid says where the
        /// beats are, so the count is a lookup.
        /// </remarks>
        [Test]
        public void TestBeatsFollowABeatGridExactly()
        {
            var source = new ManualClock();
            var clock = new FramedClock(source);

            // Half-second beats, then a change to 300ms and back - the shape a tempo change makes.
            double[] beats = { 0, 500, 1000, 1500, 1800, 2100, 2400, 2900, 3400, 3900 };
            var grid = BeatGrid.FromNormalisedBeats(beats, 0);

            var container = new Probe { Clock = clock, Grid = grid };

            // Up to just past the last beat in the grid, so the beats counted are the grid's own.
            run(container, source, clock, 3950);

            TestContext.Out.WriteLine($"beats={container.BeatCount}");
            TestContext.Out.WriteLine($"indices={string.Join(",", container.BeatIndices)}");
            TestContext.Out.WriteLine($"lengths={string.Join(",", container.BeatLengths)}");

            Assert.Multiple(() =>
            {
                Assert.That(container.BeatCount, Is.EqualTo(beats.Length), "one callback per beat in the grid");

                // Every beat, in order, exactly once: no gaps at the tempo change and no beat counted twice.
                Assert.That(container.BeatIndices, Is.EqualTo(new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }));

                // The length handed to animations is the tempo around the beat, not the interval to the next one. Those
                // were the same thing until the intervals stopped being even: the analysis quantises beats to 20ms, and
                // in passages where it fires on subdivisions or misses beats the intervals scatter by hundreds of
                // milliseconds around a beat the listener hears as constant. Reporting the measured interval made the
                // visuals lurch on every one of those, so the length is now taken from a window and only the beat
                // positions come from the grid.
                //
                // This fixture changes tempo for three beats and back, which is shorter than any window can resolve, so
                // all ten report the tempo that dominates it. A real change, held for longer than the window, does come
                // through; the cost is that it arrives a few beats late, which is the trade this makes on purpose.
                Assert.That(container.BeatLengths, Is.All.EqualTo(500).Within(1),
                    "the tempo around these beats, and the fixture is mostly at 500ms");

                Assert.That(container.AnalysedBeatLength, Is.Not.Null, "a grid is analysed tempo, not the fallback");
            });

            // Past the end of the grid the beats continue at the last interval rather than stopping. A grid that ends
            // a little before the track does must not freeze the visuals for the rest of the track.
            int atEnd = container.BeatCount;
            container.BeatLengths.Clear();

            run(container, source, clock, 4900);

            TestContext.Out.WriteLine($"after the grid: +{container.BeatCount - atEnd} beats, lengths={string.Join(",", container.BeatLengths)}");

            Assert.Multiple(() =>
            {
                Assert.That(container.BeatCount, Is.GreaterThan(atEnd), "beats must continue past the last one in the grid");
                Assert.That(container.BeatLengths, Is.All.EqualTo(500).Within(1), "continuing at the last known beat length");
            });
        }

        /// <summary>
        /// A grid whose intervals alternate either side of one period must still animate at one speed.
        /// </summary>
        /// <remarks>
        /// This is what the model actually reports on a track that holds a tempo. Its beats are quantised to a 20ms
        /// analysis frame, so a 600ms period comes back as 580, 600, 620, 580, 600 and so on, and taking the interval
        /// to the next beat as the animation's length made the visuals speed up and slow down on every single beat of
        /// music that does not change at all.
        ///
        /// The beat times themselves must not move: a pulse that is smoothed in position as well as in length would
        /// drift off the music, which is worse than the jitter it was fixing.
        /// </remarks>
        [Test]
        public void TestQuantisedBeatsAnimateAtOneSpeed()
        {
            var source = new ManualClock();
            var clock = new FramedClock(source);

            // A steady 600ms period as reported by a tracker whose beats are quantised to 20ms.
            var beats = new double[41];
            double at = 0;

            for (int i = 0; i < beats.Length; i++)
            {
                beats[i] = at;
                at += i % 3 == 0 ? 580 : i % 3 == 1 ? 600 : 620;
            }

            var grid = BeatGrid.FromNormalisedBeats(beats, 0);
            var container = new Probe { Clock = clock, Grid = grid };

            run(container, source, clock, beats[^1] - 10);

            TestContext.Out.WriteLine($"lengths={string.Join(",", container.BeatLengths)}");

            Assert.That(container.BeatLengths.Count, Is.GreaterThan(10), "the run must cover several beats");

            double first = container.BeatLengths[0];
            double worst = 0;

            foreach (double length in container.BeatLengths)
            {
                double difference = length - first;

                if (difference < 0)
                    difference = -difference;

                if (difference > worst)
                    worst = difference;
            }

            Assert.That(worst, Is.LessThan(1),
                $"the animation length varied by {worst:0.#}ms although the tempo is constant");
        }

        /// <summary>
        /// The grid is the model's answer, so it wins when both are present, and clearing it falls back rather than
        /// going silent.
        /// </summary>
        [Test]
        public void TestTheGridTakesPrecedenceAndClearingItFallsBack()
        {
            var source = new ManualClock();
            var clock = new FramedClock(source);

            var map = new BeatTimeMap();
            map.Add(0, 90);

            var grid = BeatGrid.FromNormalisedBeats(new double[] { 0, 500, 1000, 1500, 2000, 2500 }, 0);

            var container = new Probe { Clock = clock, TimeMap = map, Grid = grid };

            run(container, source, clock, 2_500);

            Assert.That(container.BeatLengths[^1], Is.EqualTo(500).Within(1), "the grid's beat length, not the map's 666.67");

            container.Grid = null;
            container.BeatLengths.Clear();

            run(container, source, clock, 5_000);

            Assert.That(container.BeatLengths[^1], Is.EqualTo(666.67).Within(1), "falls back to the tempo map");

            container.TimeMap = null;
            container.BeatLengths.Clear();

            run(container, source, clock, 8_000);

            Assert.Multiple(() =>
            {
                Assert.That(container.AnalysedBeatLength, Is.Null, "nothing is analysed once both are gone");
                Assert.That(container.BeatLengths, Is.All.EqualTo(BeatSyncedContainer.DEFAULT_BEAT_LENGTH));
            });
        }

        /// <summary>
        /// A divisor has to subdivide the grid's own beats, which are not evenly spaced either.
        /// </summary>
        [Test]
        public void TestADivisorSubdividesTheGridsBeats()
        {
            var source = new ManualClock();
            var clock = new FramedClock(source);

            double[] beats = { 0, 400, 800, 1200 };
            var grid = BeatGrid.FromNormalisedBeats(beats, 0);

            var container = new Probe { Clock = clock, Grid = grid, Divisor = 2 };

            run(container, source, clock, 1_400);

            TestContext.Out.WriteLine($"indices={string.Join(",", container.BeatIndices)} lengths={string.Join(",", container.BeatLengths)}");

            Assert.Multiple(() =>
            {
                Assert.That(container.BeatIndices, Is.EqualTo(new[] { 0, 1, 2, 3, 4, 5, 6, 7 }), "two sub-beats per beat");
                Assert.That(container.BeatLengths, Is.All.EqualTo(200).Within(1), "half of a 400ms beat");
            });
        }

        /// <summary>
        /// Exposes the protected update and the beat callback for assertion.
        /// </summary>
        private partial class Probe : BeatSyncedContainer
        {
            public int BeatCount { get; private set; }

            public List<int> BeatIndices { get; } = new List<int>();

            public List<double> BeatLengths { get; } = new List<double>();

            public void Tick() => Update();

            protected override void OnNewBeat(int beatIndex, double beatLength)
            {
                BeatCount++;
                BeatIndices.Add(beatIndex);
                BeatLengths.Add(beatLength);
            }
        }
    }
}
