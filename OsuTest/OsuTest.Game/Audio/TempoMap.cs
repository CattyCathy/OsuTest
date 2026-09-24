using System;
using System.Collections.Generic;
using ParaTactus;

namespace OsuTest.Game.Audio
{
    /// <summary>
    /// A linear tempo map: the track is written in metrical units, and the wall-clock start of each
    /// section is derived from how many beats the previous sections consumed.
    /// </summary>
    /// <remarks>
    /// Deliberately built in beats rather than in seconds. Authoring a tempo-changing track by absolute
    /// timestamps means every section boundary has to be recomputed by hand when a tempo changes, and the
    /// rounding error accumulates into beat placement that is subtly wrong - which would then be measured
    /// as a failure of the analyser rather than of the fixture.
    /// </remarks>
    public class TempoMap
    {
        private readonly List<Segment> segments = new List<Segment>();

        /// <summary>
        /// One tempo section, expressed as a number of beats at a fixed tempo.
        /// </summary>
        public readonly record struct Segment(double StartTime, double Bpm, int Beats);

        /// <summary>All sections, with their start times resolved.</summary>
        public IReadOnlyList<Segment> Segments => segments;

        /// <summary>Total length in milliseconds.</summary>
        public double Duration { get; private set; }

        /// <summary>Adds a section of <paramref name="bars"/> bars in 4/4 at the given tempo.</summary>
        public TempoMap AddSection(double bpm, int bars)
        {
            if (bpm <= 0)
                throw new ArgumentOutOfRangeException(nameof(bpm));

            if (bars <= 0)
                throw new ArgumentOutOfRangeException(nameof(bars));

            double start = segments.Count == 0 ? 0 : segments[^1].StartTime + segments[^1].Beats * 60000 / segments[^1].Bpm;
            segments.Add(new Segment(start, bpm, bars * 4));
            Duration = start + bars * 4 * 60000 / bpm;

            return this;
        }

        /// <summary>
        /// Every beat time in the map. Beats are generated inside each section from that section's own
        /// start time, so a tempo change never shifts the phase of the beats before it.
        /// </summary>
        public double[] BuildBeatTimes()
        {
            var beats = new List<double>();

            foreach (var segment in segments)
            {
                double beatLength = 60000 / segment.Bpm;

                for (int i = 0; i < segment.Beats; i++)
                    beats.Add(segment.StartTime + i * beatLength);
            }

            return beats.ToArray();
        }

        /// <summary>
        /// The tempo in effect at a given time, expressed as a beat length in milliseconds.
        /// </summary>
        public double NearestBeatLength(double time)
        {
            foreach (var segment in segments)
            {
                double end = segment.StartTime + segment.Beats * 60000 / segment.Bpm;

                if (time >= segment.StartTime && time < end)
                    return 60000 / segment.Bpm;
            }

            return segments.Count == 0 ? 500 : 60000 / segments[^1].Bpm;
        }
    }
}
