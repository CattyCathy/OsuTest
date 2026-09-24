#nullable enable

using System;
using osu.Framework.Graphics;
using osu.Framework.Graphics.Containers;
using ParaTactus;

namespace OsuTest.Game.Graphics
{
    /// <summary>
    /// A container that fires a callback once per beat, using an analysed <see cref="BeatTimeMap"/>.
    /// </summary>
    /// <remarks>
    /// This is the stripped-down equivalent of osu!'s <c>BeatSyncedContainer</c>, reconstructed on top of
    /// this project's own tempo analysis because <c>osu.Game</c> is not referenced here. The mechanics it
    /// reproduces, and why each one exists:
    /// <list type="bullet">
    /// <item><b>Poll, do not schedule.</b> Every frame the beat number is derived from the clock as a pure
    /// function of time. A timer that ticks every beat would drift, break on seek and pile up after a
    /// stall; a derived index cannot.</item>
    /// <item><b>Fire only on change.</b> Comparing against the previous beat number is what makes the
    /// polling idempotent - one call per beat no matter the frame rate.</item>
    /// <item><b>Cancel out the detection lag.</b> An event can only be noticed on the frame after the beat
    /// happened. The callback runs inside a delayed sequence backdated by the time already elapsed since
    /// the beat, so animations start exactly on the beat rather than a frame or two late.</item>
    /// <item><b>Keep a usable idle behaviour.</b> With no analysed map, or while the track is paused, the
    /// container falls back to a fixed tempo instead of going dead - the same reason osu!'s logo keeps
    /// pulsing at 60 BPM on the main menu when nothing is playing.</item>
    /// </list>
    /// </remarks>
    public partial class BeatSyncedContainer : Container
    {
        /// <summary>Fallback beat length used when no tempo information is available.</summary>
        public const double DEFAULT_BEAT_LENGTH = 500;

        /// <summary>
        /// How long before a beat the callback fires, so that animations have room to ease in and reach
        /// their peak exactly on the beat.
        /// </summary>
        public double EarlyActivationMilliseconds { get; set; } = 60;

        /// <summary>How many beats per beat length to trigger. 2 fires on every half beat.</summary>
        public int Divisor { get; set; } = 1;

        /// <summary>
        /// The beat grid to follow, from the model. Takes precedence over <see cref="TimeMap"/> when both are set.
        /// </summary>
        /// <remarks>
        /// A grid is what a track with a changing tempo needs. The map below it is a list of tempo sections, and
        /// deriving a beat number from one means dividing the clock by a beat length - which is only a valid way to
        /// count beats while the beat length is constant. A grid says where the beats are instead, so the number comes
        /// from a lookup and stays right through a ramp.
        /// </remarks>
        public BeatGrid? Grid { get; set; }

        /// <summary>
        /// The tempo map to follow. When null (or empty) the container free-runs at
        /// <see cref="DEFAULT_BEAT_LENGTH"/> so that visuals still animate.
        /// </summary>
        public BeatTimeMap? TimeMap { get; set; }

        /// <summary>The beat time returned by the last successful analysis. Null when free-running.</summary>
        public double? AnalysedBeatLength { get; private set; }

        /// <summary>Index of the most recently fired beat, or -1 before the first.</summary>
        public int CurrentBeatIndex { get; private set; } = -1;

        /// <summary>Milliseconds until the next beat.</summary>
        public double TimeUntilNextBeat { get; private set; }

        /// <summary>Milliseconds since the last beat.</summary>
        public double TimeSinceLastBeat { get; private set; }

        private int lastBeat = int.MinValue;

        /// <summary>
        /// Invoked once per beat, with the animation start time already backdated to the beat itself.
        /// </summary>
        public event Action<int, double>? OnBeat;

        /// <summary>
        /// Invoked once per beat, with the animation start time already backdated to the beat itself.
        /// </summary>
        protected virtual void OnNewBeat(int beatIndex, double beatLength)
        {
        }

        protected override void Update()
        {
            base.Update();

            double currentTime = Clock.CurrentTime + EarlyActivationMilliseconds;

            int divisor = Math.Max(1, Divisor);
            int beatIndex;
            double beatLength;
            double beatStart;

            // How long until the next pulse, kept separate from the length reported to the animation. They are not the
            // same quantity: the pulse has to land on the grid's next beat, while the length handed to an animation is
            // the tempo around here so that it does not change merely because the grid's beats are quantised.
            double step;

            int gridIndex = Grid != null ? Grid.IndexAt(currentTime) : -1;

            if (Grid != null && !Grid.IsEmpty)
            {
                if (gridIndex < 0)
                {
                    // Before the grid's first beat it is extended backwards at its own opening interval rather than
                    // falling back to the free-run tempo. A track with an intro would otherwise pulse at 120 BPM until
                    // its first beat and then jump to the track's real tempo, which looks like the visuals starting
                    // out of sync and catching up.
                    step = Math.Max(1, Grid.BeatLengthAt(0)) / divisor;
                    int back = (int)Math.Ceiling((Grid.Beats[0] - currentTime) / step);

                    beatLength = ClampBeatLength(smoothedBeatLength(Grid, 0)) / divisor;
                    beatIndex = -back * divisor;
                    beatStart = Grid.Beats[0] - back * step;
                    AnalysedBeatLength = beatLength;
                }
                else
                {
                    // Taken from the grid, never from the reported length. Using the reported length here was tried and
                    // is wrong: when the two differ the pulse fires early or late, and because the beat it belongs to is
                    // then looked up again from the shifted time, the next one fires late by the same amount and the
                    // rhythm wobbles exactly as if the tempo were changing.
                    double span = gridIndex + 1 < Grid.Beats.Count
                        ? Grid.Beats[gridIndex + 1] - Grid.Beats[gridIndex]
                        : Math.Max(1, Grid.BeatLengthAt(gridIndex));

                    step = span / divisor;

                    double offset = currentTime - Grid.Beats[gridIndex];

                    // Deliberately not clamped: past the grid's last beat there is no next one to stop at, and the
                    // beats have to keep coming at the last known interval rather than freezing.
                    int sub = Math.Max(0, (int)Math.Floor(offset / step));

                    beatIndex = gridIndex * divisor + sub;
                    beatStart = Grid.Beats[gridIndex] + sub * step;

                    beatLength = ClampBeatLength(smoothedBeatLength(Grid, gridIndex)) / divisor;
                    AnalysedBeatLength = beatLength;
                }
            }
            else
            {
                // Free-run on the wall clock when the music is not playing, so the visuals never freeze just
                // because there is no track (or it is paused).
                step = ClampBeatLength(TimeMap?.BeatLengthAt(currentTime) ?? DEFAULT_BEAT_LENGTH) / divisor;
                AnalysedBeatLength = TimeMap != null && TimeMap.KeyframeCount > 0 ? step : null;

                if (step <= 0)
                    step = DEFAULT_BEAT_LENGTH;

                beatLength = step;
                beatIndex = (int)Math.Floor(currentTime / step);
                beatStart = beatIndex * step;
            }

            TimeUntilNextBeat = beatStart + step - currentTime;
            TimeSinceLastBeat = currentTime - beatStart;

            if (beatIndex == lastBeat)
                return;

            // Guard against a huge burst of callbacks after an extreme seek or a long analysis gap.
            if (lastBeat != int.MinValue && Math.Abs(beatIndex - lastBeat) > 8)
            {
                lastBeat = beatIndex;
                return;
            }

            lastBeat = beatIndex;
            CurrentBeatIndex = beatIndex;

            // Backdate the sequence so the animation is anchored to the beat, not to this frame.
            using (BeginDelayedSequence(-TimeSinceLastBeat))
            {
                OnNewBeat(beatIndex, beatLength);
                OnBeat?.Invoke(beatIndex, beatLength);
            }
        }

        /// <summary>
        /// Keeps analysed beat lengths sane: a garbage tempo estimate must never make the visuals strobe.
        /// </summary>
        /// <summary>
        /// How long a beat lasts, taken as the tempo around it rather than as the interval to the next beat.
        /// </summary>
        /// <remarks>
        /// The single interval was what this used, and it is the wrong quantity wherever the tracker's beats are not
        /// evenly spaced. Two things cause that. The grid's beats are quantised to a 20ms analysis frame, so a 600ms
        /// period comes back as 580, 600, 620 and the animation changes speed on every beat of music that does not
        /// change at all. And where the tracker fires on subdivisions or misses beats, the intervals scatter much
        /// further - Designant's 145-175s section mixes gaps between 200ms and 460ms around a 300ms beat - so the
        /// animation wobbles continuously through a passage whose tempo a listener hears as constant.
        ///
        /// A windowed tempo cannot wobble, because one wrong interval among a dozen does not move a median, and
        /// <see cref="BeatGrid.BpmAt"/> already falls back to a much wider window when the beats it is measuring between
        /// disagree with each other. That fallback is what makes this safe to use unconditionally: a scattered passage
        /// is answered from its surroundings rather than from the scatter.
        ///
        /// What this deliberately does not change is where a beat starts. The pulse still lands exactly on the beat the
        /// grid reports; only the length of the animation driven from it is smoothed. A grid whose beats are right
        /// therefore still shows every beat on time, which is the property the visuals exist for.
        ///
        /// The cost is that at a genuine tempo change the length lags by a few beats, because the window still holds
        /// more of the old tempo than the new one. That is a deliberate trade: a change is heard once and passes,
        /// whereas an uneven pulse through a steady passage is visible for its whole length.
        /// </remarks>
        private static double smoothedBeatLength(BeatGrid grid, int index)
        {
            double beat = grid.Beats[Math.Clamp(index, 0, grid.Beats.Count - 1)];
            double bpm = grid.BpmAt(beat);

            return bpm > 0 ? 60000.0 / bpm : grid.BeatLengthAt(index);
        }

        private static double ClampBeatLength(double beatLength)
            => Math.Clamp(beatLength, 150, 2000);
    }
}
