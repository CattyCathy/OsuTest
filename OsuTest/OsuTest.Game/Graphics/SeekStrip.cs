using System;
using osu.Framework.Graphics;
using osu.Framework.Graphics.Containers;
using osu.Framework.Graphics.Shapes;
using osu.Framework.Input.Events;
using osuTK;
using osuTK.Graphics;
using ParaTactus;

namespace OsuTest.Game.Graphics
{
    /// <summary>
    /// A draggable strip that seeks the track.
    /// </summary>
    /// <remarks>
    /// Hand drawn rather than built on <c>SliderBar</c>, which in this version of the framework is an abstract base with
    /// no visual implementation and no minimum or maximum of its own, so using it would mean writing this anyway plus a
    /// normalisation layer on top. What is needed here is one number in and one number out.
    ///
    /// It exists because the failures being chased are in specific sections of specific tracks. Waiting for a three
    /// hundred second track to reach 284 seconds, repeatedly, is not a way to work.
    /// </remarks>
    public partial class SeekStrip : CompositeDrawable
    {
        /// <summary>Raised with a fraction of the track when the user drags or clicks.</summary>
        public Action<double> SeekRequested;

        /// <summary>The fraction the strip is showing, from 0 to 1.</summary>
        public double Progress { get; private set; }

        /// <summary>
        /// How many frames a seek is protected from the position readout.
        /// </summary>
        /// <remarks>
        /// Counted in frames rather than milliseconds on purpose. The first version used the drawable's own clock, and
        /// if that clock does not advance the way the code assumes the hold never expires and the strip stops following
        /// playback entirely - which is exactly what happened: the seek worked, the audio moved, and the handle stayed
        /// pinned at the start because every later update was still being suppressed.
        /// </remarks>
        private const int hold_frames = 30;

        /// <summary>How many frames after a release the seek is asked for a second time.</summary>
        private const int reassert_delay = 10;

        private readonly Box fill;
        private readonly Box nub;

        private int held;
        private int reassert;

        public SeekStrip()
        {
            InternalChildren = new Drawable[]
            {
                new Box
                {
                    RelativeSizeAxes = Axes.Both,
                    Colour = new Color4(1f, 1f, 1f, 0.12f),
                },
                fill = new Box
                {
                    RelativeSizeAxes = Axes.Y,
                    Width = 0,
                    Colour = new Color4(0.4f, 0.85f, 1f, 0.75f),
                },
                nub = new Box
                {
                    Origin = Anchor.CentreLeft,
                    Size = new Vector2(8, 20),
                    Colour = new Color4(0.4f, 0.85f, 1f, 1f),
                },
            };
        }

        /// <summary>
        /// Shows where the playhead is, unless a seek is still in flight.
        /// </summary>
        public void ShowPlayhead(double fraction)
        {
            if (held > 0)
                return;

            Progress = Math.Clamp(fraction, 0, 1);
        }

        /// <summary>Asks for the strip to show the scrubbed position for a short while.</summary>
        public void Hold() => held = hold_frames;

        protected override void Update()
        {
            base.Update();

            if (held > 0)
                held--;

            if (reassert > 0 && --reassert == 0)
            {
                note("重试", 0, Progress);
                SeekRequested?.Invoke(Progress);
            }

            // Laid out every frame from the stored fraction rather than when the fraction is set. A drag can arrive
            // before the strip has been laid out, in which case its width is still zero and a one-off calculation puts
            // the handle at the start and leaves it there.
            float width = DrawWidth;
            float x = (float)(width * Progress);

            fill.Width = x;
            nub.X = x;
        }

        /// <summary>
        /// A short record of the input this strip has seen, for diagnosis.
        /// </summary>
        /// <remarks>
        /// Added after three separate theories about why a scrub came undone on release all turned out to be wrong. The
        /// only reliable way to find out which events the strip actually receives, and in what order, is to have it say
        /// so on screen.
        /// </remarks>
        public string Trace { get; private set; } = "无";

        private void note(string what, float x, double fraction)
        {
            Trace = $"{what} x={x:0} 比例={fraction:0.000}";
        }

        protected override bool OnMouseDown(MouseDownEvent e)
        {
            note("按下", e.MousePosition.X, Math.Clamp(e.MousePosition.X / Math.Max(1, DrawWidth), 0, 1));
            seekTo(e.MousePosition.X);
            return true;
        }

        protected override bool OnDragStart(DragStartEvent e) => true;

        protected override void OnDrag(DragEvent e)
        {
            base.OnDrag(e);
            note("拖动", e.MousePosition.X, Math.Clamp(e.MousePosition.X / Math.Max(1, DrawWidth), 0, 1));
            seekTo(e.MousePosition.X);
        }

        protected override void OnMouseUp(MouseUpEvent e)
        {
            base.OnMouseUp(e);
            note("松开", e.MousePosition.X, Math.Clamp(e.MousePosition.X / Math.Max(1, DrawWidth), 0, 1));

            // The seek is asked for again a few frames later. Whatever the audio source does with the last request
            // during a release, it evidently does not keep it: the trace shows the position and the fraction agreeing
            // at the moment of release and the reported position then being somewhere else entirely, which is the
            // seek being dropped rather than applied. One repeat after the dust settles is cheaper than another
            // theory.
            reassert = reassert_delay;
        }

        /// <summary>
        /// Swallows the click that follows a press on the strip.
        /// </summary>
        /// <remarks>
        /// Returning true from the press alone was not enough to keep a click from bubbling to whatever else is
        /// listening, though swallowing it here did not turn out to be the cause of the scrub coming undone either.
        /// </remarks>
        protected override bool OnClick(ClickEvent e)
        {
            note("点击", e.MousePosition.X, Math.Clamp(e.MousePosition.X / Math.Max(1, DrawWidth), 0, 1));
            reassert = reassert_delay;
            return true;
        }

        /// <summary>
        /// Turns a position inside the strip into a seek.
        /// </summary>
        /// <remarks>
        /// The position comes from the event rather than from converting the screen-space mouse position by hand. The
        /// event already carries it in this drawable's own space, which is the space the fraction is a fraction of, and
        /// doing the conversion separately is what left every seek landing at the start.
        /// </remarks>
        private void seekTo(float localX)
        {
            float width = DrawWidth;

            if (width <= 0)
                return;

            Progress = Math.Clamp(localX / width, 0, 1);
            SeekRequested?.Invoke(Progress);
        }
    }
}
