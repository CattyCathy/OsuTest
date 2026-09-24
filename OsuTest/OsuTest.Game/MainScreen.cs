using System;
using System.Linq;
using System.Reflection;
using osu.Framework.Allocation;
using osu.Framework.Graphics;
using osu.Framework.Graphics.Containers;
using osu.Framework.Graphics.Shapes;
using osu.Framework.Graphics.Sprites;
using osu.Framework.Graphics.UserInterface;
using osu.Framework.Input;
using osu.Framework.Screens;
using osuTK.Graphics;
using OsuTest.Game.Graphics;
using ParaTactus;

namespace OsuTest.Game
{
    public partial class MainScreen : Screen
    {
        [BackgroundDependencyLoader]
        private void load()
        {
            InternalChildren = new Drawable[]
            {
                new Box
                {
                    Colour = Color4.Violet,
                    RelativeSizeAxes = Axes.Both,
                },
                new SpriteText
                {
                    Y = 20,
                    Text = "Main Screen",
                    Anchor = Anchor.TopCentre,
                    Origin = Anchor.TopCentre,
                    Font = FontUsage.Default.With(family: OsuTestGameBase.DefaultFontFamily, size: 40)
                },
                new SpinningBox
                {
                    Anchor = Anchor.Centre,
                },

                // Replaces the fixed-duration spinner above with one driven by an analysed tempo map.
                new BeatSyncDemo
                {
                    RelativeSizeAxes = Axes.Both,
                },
            };
        }
    }
}
