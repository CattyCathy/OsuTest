using osu.Framework.Allocation;
using osu.Framework.Graphics;
using osu.Framework.Graphics.Containers;
using osu.Framework.IO.Stores;
using osuTK;
using OsuTest.Resources;
using ParaTactus;

namespace OsuTest.Game
{
    public partial class OsuTestGameBase : osu.Framework.Game
    {
        // Anything in this class is shared between the test browser and the game implementation.
        // It allows for caching global dependencies that should be accessible to tests, or changing
        // the screen scaling for all components including the test browser and framework overlays.

        protected override Container<Drawable> Content { get; }

        /// <summary>
        /// Font families bundled for non-Latin text, mirroring osu!lazer's font set.
        /// </summary>
        /// <remarks>
        /// osu!framework never rasterises TTF or system fonts at runtime, and the fonts it
        /// ships (Roboto, Roboto Condensed, FontAwesome) contain no CJK glyphs, so Chinese,
        /// Japanese and Korean characters are silently dropped from <c>SpriteText</c>.
        /// osu!lazer solves this by baking Noto bitmap fonts and registering them with
        /// <c>Game.AddFont</c> in <c>OsuGameBase.InitialiseFonts()</c>; the same fonts and
        /// the same registration are used here.
        /// </remarks>
        public static class FontFamilies
        {
            /// <summary>Latin, digits, punctuation, fullwidth forms (also covers CJK text).</summary>
            public const string Basic = "Noto-Basic";

            /// <summary>Han ideographs for Chinese and Japanese.</summary>
            public const string Cjk = "Noto-CJK-Basic";

            /// <summary>Additional CJK compatibility ideographs.</summary>
            public const string CjkCompatibility = "Noto-CJK-Compatibility";

            /// <summary>Hangul syllables for Korean.</summary>
            public const string Hangul = "Noto-Hangul";
        }

        /// <summary>
        /// The family to use for general UI text so that Chinese renders correctly.
        /// </summary>
        public const string DefaultFontFamily = FontFamilies.Basic;

        protected OsuTestGameBase()
        {
            // Ensure game and tests scale with window size and screen DPI.
            base.Content.Add(Content = new DrawSizePreservingFillContainer
            {
                // You may want to change TargetDrawSize to your "default" resolution, which will decide how things scale and position when using absolute coordinates.
                TargetDrawSize = new Vector2(1366, 768)
            });
        }

        [BackgroundDependencyLoader]
        private void load()
        {
            Resources.AddStore(new DllResourceStore(typeof(OsuTestResources).Assembly));

            // Each of these is a baked bitmap font: a .bin descriptor plus _00.png..
            // atlas pages, resolved relative to the resource store.
            AddFont(Resources, @"Fonts/Noto-Basic");
            AddFont(Resources, @"Fonts/Noto-CJK-Basic");
            AddFont(Resources, @"Fonts/Noto-CJK-Compatibility");
            AddFont(Resources, @"Fonts/Noto-Hangul");
        }
    }
}
