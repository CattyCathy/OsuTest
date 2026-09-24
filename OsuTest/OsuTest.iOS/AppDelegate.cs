using osu.Framework.iOS;
using OsuTest.Game;
using ParaTactus;

namespace OsuTest.iOS
{
    /// <inheritdoc />
    public class AppDelegate : GameApplicationDelegate
    {
        /// <inheritdoc />
        protected override osu.Framework.Game CreateGame() => new OsuTestGame();
    }
}
