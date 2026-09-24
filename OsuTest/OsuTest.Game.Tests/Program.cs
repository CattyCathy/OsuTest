using osu.Framework;
using osu.Framework.Platform;
using ParaTactus;

namespace OsuTest.Game.Tests
{
    public static class Program
    {
        public static void Main()
        {
            using (GameHost host = Host.GetSuitableDesktopHost("visual-tests"))
            using (var game = new OsuTestTestBrowser())
                host.Run(game);
        }
    }
}
