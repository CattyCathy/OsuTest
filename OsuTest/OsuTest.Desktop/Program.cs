using osu.Framework.Platform;
using osu.Framework;
using OsuTest.Game;
using ParaTactus;

namespace OsuTest.Desktop
{
    public static class Program
    {
        public static void Main()
        {
            using (GameHost host = Host.GetSuitableDesktopHost(@"OsuTest"))
            using (osu.Framework.Game game = new OsuTestGame())
                host.Run(game);
        }
    }
}
