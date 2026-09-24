using osu.Framework.Testing;
using ParaTactus;

namespace OsuTest.Game.Tests.Visual
{
    public abstract partial class OsuTestTestScene : TestScene
    {
        protected override ITestSceneTestRunner CreateRunner() => new OsuTestTestSceneTestRunner();

        private partial class OsuTestTestSceneTestRunner : OsuTestGameBase, ITestSceneTestRunner
        {
            private TestSceneTestRunner.TestRunner runner;

            protected override void LoadAsyncComplete()
            {
                base.LoadAsyncComplete();
                Add(runner = new TestSceneTestRunner.TestRunner());
            }

            public void RunTestBlocking(TestScene test) => runner.RunTestBlocking(test);
        }
    }
}
