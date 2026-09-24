using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using OsuTest.Game.Audio;
using ParaTactus;

namespace OsuTest.Game.Tests.Audio
{
    /// <summary>
    /// Tests the directory listing behind the in-game file picker.
    /// </summary>
    /// <remarks>
    /// This is the part of the picker that can be wrong without anything looking wrong on screen - a missing
    /// extension filter silently hides the user's music, and an unhandled <c>UnauthorizedAccessException</c>
    /// inside the update thread takes the whole screen down. Both are cheap to test against real temporary
    /// directories.
    /// </remarks>
    [TestFixture]
    public class AudioFileBrowserTest
    {
        private string root = null!;

        [SetUp]
        public void SetUp()
        {
            root = Path.Combine(Path.GetTempPath(), $"osutest-browser-{Guid.NewGuid():N}");
            Directory.CreateDirectory(root);

            Directory.CreateDirectory(Path.Combine(root, "album-b"));
            Directory.CreateDirectory(Path.Combine(root, "album-a"));

            File.WriteAllText(Path.Combine(root, "song.mp3"), "x");
            File.WriteAllText(Path.Combine(root, "SONG.OGG"), "x");
            File.WriteAllText(Path.Combine(root, "notes.txt"), "x");
            File.WriteAllText(Path.Combine(root, "cover.png"), "x");
        }

        [TearDown]
        public void TearDown()
        {
            try
            {
                Directory.Delete(root, true);
            }
            catch (IOException)
            {
                // A leftover temp directory is not worth failing a run over.
            }
        }

        [Test]
        public void TestFiltersAndSorts()
        {
            var listing = AudioFileBrowser.List(root, AudioFileBrowser.DEFAULT_EXTENSIONS);

            Assert.Multiple(() =>
            {
                Assert.That(listing.Error, Is.Null);

                // Both audio files and only those: the extension match has to be case-insensitive, and the
                // non-audio files must be filtered out. Ordering is by ordinal-ignore-case, which puts
                // "song.mp3" before "SONG.OGG" because the comparison ignores case but the sort does not
                // normalise it.
                Assert.That(listing.Files.Select(f => f.Name.ToLowerInvariant()), Is.EqualTo(new[] { "song.mp3", "song.ogg" }));

                Assert.That(listing.Directories.Select(d => d.Name), Is.EqualTo(new[] { "album-a", "album-b" }), "directories sorted");
            });
        }

        [Test]
        public void TestCustomExtensionListIsHonoured()
        {
            var listing = AudioFileBrowser.List(root, new[] { ".mp3" });

            Assert.That(listing.Files.Select(f => f.Name), Is.EqualTo(new[] { "song.mp3" }));
        }

        [Test]
        public void TestTruncationIsReported()
        {
            var listing = AudioFileBrowser.List(root, AudioFileBrowser.DEFAULT_EXTENSIONS, maxPerGroup: 1);

            Assert.Multiple(() =>
            {
                Assert.That(listing.Truncated, Is.True, "truncation must be reported so the UI can say so");
                Assert.That(listing.Files, Has.Count.EqualTo(1));
                Assert.That(listing.Directories, Has.Count.EqualTo(1));
            });
        }

        [Test]
        public void TestMissingDirectoryIsReportedNotThrown()
        {
            var listing = AudioFileBrowser.List(Path.Combine(root, "does-not-exist"), AudioFileBrowser.DEFAULT_EXTENSIONS);

            Assert.Multiple(() =>
            {
                Assert.That(listing.Error, Is.Not.Null, "a broken path is data, not an exception out of the update thread");
                Assert.That(listing.Files, Is.Empty);
                Assert.That(listing.Directories, Is.Empty);
            });
        }

        [Test]
        public void TestStartDirectoryIsAlwaysUsable()
        {
            string resolved = AudioFileBrowser.ResolveStartDirectory(root);
            Assert.That(Directory.Exists(resolved), Is.True);

            // A preference that does not exist must be skipped rather than returned and then fail to list.
            resolved = AudioFileBrowser.ResolveStartDirectory(Path.Combine(root, "nope"));
            Assert.That(Directory.Exists(resolved), Is.True);
            Assert.That(resolved, Is.Not.EqualTo(Path.Combine(root, "nope")));
        }
    }
}
