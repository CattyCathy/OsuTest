#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParaTactus;

namespace OsuTest.Game.Audio
{
    /// <summary>
    /// Directory listing logic for the in-game audio picker, kept separate from the drawable that displays it.
    /// </summary>
    /// <remarks>
    /// Splitting this out is what makes the browse behaviour testable: filtering by extension, sorting,
    /// surviving an unlistable directory and choosing a sensible start folder are all decisions that can be
    /// exercised against real temporary directories, without a window.
    /// </remarks>
    public static class AudioFileBrowser
    {
        /// <summary>Extensions offered by the picker, in the order they appear in dialogs.</summary>
        public static readonly string[] DEFAULT_EXTENSIONS = { ".mp3", ".ogg", ".wav", ".m4a", ".aac", ".flac", ".opus" };

        /// <summary>
        /// One directory listing: what the picker shows, and what went wrong if anything.
        /// </summary>
        public readonly record struct Listing(
            IReadOnlyList<DirectoryInfo> Directories,
            IReadOnlyList<FileInfo> Files,
            string? Error,
            bool Truncated);

        /// <summary>
        /// Lists a directory, hiding hidden folders, filtering to the allowed extensions and sorting both
        /// groups by name.
        /// </summary>
        /// <param name="directory">The directory to list.</param>
        /// <param name="allowedExtensions">Extensions to include, compared case-insensitively.</param>
        /// <param name="maxPerGroup">
        /// Cap on entries returned per group. A directory with tens of thousands of files would otherwise build
        /// that many drawables at once.
        /// </param>
        public static Listing List(string directory, string[] allowedExtensions, int maxPerGroup = int.MaxValue)
        {
            var extensions = allowedExtensions.Select(e => e.ToLowerInvariant()).ToArray();

            try
            {
                var info = new DirectoryInfo(directory);

                var directories = info.GetDirectories()
                                      .Where(d => (d.Attributes & FileAttributes.Hidden) == 0)
                                      .OrderBy(d => d.Name, StringComparer.OrdinalIgnoreCase)
                                      .ToArray();

                var files = info.GetFiles()
                                .Where(f => extensions.Contains(f.Extension.ToLowerInvariant()))
                                .OrderBy(f => f.Name, StringComparer.OrdinalIgnoreCase)
                                .ToArray();

                bool truncated = directories.Length > maxPerGroup || files.Length > maxPerGroup;

                return new Listing(
                    directories.Take(maxPerGroup).ToArray(),
                    files.Take(maxPerGroup).ToArray(),
                    null,
                    truncated);
            }
            catch (Exception e) when (e is UnauthorizedAccessException or IOException or ArgumentException)
            {
                // Protected directories such as C:\Windows are simply not listable. Reporting it as data rather
                // than throwing keeps the failure inside the picker, where the user can navigate back out.
                return new Listing(Array.Empty<DirectoryInfo>(), Array.Empty<FileInfo>(), e.Message, false);
            }
        }

        /// <summary>
        /// Picks a directory to open first: the preferred one if usable, otherwise the music folder, the user's
        /// home, then the current drive root.
        /// </summary>
        public static string ResolveStartDirectory(string? preferred = null)
        {
            foreach (string? candidate in new[]
                     {
                         preferred,
                         Environment.GetFolderPath(Environment.SpecialFolder.MyMusic),
                         Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                     })
            {
                if (!string.IsNullOrEmpty(candidate) && Directory.Exists(candidate))
                    return candidate;
            }

            return Path.GetPathRoot(Environment.CurrentDirectory) ?? Path.DirectorySeparatorChar.ToString();
        }
    }
}
