#nullable enable

using System;
using System.IO;
using osu.Framework.Allocation;
using osu.Framework.Extensions.Color4Extensions;
using osu.Framework.Graphics;
using osu.Framework.Graphics.Containers;
using osu.Framework.Graphics.Shapes;
using osu.Framework.Graphics.Sprites;
using osu.Framework.Input.Events;
using osuTK;
using osuTK.Graphics;
using OsuTest.Game.Audio;
using ParaTactus;

namespace OsuTest.Game.Graphics
{
    /// <summary>
    /// An in-game audio file browser, built entirely from framework primitives.
    /// </summary>
    /// <remarks>
    /// Needed because the operating system file chooser is not always available:
    /// <c>GameHost.CreateSystemFileSelector</c> returns null when the platform has no implementation, and the
    /// framework build this project targets ships no native file dialog library at all - a scan of the shipped
    /// assemblies and runtime natives finds no such dependency. Depending on the system chooser alone therefore
    /// means the feature does not exist for some users, so the picker is implemented here instead.
    ///
    /// This is deliberately plain: list the directory, one row per entry, click to descend or to choose. No text
    /// entry, no search, no drag and drop. The goal is that choosing a file always works, not that browsing a
    /// music library with it is pleasant.
    /// </remarks>
    public partial class InGameFilePicker : CompositeDrawable
    {
        /// <summary>
        /// Caps how many entries are shown per group. A folder with tens of thousands of files would otherwise
        /// build that many drawables and stall the update thread.
        /// </summary>
        private const int max_entries = 300;

        private readonly string[] allowedExtensions;

        /// <summary>Invoked with the chosen file. Not invoked when the picker is cancelled.</summary>
        public event Action<FileInfo>? FileSelected;

        private DirectoryInfo currentDirectory;
        private FillFlowContainer entries = null!;
        private SpriteText pathText = null!;
        private SpriteText statusText = null!;

        public InGameFilePicker(string startDirectory, string[] allowedExtensions)
        {
            this.allowedExtensions = allowedExtensions;

            currentDirectory = new DirectoryInfo(AudioFileBrowser.ResolveStartDirectory(startDirectory));

            RelativeSizeAxes = Axes.Both;
            Depth = float.MinValue;
            Alpha = 0;
            AlwaysPresent = true;
        }

        [BackgroundDependencyLoader]
        private void load()
        {
            InternalChildren = new Drawable[]
            {
                // Scrim: the browser sits over an animated scene, so it needs an opaque backdrop.
                new Box
                {
                    RelativeSizeAxes = Axes.Both,
                    Colour = Color4.Black.Opacity(0.9f),
                },
                new FillFlowContainer
                {
                    Anchor = Anchor.TopLeft,
                    Origin = Anchor.TopLeft,
                    Margin = new MarginPadding { Left = 16, Top = 16 },
                    Direction = FillDirection.Vertical,
                    Spacing = new Vector2(0, 4),
                    Children = new Drawable[]
                    {
                        new SpriteText { Text = "选择一个音频文件", Font = FontUsage.Default.With(size: 24) },
                        pathText = new SpriteText { Font = FontUsage.Default.With(size: 18) },
                        statusText = new SpriteText { Font = FontUsage.Default.With(size: 16), Colour = Color4.Gray },
                    },
                },
                new Container
                {
                    RelativeSizeAxes = Axes.Both,
                    Padding = new MarginPadding { Top = 104, Bottom = 16, Horizontal = 16 },
                    Child = new BasicScrollContainer
                    {
                        RelativeSizeAxes = Axes.Both,
                        Child = entries = new FillFlowContainer
                        {
                            RelativeSizeAxes = Axes.X,
                            AutoSizeAxes = Axes.Y,
                            Direction = FillDirection.Vertical,
                            Spacing = new Vector2(0, 2),
                        },
                    },
                },
                new TextButton("取消", HidePicker)
                {
                    Anchor = Anchor.TopRight,
                    Origin = Anchor.TopRight,
                    Margin = new MarginPadding(16),
                },
            };

            refresh();
        }

        /// <summary>Shows the picker and re-reads the directory, so a file added meanwhile shows up.</summary>
        public void ShowPicker()
        {
            this.FadeIn(150, Easing.Out);
            refresh();
        }

        /// <summary>
        /// Hides the picker and takes it out of the scene.
        /// </summary>
        /// <remarks>
        /// Fading out was not enough, and the consequence was severe: a drawable that has been faded to zero alpha is
        /// still in the scene tree and is still hit tested, so the whole screen-wide browser stayed underneath the
        /// pointer while invisible. Every click on empty space landed on whichever file row was at that position, which
        /// is why choosing a track and then clicking anywhere kept loading a different one. Expiring it is what
        /// actually stops it receiving input; the fade is only there so it does not vanish in one frame.
        /// </remarks>
        public void HidePicker() => this.FadeOut(150, Easing.Out).OnComplete(_ => Expire());

        /// <summary>
        /// Rebuilds the list for the current directory.
        /// </summary>
        private void refresh()
        {
            entries.Clear();
            pathText.Text = currentDirectory.FullName;

            DirectoryInfo? parent = currentDirectory.Parent;

            if (parent != null)
                entries.Add(new TextButton(".. (上一级)", () => navigate(parent)));

            var listing = AudioFileBrowser.List(currentDirectory.FullName, allowedExtensions, max_entries);

            if (listing.Error != null)
            {
                statusText.Text = $"无法读取该目录：{listing.Error}";
                return;
            }

            statusText.Text = $"{listing.Directories.Count} 个文件夹，{listing.Files.Count} 个音频文件"
                              + (listing.Truncated ? $"（已按上限 {max_entries} 截断）" : string.Empty);

            foreach (DirectoryInfo directory in listing.Directories)
                entries.Add(new TextButton($"[目录] {directory.Name}", () => navigate(directory)));

            foreach (FileInfo file in listing.Files)
                entries.Add(new TextButton(file.Name, () => choose(file)) { Indent = 1 });
        }

        private void navigate(DirectoryInfo directory)
        {
            currentDirectory = directory;
            refresh();
        }

        private void choose(FileInfo file)
        {
            HidePicker();
            FileSelected?.Invoke(file);
        }

        /// <summary>
        /// A flat, full-width row. Built from a container rather than a themed button control because this
        /// project does not reference osu.Game, which is where those live.
        /// </summary>
        private partial class TextButton : Container
        {
            private readonly Box background;
            private readonly Action action;

            public TextButton(string label, Action action)
            {
                this.action = action;

                RelativeSizeAxes = Axes.X;
                Height = 30;
                Masking = true;
                CornerRadius = 3;
                Padding = new MarginPadding { Horizontal = 10 };

                InternalChildren = new Drawable[]
                {
                    background = new Box
                    {
                        RelativeSizeAxes = Axes.Both,
                        Colour = Color4.Transparent,
                    },
                    new SpriteText
                    {
                        Text = label,
                        Font = FontUsage.Default.With(size: 17),
                        Anchor = Anchor.CentreLeft,
                        Origin = Anchor.CentreLeft,
                    },
                };
            }

            private int indent;

            /// <summary>Visual nesting for files, which sit under the directory entries.</summary>
            public int Indent
            {
                get => indent;
                set
                {
                    indent = value;
                    Padding = new MarginPadding { Horizontal = 10 + value * 22 };
                }
            }

            protected override bool OnClick(ClickEvent e)
            {
                action();
                return true;
            }

            protected override bool OnHover(HoverEvent e)
            {
                background.FadeColour(Color4Extensions.FromHex(@"2b3350"), 80, Easing.Out);
                return true;
            }

            protected override void OnHoverLost(HoverLostEvent e)
            {
                background.FadeColour(Color4.Transparent, 120, Easing.Out);
                base.OnHoverLost(e);
            }
        }
    }
}
