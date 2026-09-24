#nullable enable

using System;
using System.Linq;
using osu.Framework.Allocation;
using osu.Framework.Audio;
using osu.Framework.Bindables;
using osu.Framework.Extensions.Color4Extensions;
using osu.Framework.Graphics;
using osu.Framework.Graphics.Containers;
using osu.Framework.Graphics.Shapes;
using osu.Framework.Graphics.Sprites;
using osu.Framework.Input.Events;
using osuTK;
using osuTK.Graphics;
using ParaTactus;

namespace OsuTest.Game.Graphics
{
    /// <summary>
    /// Lets the user choose the audio output device from inside the game.
    /// </summary>
    /// <remarks>
    /// Needed because the alternative is editing <c>framework.ini</c> by hand: an unset or wrong
    /// <c>AudioDevice</c> leaves BASS initialised against no output device, which means files decode and analyse
    /// fine but cannot be played at all. That failure looks like a file problem from the outside, so the fix has
    /// to be reachable from the same screen that reports it.
    ///
    /// The list comes from <see cref="AudioManager.AudioDeviceNames"/>, and the convention documented there is
    /// that the entry named "Default" maps to an empty device string - not to a device literally called Default.
    /// Writing the selected name into <see cref="AudioManager.AudioDevice"/> is all that is required; the
    /// framework re-initialises BASS itself and publishes the result back through the same bindable.
    /// </remarks>
    public partial class AudioDeviceSelector : CompositeDrawable
    {
        [Resolved]
        private AudioManager audio { get; set; } = null!;

        private readonly Bindable<string?> selectedDevice = new Bindable<string?>();

        /// <summary>
        /// Raised after the framework has re-initialised audio for a newly selected device. Lets the caller
        /// retry whatever failed because there was no output device.
        /// </summary>
        public event Action? DeviceChanged;

        private FillFlowContainer devices = null!;
        private SpriteText activeText = null!;

        public AudioDeviceSelector()
        {
            AutoSizeAxes = Axes.Both;
        }

        [BackgroundDependencyLoader]
        private void load()
        {
            InternalChildren = new Drawable[]
            {
                new FillFlowContainer
                {
                    AutoSizeAxes = Axes.Both,
                    Direction = FillDirection.Vertical,
                    Spacing = new Vector2(0, 4),
                    Children = new Drawable[]
                    {
                        new SpriteText { Text = "Audio output device", Font = FontUsage.Default.With(size: 20) },
                        activeText = new SpriteText { Font = FontUsage.Default.With(size: 16), Colour = Color4.Gray },
                        devices = new FillFlowContainer
                        {
                            AutoSizeAxes = Axes.Both,
                            Direction = FillDirection.Vertical,
                            Spacing = new Vector2(0, 2),
                            Padding = new MarginPadding { Top = 4 },
                        },
                    },
                },
            };

            // Bound rather than assigned once, so the highlight follows the framework's own state - including
            // when it falls back because the requested device could not be opened.
            selectedDevice.BindTo(audio.AudioDevice);
            selectedDevice.BindValueChanged(_ =>
            {
                updateActiveText();
                DeviceChanged?.Invoke();
            }, true);

            rebuildDeviceButtons();
        }

        /// <summary>
        /// Rebuilds the device list. Worth calling when something has just failed, since a device may have been
        /// plugged in or removed since the screen was opened.
        /// </summary>
        public void Rebuild() => Schedule(rebuildDeviceButtons);

        private void rebuildDeviceButtons()
        {
            devices.Clear();

            var names = audio.AudioDeviceNames.ToArray();

            if (names.Length == 0)
            {
                devices.Add(new SpriteText
                {
                    Text = "No audio device detected (BASS may not have initialised)",
                    Font = FontUsage.Default.With(size: 16),
                    Colour = Color4.OrangeRed,
                });

                updateActiveText();
                return;
            }

            foreach (string name in names)
            {
                // "Default" is the framework's alias for "let BASS pick", which is the empty string. Selecting it
                // is how a machine whose AudioDevice was left blank gets sound back.
                string value = name == "Default" ? string.Empty : name;

                devices.Add(new DeviceButton(name, () =>
                {
                    try
                    {
                        audio.AudioDevice.Value = value;
                    }
                    catch (Exception e)
                    {
                        activeText.Text = $"Switching device failed: {e.GetType().Name}: {e.Message}";
                    }
                }));
            }

            updateActiveText();
        }

        private void updateActiveText()
        {
            string current = string.IsNullOrEmpty(selectedDevice.Value) ? "Default" : selectedDevice.Value!;
            activeText.Text = $"Current: {current}";
        }

        /// <summary>
        /// A flat row with a highlighted state, built from a container because this project does not reference
        /// osu.Game, which is where the styled controls live.
        /// </summary>
        private partial class DeviceButton : Container
        {
            private readonly Box background;
            private readonly SpriteText label;
            private readonly Action action;

            public DeviceButton(string name, Action action)
            {
                this.action = action;

                AutoSizeAxes = Axes.Both;
                Masking = true;
                CornerRadius = 3;
                Padding = new MarginPadding { Horizontal = 10, Vertical = 5 };

                Child = new Container
                {
                    AutoSizeAxes = Axes.Both,
                    Children = new Drawable[]
                    {
                        background = new Box
                        {
                            RelativeSizeAxes = Axes.Both,
                            Colour = Color4Extensions.FromHex(@"2b3350"),
                        },
                        label = new SpriteText
                        {
                            Text = name,
                            Font = FontUsage.Default.With(size: 16),
                        },
                    },
                };
            }

            protected override bool OnClick(ClickEvent e)
            {
                action();
                return true;
            }

            protected override bool OnHover(HoverEvent e)
            {
                background.FadeColour(Color4Extensions.FromHex(@"3d4870"), 80, Easing.Out);
                return true;
            }

            protected override void OnHoverLost(HoverLostEvent e)
            {
                background.FadeColour(Color4Extensions.FromHex(@"2b3350"), 120, Easing.Out);
                base.OnHoverLost(e);
            }
        }
    }
}
