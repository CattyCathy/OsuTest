#nullable enable

using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using osu.Framework.Allocation;
using osu.Framework.Audio;
using osu.Framework.Extensions.Color4Extensions;
using osu.Framework.Graphics;
using osu.Framework.Graphics.Audio;
using osu.Framework.Graphics.Colour;
using osu.Framework.Graphics.Containers;
using osu.Framework.Graphics.UserInterface;
using osu.Framework.Graphics.Shapes;
using osu.Framework.Graphics.Sprites;
using osu.Framework.Input.Events;
using osu.Framework.Platform;
using osu.Framework.Timing;
using osu.Framework.Utils;
using osuTK;
using osuTK.Graphics;
using ManagedBass;
using ParaTactus;
using OsuTest.Game.Audio;
using ParaTactus.Decoding;

namespace OsuTest.Game.Graphics
{
    /// <summary>
    /// Drives visuals from the analysed tempo of a track, and lets the track be swapped at runtime.
    /// </summary>
    /// <remarks>
    /// Three things are being demonstrated:
    /// <list type="bullet">
    /// <item><b>Two independent time axes.</b> Beat-driven motion (the ring pulse, the rotating box) fires once
    /// per beat and takes its durations from the analysed beat length, so it speeds up on its own when the
    /// tempo changes. Amplitude-driven motion (the bars, the glow) follows the live audio level every frame and
    /// knows nothing about beats.</item>
    /// <item><b>A swappable source.</b> Decoding and analysis are a pipeline that can be re-run against any file
    /// at runtime, which is the difference between a demo and something usable.</item>
    /// <item><b>Choosing that source.</b> The operating system chooser is used when the platform provides one,
    /// with an in-game browser as the fallback - see <see cref="InGameFilePicker"/> for why that fallback is
    /// not optional.</item>
    /// </list>
    /// The synced container is placed on the track's clock, so every transform inside it is expressed in music
    /// time: pausing the music freezes those animations and changing the playback rate scales them, with no code
    /// written for either case.
    /// </remarks>
    public partial class BeatSyncDemo : CompositeDrawable
    {
        private const int bar_count = 48;

        private static readonly string[] audio_extensions = AudioFileBrowser.DEFAULT_EXTENSIONS;

        private BeatSyncedContainer beatSync = null!;
        private Container ring = null!;
        private Container spinner = null!;
        private Box glow = null!;
        private Box[] bars = Array.Empty<Box>();
        private FillFlowContainer picker = null!;

        /// <summary>Scrubs the track, so a section can be reached without waiting for it to play.</summary>
        private SeekStrip seekStrip = null!;

        private SpriteText sourceInfo = null!;
        private SpriteText tempoInfo = null!;
        private SpriteText timeInfo = null!;
        private SpriteText status = null!;

        [Resolved]
        private AudioManager audio { get; set; } = null!;

        [Resolved]
        private GameHost host { get; set; } = null!;

        private DrawableTrack? track;

        /// <summary>
        /// Direct BASS playback, used when the framework's track store cannot open the file. Its position drives
        /// <see cref="manualClock"/>, which is what the visuals run on.
        /// </summary>
        private BassAudioSource? bassSource;

        /// <summary>Music clock used when playback is handled by BASS rather than by a framework track.</summary>
        private ManualClock? manualClock;

        private BeatTimeMap timeMap = new BeatTimeMap();

        /// <summary>The beats the animation follows, from the model. Null until a source has been analysed.</summary>
        private BeatGrid? beatGrid;

        /// <summary>Analyses tracks with the beat model, reusing a cached grid when there is one.</summary>
        private readonly BeatGridProvider beatGrids = new BeatGridProvider(defaultModelPath(), defaultCacheDirectory(), BassAudioDecoder.Default);

        /// <summary>Why playback is unavailable, when it is. Analysis and visuals do not depend on it.</summary>
        private string? playbackError;

        /// <summary>A self-check report being displayed, which the per-frame readout must not overwrite.</summary>
        private string? selfCheckReport;

        /// <summary>The unmodified grid of the current source, so offsets can be re-applied from scratch.</summary>

        /// <summary>Manual timing offset applied to the beat grid, in milliseconds.</summary>
        private double beatOffset;

        private SpriteText offsetInfo = null!;

        /// <summary>
        /// Incremented on every source change so a slow analysis of a previous source can be discarded when it
        /// finishes after a newer one has already been applied.
        /// </summary>
        private int loadGeneration;

        /// <summary>Cancels the analysis of a source that has been superseded by a newer selection.</summary>
        private CancellationTokenSource? loadCancellation;

        private InGameFilePicker? filePicker;
        private AudioDeviceSelector deviceSelector = null!;

        /// <summary>The source currently loaded, so it can be re-opened after a device change.</summary>
        private string? currentSourcePath;
        private string? currentSourceName;

        [BackgroundDependencyLoader]
        private void load()
        {
            InternalChildren = new Drawable[]
            {
                new Box
                {
                    RelativeSizeAxes = Axes.Both,
                    Colour = ColourInfo.GradientVertical(Color4Extensions.FromHex(@"151a2d"), Color4Extensions.FromHex(@"08080f")),
                },
                createBars(),
                beatSync = new BeatSyncedContainer
                {
                    Anchor = Anchor.Centre,
                    Origin = Anchor.Centre,
                    AutoSizeAxes = Axes.Both,
                    // Created empty on purpose: a FramedClock with no source exposes the wall clock, so the
                    // visuals keep animating before a source is chosen. The music is attached as the source once
                    // the analysis is done.
                    Clock = new FramedClock(),
                    Children = new Drawable[]
                    {
                        ring = new Container
                        {
                            Anchor = Anchor.Centre,
                            Origin = Anchor.Centre,
                            Size = new Vector2(220),
                            Child = new CircularContainer
                            {
                                RelativeSizeAxes = Axes.Both,
                                Masking = true,
                                BorderThickness = 6,
                                BorderColour = Color4.White,
                                Children = new Drawable[]
                                {
                                    glow = new Box { RelativeSizeAxes = Axes.Both, Alpha = 0.08f },
                                },
                            },
                        },
                        spinner = new Container
                        {
                            Anchor = Anchor.Centre,
                            Origin = Anchor.Centre,
                            Size = new Vector2(90),
                            Child = new Box { RelativeSizeAxes = Axes.Both, Colour = Color4Extensions.FromHex(@"ff66ab") },
                        },
                    },
                },
                new FillFlowContainer
                {
                    Anchor = Anchor.TopLeft,
                    Origin = Anchor.TopLeft,
                    Margin = new MarginPadding(16),
                    Direction = FillDirection.Vertical,
                    Spacing = new Vector2(0, 4),
                    Width = 620,
                    Children = new Drawable[]
                    {
                        new SpriteText { Text = "BPM analysis -> animation", Font = FontUsage.Default.With(size: 26) },
                        sourceInfo = new SpriteText { Text = string.Empty, Font = FontUsage.Default.With(size: 18) },
                        tempoInfo = new SpriteText { Text = string.Empty, Font = FontUsage.Default.With(size: 18) },
                        timeInfo = new SpriteText { Text = string.Empty, Font = FontUsage.Default.With(size: 18) },
                        offsetInfo = new SpriteText { Text = "Time offset 0ms", Font = FontUsage.Default.With(size: 18) },
                        status = new SpriteText { Text = string.Empty, Font = FontUsage.Default.With(size: 18) },
                    },
                },
                picker = new FillFlowContainer
                {
                    Anchor = Anchor.TopRight,
                    Origin = Anchor.TopRight,
                    Margin = new MarginPadding(16),
                    Direction = FillDirection.Vertical,
                    Spacing = new Vector2(0, 6),
                    AutoSizeAxes = Axes.Both,
                    Children = new Drawable[]
                    {
                        // Placed with the source controls rather than hidden in a settings screen: when playback
                        // fails, the device is the thing to fix, and it has to be fixable from where the failure
                        // is reported.
                        deviceSelector = new AudioDeviceSelector(),

                        // A scrubber, because the failures being chased are in specific sections of specific tracks.
                        seekStrip = new SeekStrip
                        {
                            Width = 300,
                            Height = 18,
                            SeekRequested = fraction =>
                            {
                                double total = bassSource?.Length ?? track?.Length ?? 0;

                                if (total <= 0)
                                    return;

                                // Whether playback had already run off the end, decided before moving the position. The
                                // source does not report that itself: when the pump reaches the end of the file BASS
                                // stops the stream but the running flag stays set, so "finished" and "playing" look
                                // identical from outside, and a seek after the end moved the position without anything
                                // playing - the visuals and the spectrum followed along in silence.
                                double here = bassSource?.CurrentTime ?? track?.CurrentTime ?? 0;
                                bool finished = here >= total - 250;

                                double target = fraction * total;

                                // The audible source first: it is the one that decides what is heard next.
                                bassSource?.Seek(target);
                                track?.Seek(target);

                                // Only restarted in that case, so a player stopped part way through stays stopped when it
                                // is scrubbed rather than springing back to life under the pointer.
                                if (finished)
                                {
                                    bassSource?.Start();
                                    track?.Start();
                                }

                                seekStrip.Hold();
                            },
                        },
                    },
                },
            };

            beatSync.OnBeat += onBeat;

            // The framework re-initialises BASS itself when the device changes, which takes a moment. Pushing the
            // reaction to the next scheduled frame lets that settle before anything is retried against it, and
            // also refreshes the device list in case it changed.
            deviceSelector.DeviceChanged += () => Schedule(() =>
            {
                deviceSelector.Rebuild();

                if (playbackError != null)
                    reloadCurrentSource();
            });

            addPickerButton("Load built-in demo (120 -> 160 BPM)", loadGeneratedDemo);
            addPickerButton("Choose audio file...", presentFileSelector);
            addPickerButton("Stop playback", stopPlayback);
            addPickerButton("Resume playback", startPlayback);
            addPickerButton("Playback rate 1.5x", () => setRate(1.5));
            addPickerButton("Playback rate 0.75x", () => setRate(0.75));
            addPickerButton("Reload current source", reloadCurrentSource);
            addPickerButton("Rescan audio devices", () => deviceSelector.Rebuild());
            addPickerButton("Audio self-check", runAudioSelfCheck);
            addPickerButton("Show tempo timeline", showTempoTimeline);
            addPickerButton("Export tempo diagnostics", exportTempoDiagnostics);
            addPickerButton("Offset -10ms", () => adjustOffset(-10));
            addPickerButton("Offset +10ms", () => adjustOffset(10));
            addPickerButton("Reset offset", () => adjustOffset(-beatOffset, absolute: true));

            loadGeneratedDemo();
        }

        /// <summary>
        /// Shows every point where the reported tempo changes, so a wrong reading can be located in time.
        /// </summary>
        /// <remarks>
        /// The readout already shows the tempo at the current position, which is not enough to diagnose a complaint
        /// about a change appearing in the wrong place: what is needed is the sequence of changes and the confidence
        /// behind each one. Confidence matters because a change backed by a weak estimate in a quiet passage looks
        /// exactly like a real change everywhere else.
        /// </remarks>
        private void showTempoTimeline()
        {
            if (beatGrid == null)
            {
                selfCheckReport = "No source loaded yet";
                status.Text = selfCheckReport;
                Scheduler.AddDelayed(() => selfCheckReport = null, 8000);
                return;
            }

            var report = new System.Text.StringBuilder();
            report.Append($"{beatGrid.Beats.Count} beats over {beatGrid.Duration / 1000:0.#}s "
                          + $"metrical level {beatGrid.MetricalShift:+#;-#;0}");

            // Runs of consecutive beats at the same tempo, which is what the per-keyframe list used to show. There is no
            // confidence column any more: the beats are the model's answer rather than an estimate of an onset envelope,
            // so there is nothing to be unsure about.
            int from = 0;

            for (int i = 1; i <= beatGrid.Beats.Count; i++)
            {
                bool changed = i == beatGrid.Beats.Count
                               || Math.Abs(beatGrid.BpmAt(beatGrid.Beats[i]) - beatGrid.BpmAt(beatGrid.Beats[from])) > 1;

                if (!changed)
                    continue;

                double until = i < beatGrid.Beats.Count ? beatGrid.Beats[i] : beatGrid.Duration;

                report.Append($"\n{beatGrid.Beats[from] / 1000:0.00}s - {until / 1000:0.00}s"
                              + $"  {beatGrid.BpmAt(beatGrid.Beats[from]):0.#} BPM  ({i - from} ??"
                              + (until - beatGrid.Beats[from] < 2000 ? "  <- very close" : string.Empty));

                from = i;
            }

            selfCheckReport = report.ToString();
            status.Text = selfCheckReport;
            Scheduler.AddDelayed(() => selfCheckReport = null, 30_000);
        }

        /// <summary>
        /// Writes the raw per-hop tempo estimates to a file, for diagnosing an unstable region.
        /// </summary>
        /// <remarks>
        /// The keyframe list and the on-screen readout both show the result after stabilisation, which is exactly what
        /// is not needed when asking why stability was lost in a particular stretch. This dumps the estimates as they
        /// leave the estimator, so an unstable region can be examined window by window rather than guessed at.
        ///
        /// Written to the temp directory, and the path is put on screen so it can be found.
        /// </remarks>
        private void exportTempoDiagnostics()
        {
            if (beatGrid == null)
            {
                status.Text = "No source loaded yet";
                return;
            }

            string name = currentSourceName ?? (currentSourcePath != null ? Path.GetFileName(currentSourcePath) : "<unknown>");
            string target = Path.Combine(Path.GetTempPath(), $"osutest-beats-{DateTime.Now:yyyyMMdd-HHmmss}.csv");

            status.Text = "Exporting diagnostics...";

            BeatGrid grid = beatGrid;

            Task.Run(() =>
            {
                try
                {
                    // One row per beat rather than per analysis hop: the model reports beats, so the interval between
                    // consecutive rows is the tempo it heard at that point.
                    using (var writer = new StreamWriter(target))
                    {
                        writer.WriteLine($"# source\t{name}");
                        writer.WriteLine($"# file\t{currentSourcePath}");
                        writer.WriteLine($"# metricalShift\t{grid.MetricalShift}");
                        writer.WriteLine("index\ttime_ms\tbeat_length_ms\tbpm\tphase");

                        for (int i = 0; i < grid.Beats.Count; i++)
                        {
                            double length = grid.BeatLengthAt(i);

                            writer.WriteLine($"{i}\t{grid.Beats[i]:0.###}\t{length:0.###}\t"
                                             + $"{(length > 0 ? 60000 / length : 0):0.###}\t{grid.PhaseAt(grid.Beats[i]):0.####}");
                        }
                    }

                    Schedule(() =>
                    {
                        selfCheckReport = $"Export failed: {grid.Beats.Count} beats\n{target}";
                        status.Text = selfCheckReport;
                        Scheduler.AddDelayed(() => selfCheckReport = null, 30_000);
                    });
                }
                catch (Exception e)
                {
                    Schedule(() => status.Text = $"Diagnostics failed: {e.GetType().Name}: {e.Message}");
                }
            });
        }

        /// <summary>
        /// Applies a manual timing offset to the beat grid.
        /// </summary>
        /// <remarks>
        /// Output latency cannot be measured from the audio - it depends on the buffer, the driver and any
        /// wireless headphones in the chain - so it has to be adjustable by ear. The generated demo track is
        /// aligned to the grid by construction; real music may not be, since an analysis can only place beats
        /// where the onsets are.
        /// </remarks>
        /// <summary>
        /// Turns a grid into the tempo map the on-screen readouts were written against.
        /// </summary>
        /// <remarks>
        /// One keyframe per beat, so the map says the same thing the grid does rather than approximating it with a
        /// handful of sections. It is a view for display only: nothing that drives an animation reads it.
        /// </remarks>
        private static BeatTimeMap buildTimeMap(BeatGrid grid)
        {
            var map = new BeatTimeMap();

            for (int i = 0; i < grid.Beats.Count; i++)
            {
                double length = grid.BeatLengthAt(i);

                if (length > 0)
                    map.Add(grid.Beats[i], 60000 / length);
            }

            return map;
        }

        /// <summary>
        /// Where the beat model lives: <c>OSUTEST_MODEL</c> if it is set and present, otherwise the quantised export
        /// and then the full one, both of which are looked for in the user's downloads.
        /// </summary>
        private static string defaultModelPath()
        {
            string? configured = Environment.GetEnvironmentVariable("OSUTEST_MODEL");

            if (!string.IsNullOrEmpty(configured))
            {
                if (File.Exists(configured))
                    return configured;

                throw new FileNotFoundException($"OSUTEST_MODEL points at {configured}, and there is no file there.");
            }

            string downloads = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads");

            string quantised = Path.Combine(downloads, "beat-this-final0-int8.onnx");

            // Measured on Designant, the quantised model is not the same beats as the float one: 550 regularised beats
            // against 556, and about a tenth of them land more than 20ms apart. It is not worse where it counts,
            // though - the median residual to the beatmap's own grid is 29ms for both, p90 131ms against 141ms, and
            // 35.3% of beats more than 60ms out against 36.0%. A third of the size for that is the trade, so it stays
            // preferred. The numbers are in docs/model.md in the library repository.
            if (File.Exists(quantised))
                return quantised;

            string floatBuild = Path.Combine(downloads, "beat-this-final0.onnx");

            if (File.Exists(floatBuild))
                return floatBuild;

            // Named here rather than left to ONNX Runtime, which reports a missing model as a failed session open and
            // says nothing about where the file was expected or how to get one.
            throw new FileNotFoundException(
                $"No ONNX model found. Looked for {quantised} and {floatBuild}. Download either from "
                + "https://github.com/CattyCathy/beat-this-onnx/releases, or set OSUTEST_MODEL to its path.");
        }

        private static string defaultCacheDirectory()
        {
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OsuTest", "beatgrids");
        }

        private void adjustOffset(double delta, bool absolute = false)
        {
            if (beatGrid == null)
            {
                status.Text = "No source loaded yet";
                return;
            }

            beatOffset = absolute ? delta : beatOffset + delta;

            beatGrid = beatGrid.WithOffset(beatOffset);
            beatSync.Grid = beatGrid;
            timeMap = buildTimeMap(beatGrid);

            offsetInfo.Text = $"Time offset: {beatOffset:+0;-0;0}ms";
            status.Text = $"time offset {beatOffset:+0;-0;0}ms"
                          + (beatOffset == 0 ? ", none applied" : ", applied");
        }

        /// <summary>
        /// Reports what the audio stack looks like from inside the running game, on screen.
        /// </summary>
        /// <remarks>
        /// The same information a log would carry, but reachable without leaving the game or reading a file. It
        /// answers the questions that distinguish the failure modes: is the framework's device list populated, can
        /// a silent track be created, can a real track be opened, and what exactly happens when it cannot.
        /// </remarks>
        private void runAudioSelfCheck()
        {
            var report = new System.Text.StringBuilder();

            var names = audio.AudioDeviceNames.ToArray();
            report.Append($"Devices ({names.Length}): {(names.Length == 0 ? "<empty>" : string.Join(" | ", names))}");

            string current = string.IsNullOrEmpty(audio.AudioDevice.Value) ? "Default(??" : audio.AudioDevice.Value!;
            report.Append($"\nCurrent device: {current}");

            try
            {
                var silent = audio.Tracks.GetVirtual(5000);
                report.Append($"\nSilent track: {(silent == null ? "null" : $"ok, length={silent.Length:0}ms")}");
            }
            catch (Exception e)
            {
                report.Append($"\nFramework track failed: {e.GetType().Name}: {e.Message}");
            }

            if (currentSourcePath != null)
            {
                try
                {
                    var probe = audio.Tracks.Get(currentSourcePath);

                    // Null here is the expected outcome for a user-picked file rather than a fault: the track store
                    // resolves names through the stores attached to it, not through the file system, and the store
                    // that could be given a directory is internal to the framework. The BASS path below covers it.
                    report.Append($"\nFramework track store: {(probe == null ? "null (expected for a file the user chose: the direct BASS path handles it)" : $"ok, {probe.GetType().Name}, length={probe.Length:0}ms")}");
                }
                catch (Exception e)
                {
                    report.Append($"\nFramework track failed: {e.GetType().Name}: {e.Message}");
                }
            }

            report.Append($"\nDirect BASS playback: {(bassSource == null ? "not used" : $"in use, position {bassSource.CurrentTime:0}ms / {bassSource.Length:0}ms, running={bassSource.IsRunning}")}");

            if (beatGrid != null)
            {
                // What the old alignment measurement said in one line, now that the beats are the model's answer
                // rather than an estimate: the spread of the intervals is the honest measure of how steady the
                // reading is, and the level says which pulse it settled on.
                var intervals = new System.Collections.Generic.List<double>();

                for (int i = 0; i < beatGrid.Beats.Count; i++)
                {
                    double length = beatGrid.BeatLengthAt(i);

                    if (length > 0)
                        intervals.Add(length);
                }

                if (intervals.Count > 0)
                {
                    intervals.Sort();

                    double median = intervals[intervals.Count / 2];
                    double spread = intervals[intervals.Count - 1] - intervals[0];

                    report.Append($"\nAverage: {beatGrid.Beats.Count} beats, median {median:0.#}ms ({60000 / median:0.#} BPM), "
                                  + $"spread {spread:0.#}ms, metrical level {beatGrid.MetricalShift:+#;-#;0}");
                }
            }

            report.Append($"\nLast playback failure: {playbackError ?? "<none>"}");

            status.Text = report.ToString();
            deviceSelector.Rebuild();

            // The per-frame readout would otherwise overwrite this immediately.
            selfCheckReport = report.ToString();
            Scheduler.AddDelayed(() => selfCheckReport = null, 15000);
        }

        // ---- source management ----

        /// <summary>
        /// Re-analyses and re-opens the current source, picking up a device change or a file replaced on disk.
        /// </summary>
        private void reloadCurrentSource()
        {
            if (currentSourcePath == null)
            {
                status.Text = "No source loaded yet";
                return;
            }

            if (!File.Exists(currentSourcePath))
            {
                status.Text = $"Source file does not exist: {currentSourcePath}";
                return;
            }

            status.Text = "Reloading the current source...";
            loadSource(currentSourcePath, currentSourceName ?? Path.GetFileName(currentSourcePath));
        }

        /// <summary>
        /// Generates the built-in tempo-changing fixture and loads it, creating the WAV only when missing.
        /// </summary>
        private void loadGeneratedDemo()
        {
            string path = DemoTrackGenerator.GetDefaultPath();
            var map = DemoTrackGenerator.CreateTempoMap();

            if (!File.Exists(path))
                DemoTrackGenerator.WriteTo(path);

            loadSource(path, $"Built-in demo ({map.Segments[0].Bpm:0.#} -> {map.Segments[^1].Bpm:0.#} BPM)");
        }

        /// <summary>
        /// Loads an audio file chosen by the user, preferring the operating system's chooser and falling back to
        /// the in-game browser when the platform has no implementation of it.
        /// </summary>
        private void presentFileSelector()
        {
            try
            {
                var selector = host.CreateSystemFileSelector(audio_extensions);

                if (selector == null)
                {
                    // Not all platforms implement this, and the framework build this project targets ships no
                    // native file dialog library at all, so this is an expected path rather than an error.
                    presentInGamePicker("The system file picker is unavailable; using the in-game one");
                    return;
                }

                selector.Selected += file => Schedule(() => loadSource(file.FullName, file.Name));
                selector.Present();
            }
            catch (Exception e)
            {
                presentInGamePicker($"System file picker unavailable: {e.GetType().Name}; using the in-game picker");
            }
        }

        /// <summary>
        /// Shows the built-in browser, creating it on first use.
        /// </summary>
        private void presentInGamePicker(string? message)
        {
            if (message != null)
                status.Text = message;

            // A fresh browser every time, rather than one built once and faded in and out. The picker now removes
            // itself from the scene when it closes, so the previous one is expired rather than reusable - and reusing a
            // cached one is exactly what left an invisible full-screen browser receiving clicks after it had been used.
            filePicker?.Expire();

            filePicker = new InGameFilePicker(AudioFileBrowser.ResolveStartDirectory(), audio_extensions);
            filePicker.FileSelected += file => loadSource(file.FullName, file.Name);
            AddInternal(filePicker);

            filePicker.ShowPicker();
        }

        /// <summary>
        /// Decodes and analyses a file, then hands the result to the visuals. Runs the blocking work off the
        /// update thread and is safe to call again while a previous load is still in flight.
        /// </summary>
        private void loadSource(string path, string displayName)
        {
            int generation = ++loadGeneration;

            currentSourcePath = path;
            currentSourceName = displayName;

            // Stop the previous analysis outright rather than letting it finish and be thrown away: decoding a
            // full track is the most expensive thing this screen does, and the user may well click through
            // several files.
            loadCancellation?.Cancel();
            loadCancellation?.Dispose();
            loadCancellation = new CancellationTokenSource();
            var cancellation = loadCancellation.Token;

            sourceInfo.Text = $"Source: {displayName}";
            status.Text = "Decoding and analysing...";

            // Drop the current source and take the visuals off the music clock while the new one is prepared.
            // Without this the old track keeps playing underneath the visuals, and its clock keeps driving them.
            stopAndRemoveTrack();
            beatSync.TimeMap = null;
            timeMap = new BeatTimeMap();

            Schedule(prepareSource);

            void prepareSource()
            {
                string? playbackFailure = null;
                var loaded = tryLoadTrack(path, out playbackFailure);

                if (generation != loadGeneration)
                {
                    // A newer source was selected while this one was loading; discard this one.
                    loaded?.Dispose();
                    return;
                }

                // Playback failing is not fatal. Decoding and analysis need no output device, so the tempo map
                // is still derived from the real audio; only the sound is missing, and the analysis result's
                // duration is enough to drive a silent stand-in track on the wall clock.
                playbackError = loaded == null ? playbackFailure ?? "audio system unavailable" : null;

                if (loaded != null)
                {
                    track = loaded;

                    // The framework's track store hands back the same instance for the same file, so a second load of a
                    // track that is already on screen would re-parent a drawable that still has a parent. That is what
                    // "can not change depth of drawable which is not contained within this CompositeDrawable" means.
                    if (track.Parent == null)
                        AddInternal(track);

                    // Hold the track at zero until the tempo map is ready. Playing it immediately would let the
                    // music run ahead of the grid that is supposed to drive the visuals, which is exactly the
                    // desynchronisation this whole exercise is trying to avoid.
                    track.Stop();
                }

                Task.Run(() =>
                {
                    BeatGrid? grid = null;
                    string? error = null;

                    try
                    {
                        // The model is the beat source. The hand-tuned analyser this replaced decided a tempo three
                        // different ways for the same candidate set, and measured against a beatmap's own timing points
                        // it read 32 of 35 sampled points wrong; the model gets every steady section of the same track
                        // to a metrical level of the truth.
                        grid = beatGrids.Get(path, cancellation);
                    }
                    catch (OperationCanceledException)
                    {
                        return;
                    }
                    catch (BassAudioDecoder.DecodeException e)
                    {
                        // The BASS error code is the only thing that says why a file failed, so it is passed
                        // through to the UI rather than replaced with a generic message.
                        error = $"Decode failed ({e.Error}): unsupported format or codec, or the audio system did not start";
                    }
                    catch (Exception e)
                    {
                        error = $"{e.GetType().Name}: {e.Message}";
                    }

                    Schedule(() =>
                    {
                        if (generation != loadGeneration)
                            return;

                        applyAnalysis(grid, error);
                    });
                });
            }
        }

        private void applyAnalysis(BeatGrid? grid, string? error)
        {
            if (grid == null)
            {
                status.Text = $"Load failed: {error ?? "unknown error"}";
                return;
            }

            if (grid.IsEmpty)
            {
                status.Text = "No audio device available; using a silent track";
                return;
            }

            beatGrid = grid;

            // The grid drives the animation; the time map is kept alongside only so the on-screen readouts that were
            // written against it keep working. The container prefers the grid, so the two cannot disagree about the
            // beat the visuals are on.
            beatSync.Grid = grid;
            timeMap = buildTimeMap(grid);
            beatSync.TimeMap = null;

            // A new source starts from no offset: the previous one's correction belonged to the previous file.
            beatOffset = 0;
            offsetInfo.Text = "Time offset 0ms";

            double slowest = double.MaxValue;
            double fastest = 0;

            for (int i = 0; i < grid.Beats.Count; i++)
            {
                double length = grid.BeatLengthAt(i);

                if (length <= 0)
                    continue;

                slowest = Math.Min(slowest, 60000 / length);
                fastest = Math.Max(fastest, 60000 / length);
            }

            tempoInfo.Text = $"The model found {grid.Beats.Count} beats, {slowest:0.#}-{fastest:0.#} BPM"
                             + $", metrical level {grid.MetricalShift:+#;-#;0}";

            // Playback, in order of preference: the framework's own track, then a BASS channel opened directly,
            // then silent. The middle case exists because the framework's track store cannot open files outside
            // its own resources, which is exactly what a user-picked file is - so returning null there is the
            // expected outcome, not a failure worth reporting unless the fallback fails too.
            if (track == null && bassSource == null)
            {
                bassSource = BassAudioSource.TryOpen(currentSourcePath!, out string? bassFailure);

                if (bassSource == null)
                {
                    var silent = createSilentTrack(grid.Duration);

                    if (silent == null)
                    {
                        status.Text = $"Cannot play ({grid.Beats.Count} beats), and BASS could not decode the file either: {bassFailure}";
                        return;
                    }

                    track = silent;

                    if (track.Parent == null)
                        AddInternal(track);

                    // Only the fallback's own reason is reported. The framework's store failing to find a
                    // user-picked file is expected and would only be noise here.
                    playbackError = $"BASS could not decode the file: {bassFailure}";
                }
            }

            bool usingBass = bassSource != null;

            if (usingBass)
            {
                // The BASS channel is the master clock here. Its position is copied into a manual clock each
                // frame, and the visuals hang off that clock - the same separation as a real track clock, without
                // needing a framework track object.
                manualClock = new ManualClock { CurrentTime = bassSource!.CurrentTime, Rate = 1 };
                beatSync.Clock = new FramedClock(manualClock);
                bassSource.Start();
            }
            else
            {
                beatSync.Clock = new FramedClock(track!);
                track!.Start();
            }

            status.Text = playbackError == null
                ? $"Ready: {grid.Beats.Count} beats / {grid.Duration / 1000:0.#}s"
                  + (usingBass ? ", played directly through BASS" : string.Empty)
                : $"Preview mode: {playbackError}";
        }

        /// <summary>
        /// A silent stand-in with the right duration, used when the audio system cannot open a playback stream.
        /// </summary>
        /// <remarks>
        /// This is what makes the no-output-device case survivable. Decoding and tempo analysis never needed an
        /// output device, so the only thing actually lost is the sound; substituting a clock-only track keeps
        /// every other part of the pipeline - the grid, the beat events, the music-time transforms - intact.
        /// </remarks>
        private DrawableTrack? createSilentTrack(double durationMs)
        {
            try
            {
                var silent = audio.Tracks.GetVirtual(durationMs);
                return silent == null ? null : new DrawableTrack(silent);
            }
            catch (Exception)
            {
                return null;
            }
        }

        private DrawableTrack? tryLoadTrack(string path, out string? failure)
        {
            failure = null;

            try
            {
                var loaded = audio.Tracks.Get(path);

                if (loaded == null)
                {
                    // Expected for files outside the game's own resources: the track store resolves names through
                    // the stores attached to it rather than through the file system, and the concrete store that
                    // could be given a directory is internal. The BASS fallback covers this case.
                    failure = "framework track store returned null for this path";
                    return null;
                }

                return new DrawableTrack(loaded);
            }
            catch (Exception e)
            {
                failure = $"{e.GetType().Name}: {e.Message}";
                return null;
            }
        }

        /// <summary>
        /// Names the audio device the framework is running against, which is the usual culprit when playback
        /// fails while decoding works.
        /// </summary>
        private static string describeAudioDevice()
        {
            try
            {
                string? configured = null;

                try
                {
                    configured = Bass.CurrentDevice >= 0 ? Bass.GetDeviceInfo(Bass.CurrentDevice).Name : null;
                }
                catch (Exception)
                {
                    // Querying the device is best effort; a failure here must not replace the real error.
                }

                return configured ?? "unknown";
            }
            catch (Exception)
            {
                return "unavailable";
            }
        }

        private void stopAndRemoveTrack()
        {
            if (track != null)
            {
                track.Stop();
                RemoveInternal(track, true);
                track = null;
            }

            // The BASS fallback owns an unmanaged channel, so it has to be released explicitly.
            bassSource?.Dispose();
            bassSource = null;
            manualClock = null;
        }

        /// <summary>Whether anything is currently able to play, and on which backend.</summary>
        private bool hasPlayer => track != null || bassSource != null;

        private void startPlayback()
        {
            if (bassSource != null)
                bassSource.Start();
            else
                track?.Start();
        }

        private void stopPlayback()
        {
            if (bassSource != null)
                bassSource.Stop();
            else
                track?.Stop();
        }

        private void setRate(double rate)
        {
            if (!hasPlayer)
            {
                status.Text = "No source loaded yet";
                return;
            }

            // Changing the playback rate changes the clock the visuals run on, so beat-driven animations speed up
            // with the music without anything here knowing about them.
            if (bassSource != null)
                bassSource.Rate = rate;
            else
                track!.Frequency.Value = rate;

            status.Text = $"Playback rate {rate:0.##}x (playback only, the timeline is unaffected)";
        }

        protected override void Dispose(bool isDisposing)
        {
            base.Dispose(isDisposing);

            if (!isDisposing)
                return;

            // Invalidate any analysis still in flight so its completion callback does not touch disposed state.
            loadGeneration++;
            loadCancellation?.Cancel();
            loadCancellation?.Dispose();
            loadCancellation = null;

            // Release the BASS channel, which is unmanaged.
            bassSource?.Dispose();
            bassSource = null;
        }

        // ---- layout helpers ----

        private void addPickerButton(string label, Action action)
        {
            picker.Add(new PickerButton(label, action)
            {
                Anchor = Anchor.TopRight,
                Origin = Anchor.TopRight,
            });
        }

        private Drawable createBars()
        {
            bars = new Box[bar_count];

            var flow = new FillFlowContainer
            {
                Anchor = Anchor.BottomCentre,
                Origin = Anchor.BottomCentre,
                Width = 900,
                Height = 160,
                Margin = new MarginPadding { Bottom = 40 },
                Direction = FillDirection.Horizontal,
                Spacing = new Vector2(4, 0),
            };

            for (int i = 0; i < bar_count; i++)
            {
                bars[i] = new Box
                {
                    Width = 14,
                    Height = 4,
                    Anchor = Anchor.BottomCentre,
                    Origin = Anchor.BottomCentre,
                    Colour = ColourInfo.GradientVertical(Color4Extensions.FromHex(@"66d9ff"), Color4Extensions.FromHex(@"ff66ab")),
                };

                flow.Add(bars[i]);
            }

            return flow;
        }

        // ---- per-frame updates ----

        protected override void Update()
        {
            base.Update();

            updateBars();
            updateReadout();

            // The BASS source is the one making sound whenever it is open, so it is the one the scrubber follows and
            // the one it seeks. The framework track is preferred on the other path and is a silent placeholder here,
            // and it reports its own position - which is how the bar snapped back to the start as soon as the mouse
            // came up: the seek went to BASS and the bar was reading the placeholder.
            //
            // The hold covers the frames between asking for a seek and the new position being reported, so the bar does
            // not lurch back to where it was while the request is still in flight.
            double total = bassSource?.Length ?? track?.Length ?? 0;

            if (total > 0)
                seekStrip.ShowPlayhead((bassSource?.CurrentTime ?? track?.CurrentTime ?? 0) / total);
        }

        /// <summary>
        /// Amplitude-driven motion: per-frame interpolation, no beat involvement.
        /// </summary>
        private void updateBars()
        {
            // Amplitudes come from the framework track when there is one. The BASS path has no amplitude data of
            // its own, so it is asked for a spectrum computed from the decoded audio at the current position -
            // that is what makes the bars move on that path too.
            float[] amplitudes = track != null
                ? track.CurrentAmplitudes.FrequencyAmplitudes.ToArray()
                : bassSource?.GetFrequencyAmplitudes().ToArray() ?? Array.Empty<float>();

            // Mirrored about the centre, which is how most players draw a spectrum: the bands run outwards from the
            // middle, so the number of physical bars is twice the number of bands and each band drives two of them.
            int bands = Math.Max(1, bar_count / 2);
            var ranges = new (int First, int Last)[bands];

            if (amplitudes.Length > 0)
            {
                // Worked out in hertz, not in bin numbers. The two amplitude sources have different bin widths and
                // neither reports its sample rate, so the width is taken from what each is known to produce:
                // osu.Framework's 256 bins span 0-20kHz and this project's 512 span 0-11kHz. Spacing by bin index put
                // the same number of bars on 20kHz as on 11kHz, which spends a third of the display on a range that
                // holds nothing.
                double binHz = amplitudes.Length == 256 ? 20000.0 / 256 : 11025.0 / amplitudes.Length;

                double ceiling = Math.Min(SpectrumBands.TopHz, amplitudes.Length * binHz);
                double topHz = SpectrumBands.AudibleTop(amplitudes, binHz, ceiling);

                ranges = SpectrumBands.Ranges(amplitudes.Length, bands, binHz, topHz);
            }

            for (int i = 0; i < bar_count; i++)
            {
                double level = 0;

                // Guarded because with no amplitudes there is no bin to read and the loop below would have nothing to
                // read from. An empty spectrum draws as silence rather than throwing.
                if (amplitudes.Length > 0)
                {
                    // Distance from the centre, so bar 0 and the last bar share the top band and the two middle bars
                    // share the bottom one.
                    int band = Math.Clamp((int)(Math.Abs(i - (bands - 0.5)) - 0.5), 0, bands - 1);
                    var (first, last) = ranges[band];

                    for (int j = first; j < last; j++)
                        level = Math.Max(level, amplitudes[j]);
                }

                // Logarithmic, not linear. Raw FFT magnitudes of music span several orders of magnitude - quiet
                // bins sit near zero while a bass hit reaches tens - so a linear scale pins every bar to the top
                // and shows nothing about the actual spectrum shape. A log scale is also how loudness is
                // perceived, so the bars track what is heard rather than what is measured.
                double target = 4 + 150 * barLevel(level, bar_gain);

                // Damped against the wall clock rather than the music clock: the decay should continue while the
                // music is paused, and a frozen music clock would stop it halfway.
                bars[i].Height = (float)Interpolation.Damp(bars[i].Height, target, 0.85f, Clock.ElapsedFrameTime);
            }

            float overall = amplitudes.Length > 0 ? amplitudes.Max() : 0;
            glow.Alpha = (float)Interpolation.Damp(glow.Alpha, 0.06f + barLevel(overall, bar_gain) * 0.25f, 0.9f, Clock.ElapsedFrameTime);
        }

        /// <summary>Gain applied before the logarithmic compression, chosen so a typical spectrum fills the bars.</summary>
        private const double bar_gain = 8;

        /// <summary>
        /// Maps a raw FFT magnitude onto 0..1 for display.
        /// </summary>
        /// <remarks>
        /// The scale has to be logarithmic and has to have a reference point. Linear leaves every bar pinned at the
        /// maximum, which is what was reported; a bare logarithm is unbounded, so it needs a fixed magnitude to
        /// treat as "full", and one magnitude per frequency bin so that quiet high-frequency content is still
        /// visible next to a loud bassline.
        /// </remarks>
        private static double barLevel(double magnitude, double gain)
        {
            if (magnitude <= 0)
                return 0;

            const double full_scale = 20;

            return Math.Clamp(Math.Log10(1 + magnitude * gain) / Math.Log10(1 + full_scale * gain), 0, 1);
        }

        private void updateReadout()
        {
            if (!hasPlayer)
                return;

            // The BASS path keeps the manual clock in step with the audio, so the beat container sees the same
            // advancing music time a real track clock would provide.
            double now;

            if (bassSource != null)
            {
                now = bassSource.CurrentTime;

                if (manualClock != null)
                {
                    manualClock.CurrentTime = now;
                    manualClock.Rate = bassSource.Rate;
                }
            }
            else
            {
                now = track!.CurrentTime;
            }

            // The live readout is the practical proof that tempo is tracked over time: crossing a tempo boundary
            // changes the queried beat length, because the map is queried by time.
            timeInfo.Text = $"t = {now / 1000:0.00}s   bpm(map) = {timeMap.BpmAt(now):0.#}   beatLength = {timeMap.BeatLengthAt(now):0.#}ms";

            if (selfCheckReport != null)
                return;

            if (beatSync.AnalysedBeatLength is double beatLength)
                status.Text = $"beat {beatSync.CurrentBeatIndex}   next beat in {beatSync.TimeUntilNextBeat:0}ms   animation beat length {beatLength:0.#}ms";
        }

        /// <summary>
        /// Beat-driven motion. Called once per beat, with the animation start time already backdated to the beat
        /// itself by the container.
        /// </summary>
        private void onBeat(int beatIndex, double beatLength)
        {
            if (beatIndex < 0)
                return;

            // Compress then release, the same shape osu!'s main menu logo uses - except the release duration is
            // the analysed beat length rather than a constant, so it tightens up when the track speeds up.
            ring.ScaleTo(0.9f, beatSync.EarlyActivationMilliseconds, Easing.Out).Then()
                .ScaleTo(1, beatLength * 1.5, Easing.OutQuint);

            // A quarter turn per beat: rotation rate is directly proportional to tempo.
            spinner.RotateTo(spinner.Rotation + 90, beatLength * 2, Easing.InOutSine);
        }

        /// <summary>
        /// A minimal button. Built from a container rather than a themed button control because this project does
        /// not reference osu.Game, which is where the styled buttons live.
        /// </summary>
        private partial class PickerButton : Container
        {
            private readonly Box background;

            public PickerButton(string label, Action action)
            {
                AutoSizeAxes = Axes.Both;
                Masking = true;
                CornerRadius = 4;
                Padding = new MarginPadding { Horizontal = 12, Vertical = 7 };

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
                        new SpriteText
                        {
                            Text = label,
                            Font = FontUsage.Default.With(size: 17),
                            Colour = Color4.White,
                        },
                    },
                };

                Action = action;
            }

            private Action Action { get; }

            protected override bool OnClick(ClickEvent e)
            {
                Action();
                return true;
            }

            protected override bool OnHover(HoverEvent e)
            {
                background.FadeColour(Color4Extensions.FromHex(@"3d4870"), 100, Easing.Out);
                return true;
            }

            protected override void OnHoverLost(HoverLostEvent e)
            {
                background.FadeColour(Color4Extensions.FromHex(@"2b3350"), 200, Easing.Out);
                base.OnHoverLost(e);
            }
        }
    }
}
