#nullable enable

using System;
using System.IO;
using System.Threading;
using ManagedBass;
using ParaTactus;

namespace OsuTest.Game.Audio
{
    /// <summary>
    /// Plays an audio file and reports its playback position, using BASS directly.
    /// </summary>
    /// <remarks>
    /// A fallback for files the framework cannot open. <see cref="osu.Framework.Audio.Track.ITrackStore"/>
    /// resolves names through the stores attached to it rather than through the file system, and the concrete
    /// store that could be handed a directory is internal - so for files outside the game's own resources there
    /// is no public way to get a framework track. That call returns null with no exception, which is what
    /// produced "the audio system returned no track".
    ///
    /// Rather than reimplementing <c>ITrack</c> - which drags in the adjustable-clock and adjustable-audio
    /// interfaces - this exposes the two things the visuals need: a position that advances with the music, and
    /// transport control. The caller feeds the position into a <c>ManualClock</c>, which the beat-synced
    /// container then treats as music time exactly as it would a real track clock.
    ///
    /// Playback is a BASS push stream fed from a decode channel. A decode channel on its own does not advance:
    /// it exists to be pulled from, so its position stays at zero and it cannot serve as a clock. Pushing its
    /// output into a real stream makes BASS drive the timeline, and because the samples are only ever produced
    /// on demand this works with no output device at all - on such a machine everything functions except that
    /// nothing is audible.
    /// </remarks>
    public sealed class BassAudioSource : IDisposable
    {
        private readonly int streamHandle;
        private int decoderHandle;
        private readonly StreamProcedure pump;
        private readonly double baseFrequency;

        private bool disposed;

        private BassAudioSource(int streamHandle, int decoderHandle, StreamProcedure pump, string path, double length, double baseFrequency)
        {
            this.streamHandle = streamHandle;
            this.decoderHandle = decoderHandle;
            this.pump = pump;
            Path = path;
            Length = length;
            this.baseFrequency = baseFrequency;
        }

        /// <summary>The file this source was opened from.</summary>
        public string Path { get; }

        /// <summary>Length in milliseconds.</summary>
        public double Length { get; }

        /// <summary>
        /// Position of the audio being produced, in milliseconds.
        /// </summary>
        /// <remarks>
        /// Computed as the point playback last started from plus the time actually spent playing, rather than
        /// read back from BASS.
        ///
        /// Neither channel's position is usable for this. The playback stream's own position restarts from zero
        /// on a seek, because a push stream cannot be rewound. The decoder's position runs ahead of what is
        /// audible, because the pump has to stay ahead of the stream's buffer - measured at roughly double real
        /// time, which would run the visuals visibly ahead of the music. A stopwatch is immune to both, and stops
        /// when playback is paused.
        /// </remarks>
        public double CurrentTime
        {
            get
            {
                if (disposed)
                    return 0;

                return seekBase + (IsRunning ? playbackTime.Elapsed.TotalMilliseconds : pausedAt);
            }
        }

        private double seekBase;
        private double pausedAt;
        private readonly System.Diagnostics.Stopwatch playbackTime = new System.Diagnostics.Stopwatch();

        /// <summary>Whether audio is currently being produced.</summary>
        public bool IsRunning { get; private set; }

        /// <summary>
        /// Playback rate. Applied as a frequency change, so the music - and therefore the clock derived from its
        /// position - runs faster.
        /// </summary>
        public double Rate
        {
            get => baseFrequency <= 0 ? 1 : Bass.ChannelGetAttribute(streamHandle, ChannelAttribute.Frequency) / baseFrequency;
            set
            {
                if (disposed || streamHandle == 0 || value <= 0)
                    return;

                Bass.ChannelSetAttribute(streamHandle, ChannelAttribute.Frequency, baseFrequency * value);
            }
        }

        /// <summary>
        /// Opens a file for playback, or returns null with a reason when it cannot be played.
        /// </summary>
        public static BassAudioSource? TryOpen(string path, out string? failure)
        {
            failure = null;

            if (!File.Exists(path))
            {
                failure = $"file does not exist: {path}";
                return null;
            }

            int decoder = Bass.CreateStream(path, 0, 0, BassFlags.Decode | BassFlags.Float);

            if (decoder == 0)
            {
                failure = $"BASS could not decode the file ({Bass.LastError})";
                return null;
            }

            try
            {
                var info = Bass.ChannelGetInfo(decoder);
                int channels = Math.Max(1, info.Channels);
                int frequency = info.Frequency > 0 ? info.Frequency : 44100;

                // The pump is kept in a field so the delegate cannot be collected while BASS still calls it, which
                // would crash the audio thread.
                int decoderRef = decoder;
                BassAudioSource? self = null;

                StreamProcedure pump = (handle, buffer, length, user) =>
                {
                    try
                    {
                        int read = Bass.ChannelGetData(decoderRef, buffer, length);

                        if (read > 0)
                            return read;

                        // Nothing left in the file: mark the end so BASS stops rather than reporting a stall.
                        return (int)StreamProcedureType.End;
                    }
                    catch (Exception)
                    {
                        // An exception escaping into the audio thread would take the process down. Ending the
                        // stream is the safe response.
                        return (int)StreamProcedureType.End;
                    }
                };

                int stream = Bass.CreateStream(frequency, channels, BassFlags.Float, pump, IntPtr.Zero);

                if (stream == 0)
                {
                    failure = $"BASS could not create a playback stream ({Bass.LastError})";
                    Bass.StreamFree(decoder);
                    return null;
                }

                long lengthBytes = Bass.ChannelGetLength(decoder);
                double seconds = Bass.ChannelBytes2Seconds(decoder, lengthBytes);

                double baseFrequency = Bass.ChannelGetAttribute(stream, ChannelAttribute.Frequency);

                self = new BassAudioSource(stream, decoder, pump, path, seconds * 1000, baseFrequency > 0 ? baseFrequency : frequency);
                return self;
            }
            catch (Exception e)
            {
                failure = $"{e.GetType().Name}: {e.Message}";
                Bass.StreamFree(decoder);
                return null;
            }
        }

        /// <summary>Starts or resumes playback from the current position.</summary>
        public void Start()
        {
            if (disposed || streamHandle == 0)
                return;

            Bass.ChannelPlay(streamHandle, false);
            playbackTime.Start();
            IsRunning = true;
        }

        /// <summary>Pauses playback, keeping the position.</summary>
        public void Stop()
        {
            if (disposed || streamHandle == 0)
                return;

            Bass.ChannelPause(streamHandle);

            pausedAt = playbackTime.Elapsed.TotalMilliseconds;
            playbackTime.Stop();
            IsRunning = false;
        }

        /// <summary>
        /// Moves the playback position, in milliseconds.
        /// </summary>
        /// <remarks>
        /// The decoder is seeked directly, which is what actually changes which audio is produced next; BASS
        /// supports this on a decode channel. The playback stream is deliberately not seeked, because a push
        /// stream's position restarts from zero regardless and data already buffered into it cannot be rewound -
        /// the audible effect of the skip is simply the change in what the pump delivers from here on.
        ///
        /// The reported position is then based on the new point plus elapsed playback time, so it reflects the
        /// seek immediately rather than waiting for buffered audio to drain.
        /// </remarks>
        public void Seek(double milliseconds)
        {
            if (disposed || decoderHandle == 0)
                return;

            long bytes = Bass.ChannelSeconds2Bytes(decoderHandle, milliseconds / 1000);

            if (Bass.ChannelSetPosition(decoderHandle, bytes))
            {
                seekBase = milliseconds;
                pausedAt = 0;
                playbackTime.Restart();

                if (!IsRunning)
                    playbackTime.Stop();
            }
        }

        /// <summary>
        /// Number of samples read for spectrum analysis. Small enough to be cheap per frame, large enough for a
        /// usable number of frequency bins.
        /// </summary>
        private const int spectrum_samples = 1024;

        private int spectrumHandle;
        private readonly float[] spectrumReadBuffer = new float[spectrum_samples];
        private readonly float[] spectrumWindow = new float[Fft.SIZE];
        private readonly float[] spectrumAmplitudes = new float[Fft.BIN_COUNT];
        private readonly float[] interpolatedAmplitudes = new float[Fft.BIN_COUNT];

        /// <summary>
        /// Builds the decoder used for visualisation, reporting why it could not be built.
        /// </summary>
        /// <remarks>
        /// Built lazily because a visualiser may never be shown, and on Windows BASS channel creation has to run
        /// on a thread that has been through BASS initialisation - which is the update thread here, the same one
        /// that created the player.
        /// </remarks>
        private bool ensureSpectrumDecoder(out string? failure)
        {
            failure = null;

            if (spectrumHandle != 0)
                return true;

            spectrumHandle = Bass.CreateStream(Path, 0, 0, BassFlags.Decode | BassFlags.Float);

            if (spectrumHandle == 0)
            {
                failure = $"BASS could not open a second decoder for visualisation ({Bass.LastError})";
                return false;
            }

            return true;
        }

        /// <summary>Diagnostics for the spectrum path, used by tests to locate a failure.</summary>
        public readonly record struct SpectrumDiagnostics(
            int Handle,
            bool SeekSucceeded,
            long RequestedPosition,
            int BytesRead,
            int SampleCount,
            float InputMax,
            float OutputMax);

        /// <summary>
        /// Reports the intermediate values of the spectrum computation, for diagnosing a visualiser that is not
        /// reacting. The smoothing below means a single call reads low by design, which makes a bare "it returns
        /// zeros" hard to interpret without these.
        /// </summary>
        public SpectrumDiagnostics DiagnoseSpectrum()
        {
            if (!ensureSpectrumDecoder(out _))
                return new SpectrumDiagnostics(0, false, 0, 0, 0, 0, 0);

            long position = Bass.ChannelSeconds2Bytes(spectrumHandle, CurrentTime / 1000);
            bool seeked = Bass.ChannelSetPosition(spectrumHandle, position);
            int read = Bass.ChannelGetData(spectrumHandle, spectrumReadBuffer, spectrum_samples * sizeof(float));

            float inputMax = 0;

            for (int i = 0; i < spectrum_samples; i++)
                inputMax = Math.Max(inputMax, Math.Abs(spectrumReadBuffer[i]));

            Fft.AnalyseMagnitudes(spectrumReadBuffer, spectrumAmplitudes);

            float outputMax = 0;

            foreach (float value in spectrumAmplitudes)
                outputMax = Math.Max(outputMax, value);

            return new SpectrumDiagnostics(spectrumHandle, seeked, position, read, read / sizeof(float), inputMax, outputMax);
        }

        /// <summary>
        /// Amplitude per frequency band, for a visualiser.
        /// </summary>
        /// <remarks>
        /// Computed on demand rather than by the audio thread: a decode channel does not produce amplitude data,
        /// which is why the bars were flat on this playback path while everything else worked. A second decode
        /// handle seeks to the playback position and reads a small window, which is cheap enough to do once per
        /// frame and keeps the classes of work separated - the playback handle is only ever touched by the audio
        /// thread, this one only by the update thread.
        ///
        /// This reads ahead of what is audible by however much the push stream has buffered, so the visualiser
        /// leads the sound very slightly. That is the correct trade for a visual: the alternative is no reaction
        /// at all.
        /// </remarks>
        public ReadOnlySpan<float> GetFrequencyAmplitudes()
        {
            if (disposed)
                return default;

            if (!ensureSpectrumDecoder(out _))
                return default;

            double milliseconds = CurrentTime;
            long position = Bass.ChannelSeconds2Bytes(spectrumHandle, milliseconds / 1000);

            if (!Bass.ChannelSetPosition(spectrumHandle, position))
                return interpolatedAmplitudes;

            int read = Bass.ChannelGetData(spectrumHandle, spectrumReadBuffer, spectrum_samples * sizeof(float));

            if (read <= 0)
                return interpolatedAmplitudes;

            int available = read / sizeof(float);
            int fftSize = Fft.SIZE;

            // Use the most recent complete window available, zero-padding the short tail at the end of the file.
            int offset = Math.Max(0, available - fftSize);
            int count = Math.Min(fftSize, available);

            for (int i = 0; i < count; i++)
                spectrumWindow[i] = spectrumReadBuffer[offset + i];

            for (int i = count; i < fftSize; i++)
                spectrumWindow[i] = 0;

            Fft.AnalyseMagnitudes(spectrumWindow, spectrumAmplitudes);

            // Smoothed so the bars ease rather than snapping between analysis windows.
            for (int i = 0; i < spectrumAmplitudes.Length; i++)
                interpolatedAmplitudes[i] += (spectrumAmplitudes[i] - interpolatedAmplitudes[i]) * 0.5f;

            return interpolatedAmplitudes;
        }

        public void Dispose()
        {
            if (disposed)
                return;

            disposed = true;
            IsRunning = false;

            // Stop before freeing so the audio thread cannot be inside the pump when the decoder goes away.
            if (streamHandle != 0)
            {
                Bass.ChannelStop(streamHandle);
                Bass.StreamFree(streamHandle);
            }

            if (decoderHandle != 0)
                Bass.StreamFree(decoderHandle);

            if (spectrumHandle != 0)
                Bass.StreamFree(spectrumHandle);
        }
    }
}
