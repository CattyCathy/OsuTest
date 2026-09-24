"""Repairs BeatSyncDemo.cs after its Chinese string literals were destroyed by an encoding round trip.

Run with the repository root as the working directory:
    python tools/repair_beatsyncdemo.py
"""

import io

PATH = r"OsuTest\OsuTest.Game\Graphics\BeatSyncDemo.cs"

# Line number -> replacement line. Only lines containing destroyed text are listed; every other line is untouched.
REPLACEMENTS = {
    188: '                        offsetInfo = new SpriteText { Text = "时间偏移 0ms", Font = FontUsage.Default.With(size: 18) },',
    215: '                            TooltipText = "拖动定位",',
    234: '            addPickerButton("使用内置示例曲 (120 -> 160 BPM)", loadGeneratedDemo);',
    235: '            addPickerButton("选择音频文件…", presentFileSelector);',
    236: '            addPickerButton("停止播放", stopPlayback);',
    237: '            addPickerButton("继续播放", startPlayback);',
    238: '            addPickerButton("播放速度 1.5x", () => setRate(1.5));',
    239: '            addPickerButton("播放速度 0.75x", () => setRate(0.75));',
    240: '            addPickerButton("重新加载当前音源", reloadCurrentSource);',
    241: '            addPickerButton("重新扫描音频设备", () => deviceSelector.Rebuild());',
    242: '            addPickerButton("音频自检", runAudioSelfCheck);',
    243: '            addPickerButton("显示 tempo 时间线", showTempoTimeline);',
    244: '            addPickerButton("导出 tempo 诊断文件", exportTempoDiagnostics);',
    245: '            addPickerButton("偏移 -10ms", () => adjustOffset(-10));',
    246: '            addPickerButton("偏移 +10ms", () => adjustOffset(10));',
    247: '            addPickerButton("偏移归零", () => adjustOffset(-beatOffset, absolute: true));',
    265: '                selfCheckReport = "还没有加载音源";',
    272: '            report.Append($"共 {beatGrid.Beats.Count} 拍，时长 {beatGrid.Duration / 1000:0.#}s "',
    273: '                          + $"metrical 层级 {beatGrid.MetricalShift:+#;-#;0}");',
    292: '                              + (until - beatGrid.Beats[from] < 2000 ? "  <- 很近" : string.Empty));',
    316: '                status.Text = "还没有加载音源";',
    323: '            status.Text = "正在导出诊断文件…";',
    351: '                        selfCheckReport = $"导出失败：{grid.Beats.Count} 拍\\n{target}";',
    358: '                    Schedule(() => status.Text = $"诊断失败：{e.GetType().Name}: {e.Message}");',
    424: '                status.Text = "还没有加载音源";',
    434: '            offsetInfo.Text = $"时间偏移：{beatOffset:+0;-0;0}ms";',
    435: '            status.Text = $"对应的时间偏移 {beatOffset:+0;-0;0}ms"',
    436: '                          + (beatOffset == 0 ? "，无偏移" : "，已应用"));',
    452: '            report.Append($"设备列表({names.Length}): {(names.Length == 0 ? "<空>" : string.Join(" | ", names))}");',
    455: '            report.Append($"\\n当前设备：{current}");',
    460: '                report.Append($"\\n静音音轨：{(silent == null ? "null" : $"ok, length={silent.Length:0}ms")}");',
    464: '                report.Append($"\\n框架音轨失败：{e.GetType().Name}: {e.Message}");',
    476: '                    report.Append($"\\n框架音轨存储: {(probe == null ? "null（对用户选择的文件是预期结果，由 BASS 直连接手）" : $"ok, {probe.GetType().Name}, length={probe.Length:0}ms")}");',
    480: '                    report.Append($"\\n框架音轨失败：{e.GetType().Name}: {e.Message}");',
    484: '            report.Append($"\\nBASS 直连播放: {(bassSource == null ? "未使用" : $"使用中，位置 {bassSource.CurrentTime:0}ms / {bassSource.Length:0}ms，运行中={bassSource.IsRunning}")}");',
    508: '                    report.Append($"\\n平均: {beatGrid.Beats.Count} 个拍点，中位 {median:0.#}ms（{60000 / median:0.#} BPM），"',
    509: '                                  + $"离散度 {spread:0.#}ms，metrical 层级 {beatGrid.MetricalShift:+#;-#;0}");',
    513: '            report.Append($"\\n上次播放失败原因: {playbackError ?? "<无>"}");',
    532: '                status.Text = "还没有加载音源";',
    538: '                status.Text = $"音源文件不存在：{currentSourcePath}";',
    542: '            status.Text = "正在重新加载当前音源…";',
    557: '            loadSource(path, $"使用内置示例曲（{map.Segments[0].Bpm:0.#} -> {map.Segments[^1].Bpm:0.#} BPM）");',
    574: '                    presentInGamePicker("系统文件选择器不可用，已改用游戏内选择器");',
    583: '                presentInGamePicker($"系统文件选择器不可用：{e.GetType().Name}，已改用游戏内选择器");',
    624: '            sourceInfo.Text = $"音源：{displayName}";',
    625: '            status.Text = "正在解码并分析…";',
    650: '                playbackError = loaded == null ? playbackFailure ?? "音频系统不可用" : null;',
    689: '                        error = $"解码失败（{e.Error}）：文件格式或编解码器不受支持，或音频系统未能启动";',
    711: '                status.Text = $"加载失败：{error ?? "未知错误"}";',
    717: '                status.Text = "没有可用的音频设备，已改用静音音轨";',
    732: '            offsetInfo.Text = "时间偏移 0ms";',
    748: '            tempoInfo.Text = $"模型分析出 {grid.Beats.Count} 个拍点，{slowest:0.#}-{fastest:0.#} BPM"',
    749: '                             + $"，metrical 层级 {grid.MetricalShift:+#;-#;0}";',
    765: '                        status.Text = $"无法播放（{grid.Beats.Count} 个拍点），BASS 也无法解码该文件：{bassFailure}";',
    776: '                    playbackError = $"BASS 无法解码该文件：{bassFailure}";',
    798: '                ? $"已就绪：{grid.Beats.Count} 拍 / {grid.Duration / 1000:0.#}s"',
    799: '                  + (usingBass ? "，由 BASS 直连播放" : string.Empty)',
    800: '                : $"预览模式：{playbackError}";',
    915: '                status.Text = "还没有加载音源";',
    926: '            status.Text = $"播放速度 {rate:0.##}x（仅影响播放，不影响时间轴）";',
    1118: '                status.Text = $"拍号 {beatSync.CurrentBeatIndex}   距下一拍 {beatSync.TimeUntilNextBeat:0}ms   驱动动画的拍长 {beatLength:0.#}ms";',
}

with io.open(PATH, encoding="utf-8") as handle:
    lines = handle.read().split("\n")

missing = [n for n in REPLACEMENTS if n > len(lines)]
if missing:
    raise SystemExit(f"line numbers past the end of the file: {missing}")

remaining = []

for number in sorted(REPLACEMENTS):
    if "\u951f" not in lines[number - 1]:
        remaining.append(number)

if remaining:
    raise SystemExit(f"these lines no longer hold destroyed text, refusing to guess: {remaining}")

for number, text in REPLACEMENTS.items():
    lines[number - 1] = text

with io.open(PATH, "w", encoding="utf-8", newline="\n") as handle:
    handle.write("\n".join(lines))

print(f"repaired {len(REPLACEMENTS)} lines")
