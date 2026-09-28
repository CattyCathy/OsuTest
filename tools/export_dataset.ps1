# Exports the corpus to training data, split across several processes.
#
# The export is dominated by the model's inference - 363 of 370 seconds for one four-minute track - and that already
# uses every thread the ONNX runtime will give it, so running more threads inside one process would not help. Running
# several tracks at once does. Eight processes at two threads each sit inside the 24 logical cores this machine has.

param(
    [string]$Corpus = "D:\Linux\Proj\OsuTest\train-corpus",
    [string]$Dataset = "D:\Linux\Proj\OsuTest\dataset",
    [int]$Shards = 8,
    [switch]$MapLevel
)

$ErrorActionPreference = "Stop"

$dll = "D:\Linux\Proj\ParaTactus\ParaTactus.Tests\bin\Debug\net10.0\ParaTactus.Tests.dll"

if (-not (Test-Path $dll)) { throw "build ParaTactus.Tests first" }

New-Item -ItemType Directory -Force -Path $Dataset | Out-Null

# An environment variable set here is not passed to Start-Job's child process, so anything the exporter reads has to be
# named as an argument and set inside the job.
$forceMapLevel = if ($PSBoundParameters.ContainsKey('MapLevel')) { "1" } else { "" }

$jobs = @()

for ($i = 0; $i -lt $Shards; $i++) {
    $jobs += Start-Job -ScriptBlock {
        param($Dll, $Corpus, $Dataset, $Index, $Of, $MapLevel)

        $env:OSUTEST_CORPUS = $Corpus
        $env:OSUTEST_DATASET = $Dataset
        $env:OSUTEST_SHARD = "$Index/$Of"
        $env:OSUTEST_THREADS = "2"
        $env:OSUTEST_MAP_LEVEL = $MapLevel

        & dotnet vstest $Dll --Tests:ParaTactus.Tests.DatasetExportProbe.ExportAudioAndLabelsForTraining 2>&1
    } -ArgumentList $dll, $Corpus, $Dataset, $i, $Shards, $forceMapLevel
}

Write-Host "started $Shards shards; waiting" -ForegroundColor Cyan

$done = 0

foreach ($job in $jobs) {
    $output = Receive-Job $job -Wait
    $done++
    $exported = ($output | Select-String -Pattern "exported \d+" | Select-Object -Last 1).Line
    $failed = ($output | Select-String -Pattern "failed (\d+)" | Select-Object -Last 1)
    Write-Host "[$done/$Shards] $($exported -replace '^\s+', '')"
    Remove-Job $job -Force
}

$audio = (Get-ChildItem (Join-Path $Dataset "audio") -File -ErrorAction SilentlyContinue).Count
$labels = (Get-ChildItem (Join-Path $Dataset "labels") -File -ErrorAction SilentlyContinue).Count

Write-Host ""
Write-Host "wrote $audio audio files and $labels label files to $Dataset" -ForegroundColor Green
