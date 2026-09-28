# Downloads a set of osu! beatmapsets in parallel into a corpus folder.
#
# Parallel because one connection runs at about a hundred kilobytes a second, which is a rate limit rather than a
# link speed: measured against this CDN a single map of eleven megabytes took nearly two minutes, and several at once
# finish in about the time one does. The concurrency is kept low deliberately - this is somebody's account talking to
# a service that publishes a rate limit, and a corpus is not worth an account.

param(
    [string]$Out = "D:\Linux\Proj\OsuTest\train-corpus",
    [int]$Parallel = 6,
    [string]$Queries = "Camellia,xi,Yooh,kors k,Silentroom,Kobaryo,t+pazolite,sakuraburst,Mili,Cranky,DJ Sharpnel,Frums,Hatsuki Yura,Yunomi,M2U,HyuN,Nekomata Master,Kikuo,Reol,Aimer,DragonForce,Foreground Eclipse,BilliumMoto,beatMARIO,Aoi,S3RL,Street,An,PinocchioP,ARForest",
    [int]$PerQuery = 4,
    [int]$Pages = 2
)

$ErrorActionPreference = "Stop"

$ini = Get-Content "$env:APPDATA\osu\game.ini"
$token = (($ini | Where-Object { $_ -like 'Token = *' } | Select-Object -First 1).Substring(8).Trim() -split '\|')[0]

if (-not $token) { throw "no token in game.ini" }

Add-Type -AssemblyName System.Net.Http

New-Item -ItemType Directory -Force -Path $Out | Out-Null

# One client for the searches, on this thread.
$handler = New-Object System.Net.Http.HttpClientHandler
$handler.AllowAutoRedirect = $false
$api = New-Object System.Net.Http.HttpClient($handler)
$api.Timeout = [TimeSpan]::FromSeconds(120)
$api.DefaultRequestHeaders.Authorization = New-Object System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", $token)
$api.DefaultRequestHeaders.Accept.ParseAdd("application/json")

function Get-Body([string]$uri) {
    $req = New-Object System.Net.Http.HttpRequestMessage("GET", $uri)
    $resp = $null
    $resp = $api.SendAsync($req).Result

    if (-not $resp.IsSuccessStatusCode) {
        throw "HTTP $([int]$resp.StatusCode)"
    }

    $body = $resp.Content.ReadAsStringAsync().Result
    $resp.Dispose()

    return $body
}

# The candidates, gathered first so the downloads can run without the search thread in the way.
$candidates = New-Object System.Collections.ArrayList
$seen = @{}

foreach ($q in ($Queries -split ',')) {
    $q = $q.Trim()

    if (-not $q) { continue }

    $cursor = $null

    for ($page = 0; $page -lt $Pages; $page++) {
        $uri = "https://osu.ppy.sh/api/v2/beatmapsets/search?m=0&s=ranked&q=" + [uri]::EscapeDataString($q)

        if ($cursor) { $uri += "&cursor_string=" + [uri]::EscapeDataString($cursor) }

        try {
            $result = Get-Body $uri | ConvertFrom-Json
        } catch {
            Write-Host "search '$q' failed: $($_.Exception.Message)"
            break
        }

        $taken = 0

        foreach ($set in $result.beatmapsets) {
            if ($taken -ge $PerQuery) { break }
            if ($seen.ContainsKey($set.id)) { continue }
            if ($set.video) { continue }
            if ($set.availability.download_disabled) { continue }

            $seen[$set.id] = $true
            [void]$candidates.Add([pscustomobject]@{ Id = $set.id; Title = $set.title; Artist = $set.artist; Bpm = $set.bpm })
            $taken++
        }

        if ($result.cursor -and $result.cursor._score) { $cursor = $result.cursor._score } else { break }
    }

    Write-Host "searched '$q', $($candidates.Count) candidates so far"
}

Write-Host ""
Write-Host "$($candidates.Count) beatmapsets to fetch, $Parallel at a time" -ForegroundColor Cyan
Write-Host ""

# The per-download work, as a script block run on a pool. Each run gets its own client: HttpClient is not documented
# as safe to share for concurrent requests across runspaces.
$work = {
    param($Set, $Out, $Token)

    Add-Type -AssemblyName System.Net.Http

    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.AllowAutoRedirect = $false
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(600)
    $client.DefaultRequestHeaders.Authorization = New-Object System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", $Token)

    $name = "$($Set.Id) $($Set.Artist) - $($Set.Title).osz" -replace '[\\/:*?"<>|]', '_'
    $path = Join-Path $Out $name

    if (Test-Path $path) { return "skipped $name" }

    $req = New-Object System.Net.Http.HttpRequestMessage("GET", "https://osu.ppy.sh/api/v2/beatmapsets/$($Set.Id)/download")
    $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result

    if ([int]$resp.StatusCode -ne 302) {
        $code = [int]$resp.StatusCode
        $resp.Dispose()
        return "FAILED $($Set.Id): HTTP $code"
    }

    $url = $resp.Headers.Location.ToString()
    $resp.Dispose()

    $bytes = $client.GetByteArrayAsync($url).Result
    [System.IO.File]::WriteAllBytes($path, $bytes)

    return "ok $([math]::Round($bytes.Length / 1MB, 1))MB $($Set.Title)"
}

$pool = [runspacefactory]::CreateRunspacePool(1, $Parallel)
$pool.Open()

$running = New-Object System.Collections.ArrayList

foreach ($set in $candidates) {
    $ps = [powershell]::Create()
    $ps.RunspacePool = $pool
    [void]$ps.AddScript($work).AddArgument($set).AddArgument($Out).AddArgument($token)
    [void]$running.Add([pscustomobject]@{ Shell = $ps; Handle = $ps.BeginInvoke(); Set = $set })
}

$done = 0

foreach ($item in $running) {
    $output = $item.Shell.EndInvoke($item.Handle)
    $done++
    Write-Host ("[{0,3}/{1}] {2}" -f $done, $running.Count, ($output -join ' '))
    $item.Shell.Dispose()
}

$pool.Close()
$pool.Dispose()

Write-Host ""
Write-Host "corpus now holds $((Get-ChildItem $Out -File).Count) files" -ForegroundColor Green
