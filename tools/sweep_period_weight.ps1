$py = "D:\Linux\Proj\OsuTest\.venv-train\Scripts\python.exe"

Write-Host "=== period weight 3 ==="
& $py "D:\Linux\Proj\OsuTest\tools\train_detector.py" --epochs 40 --period-weight 3 --out "D:\Linux\Proj\OsuTest\model-pw3" --cache "D:\Linux\Proj\OsuTest\mel-cache" 2>&1 | Select-Object -Last 4
Write-Host "=== period weight 10 ==="
& $py "D:\Linux\Proj\OsuTest\tools\train_detector.py" --epochs 40 --period-weight 10 --out "D:\Linux\Proj\OsuTest\model-pw10" --cache "D:\Linux\Proj\OsuTest\mel-cache" 2>&1 | Select-Object -Last 4
Write-Host "=== period weight 30 ==="
& $py "D:\Linux\Proj\OsuTest\tools\train_detector.py" --epochs 40 --period-weight 30 --out "D:\Linux\Proj\OsuTest\model-pw30" --cache "D:\Linux\Proj\OsuTest\mel-cache" 2>&1 | Select-Object -Last 4
Write-Host "=== all three done ==="
