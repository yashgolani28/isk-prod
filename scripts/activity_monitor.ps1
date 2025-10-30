param([string]$ExtraArgs = "")

$ErrorActionPreference = "Stop"
$env:APP_CONFIG = "C:\ESSI\Projects\isk6843\app_config.json"
$env:DB_DSN     = "postgresql://radar_user:essi@localhost/iwr6843_db"
$env:FFMPEG_BIN = "C:\ffmpeg\bin\ffmpeg.exe"
$env:PYTHONPATH = "C:\ESSI\Projects\isk6843"

# injected values
$root   = "C:\ESSI\Projects\isk6843"
$log    = "C:\ESSI\Projects\isk6843\system-logs\activity_monitor.log"
$python = "C:\ESSI\Projects\isk6843\.venv\Scripts\python.exe"
$script = "scripts\activity_monitor.py"
$args   = ""

# ensure working dir and logs exist
New-Item -ItemType Directory -Force -Path "C:\ESSI\Projects\isk6843\system-logs" | Out-Null
Set-Location $root

Start-Transcript -Path $log -Append | Out-Null
try {
  Write-Host "=== activity_monitor @ $(Get-Date -Format s) ==="
  Write-Host "python: $python"
  Write-Host "script: $script  args: $args  extra: $ExtraArgs"

  & $python "$root\$script" $args $ExtraArgs
  if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
    throw "process exited with code $LASTEXITCODE"
  }

  Write-Host "[OK] activity_monitor finished"
}
catch {
  Write-Host "[ERR] activity_monitor: $($_.Exception.Message)"
}
finally {
  Stop-Transcript | Out-Null
}




