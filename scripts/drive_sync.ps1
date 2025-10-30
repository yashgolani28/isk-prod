param([string]$ExtraArgs = "")

$ErrorActionPreference = "Stop"
$env:APP_CONFIG = "C:\ESSI\Projects\isk6843\app_config.json"
$env:DB_DSN     = "postgresql://radar_user:essi@localhost/iwr6843_db"
$env:FFMPEG_BIN = "C:\ffmpeg\bin\ffmpeg.exe"
$env:PYTHONPATH = "C:\ESSI\Projects\isk6843"
$env:APP_CONFIG = "C:\ESSI\Projects\isk6843\app_config.json"
$env:DB_DSN     = "postgresql://radar_user:essi@localhost/iwr6843_db"
$env:FFMPEG_BIN = "C:\ffmpeg\bin\ffmpeg.exe"

# injected values
$root   = "C:\ESSI\Projects\isk6843"
$log    = "C:\ESSI\Projects\isk6843\system-logs\drive_sync.log"
$python = "C:\ESSI\Projects\isk6843\.venv\Scripts\python.exe"
$script = "scripts\sync_violations_to_drive.py"
$args   = ""

# ensure working dir and logs exist
New-Item -ItemType Directory -Force -Path "C:\ESSI\Projects\isk6843\system-logs" | Out-Null
Set-Location $root

Start-Transcript -Path $log -Append | Out-Null
try {
  Write-Host "=== drive_sync @ $(Get-Date -Format s) ==="
  Write-Host "python: $python"
  Write-Host "script: $script  args: $args  extra: $ExtraArgs"

  & $python "$root\$script" $args $ExtraArgs
  if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
    throw "process exited with code $LASTEXITCODE"
  }

  Write-Host "[OK] drive_sync finished"
}
catch {
  Write-Host "[ERR] drive_sync: $($_.Exception.Message)"
}
finally {
  Stop-Transcript | Out-Null
}





