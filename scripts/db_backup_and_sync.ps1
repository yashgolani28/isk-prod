$ErrorActionPreference="Continue"
$Root="C:\ESSI\Projects\isk6843"
$Log = Join-Path $Root "system-logs\db_backup_and_sync.ps1.log"
$BakDir = Join-Path $Root "backups"
New-Item -ItemType Directory -Force -Path $BakDir | Out-Null
$Dump = Join-Path $BakDir ("iwr6843_db_" + (Get-Date -Format yyyyMMdd_HHmmss) + ".sql")
"$(Get-Date -Format u) [DBBACKUP] pg_dump -> $Dump" | Tee-Object $Log -Append
& "C:\Program Files\PostgreSQL\17\bin\pg_dump.exe" -U postgres -d iwr6843_db -F p -f $Dump 2>&1 | Tee-Object $Log -Append
if ($LASTEXITCODE -eq 0) {
  "$(Get-Date -Format u) [DBBACKUP] upload via rclone" | Tee-Object $Log -Append
  & rclone copy $Dump "gdrive:RadarDB" 2>&1 | Tee-Object $Log -Append
}
"$(Get-Date -Format u) [DBBACKUP] done code=$LASTEXITCODE" | Tee-Object $Log -Append
