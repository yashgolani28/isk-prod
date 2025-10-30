$ErrorActionPreference="Continue"
$Root="C:\ESSI\Projects\isk6843"
$Log = Join-Path $Root "system-logs\db_check.py.log"
$PSQL = "C:\Program Files\PostgreSQL\17\bin\psql.exe"
"$(Get-Date -Format u) [DBCHECK] start" | Tee-Object $Log -Append
& "$PSQL" -U postgres -d iwr6843_db -c "SELECT NOW() ts, COUNT(*) rows FROM radar_data;" 2>&1 | Tee-Object $Log -Append
"$(Get-Date -Format u) [DBCHECK] done code=$LASTEXITCODE" | Tee-Object $Log -Append
