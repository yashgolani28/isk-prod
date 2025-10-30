$root   = "C:\ESSI\Projects\isk6843"
$py     = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "scripts\cleanup_orphan_snapshots.py"

$env:PGHOST='127.0.0.1'
$env:PGPORT='5432'
$env:PGUSER='radar_user'
$env:PGPASSWORD='essi'
$env:PGDATABASE='iwr6843_db'
$env:DB_DSN='postgresql://radar_user:essi@127.0.0.1:5432/iwr6843_db'
$env:ORPHAN_RETENTION_DAYS='14'
$env:ORPHAN_LOG_FILE= Join-Path $root "system-logs\orphan_cleanup.log"
$env:ORPHAN_ROOTS = (Join-Path $root "snapshots") + ";" + (Join-Path $root "backups")

$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
try { [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8 } catch {}

& $py $script
exit $LASTEXITCODE
