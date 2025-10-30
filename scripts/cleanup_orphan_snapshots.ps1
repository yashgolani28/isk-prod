$ErrorActionPreference="Continue"
$Root="C:\ESSI\Projects\isk6843"
$Py  = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\cleanup_orphan_snapshots.py"
$Log = Join-Path $Root "system-logs\orphan_cleanup.log"
"$(Get-Date -Format u) [ORPHAN] start" | Tee-Object $Log -Append
& $Py $Script --cutoff-days 14 2>&1 | Tee-Object $Log -Append
"$(Get-Date -Format u) [ORPHAN] done code=$LASTEXITCODE" | Tee-Object $Log -Append
