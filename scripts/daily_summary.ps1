$ErrorActionPreference="Continue"
$Root="C:\ESSI\Projects\isk6843"
$Py  = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\daily_summary.py"
$Log = Join-Path $Root "system-logs\daily_summary.ps1.log"
"$(Get-Date -Format u) [DAILY] ps1 wrapper start" | Tee-Object $Log -Append
& $Py $Script 2>&1 | Tee-Object $Log -Append
"$(Get-Date -Format u) [DAILY] ps1 wrapper done code=$LASTEXITCODE" | Tee-Object $Log -Append
