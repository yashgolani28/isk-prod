$ErrorActionPreference="Continue"
$Root="C:\ESSI\Projects\isk6843"
$Py  = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\service_watchdog.py"
# keep ps1 wrapper minimal; python writes service_watchdog.py.log
& $Py $Script
