# daily_report_email.ps1  (windows)
$Root    = Split-Path -Parent $PSScriptRoot
$LogDir  = Join-Path $Root "system-logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$JobLog = Join-Path $LogDir "daily_report_email.ps1.log"
$UiLog  = Join-Path $LogDir "isk-app.log"   # mirror only here

$Py      = Join-Path $Root ".venv\Scripts\python.exe"
$PyScript= Join-Path $Root "scripts\daily_report_email.py"

# banner
$ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
$hdr = "[$ts] [INFO] starting daily_report_email.py"
Add-Content $JobLog -Value $hdr -Encoding utf8
Add-Content $UiLog  -Value $hdr -Encoding utf8

& $Py $PyScript 2>&1 | Tee-Object -FilePath $JobLog -Append | Out-Null  # keep detailed stdout in job log
# and just a compact trailer into both logs:
$ec = $LASTEXITCODE
$ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
$tail = if ($ec -eq 0) { "[$ts] [OK  ] email job completed" } else { "[$ts] [ERR ] exit $ec" }
Add-Content $JobLog -Value $tail -Encoding utf8
Add-Content $UiLog  -Value $tail -Encoding utf8
exit $ec

