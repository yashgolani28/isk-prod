param([string]$ServiceName = $env:ISK_SERVICE_NAME)
if (-not $ServiceName -or $ServiceName.Trim() -eq "") { $ServiceName = "ISK App" }

$Root    = Split-Path -Parent $PSScriptRoot
$LogDir  = Join-Path $Root "system-logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$JobLog = Join-Path $LogDir "service_restart.ps1.log"
$UiLog  = Join-Path $LogDir "isk-app.log"   # mirror only here (radar.log is lock-held by the app)

function Append-Line($path, $text) {
  try { Add-Content -Path $path -Value $text -Encoding utf8 -ErrorAction Stop } catch {}
}

function Write-All($lvl, $msg) {
  $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
  $line = "[$ts] [$lvl] $msg"
  Append-Line $JobLog $line
  Append-Line $UiLog  $line
}

try {
  Write-All "INFO" "restart $ServiceName - stopping"
  $nssmStop = & nssm stop $ServiceName 2>&1
  if ($nssmStop) { Add-Content -Path $JobLog -Value ($nssmStop -join [Environment]::NewLine) -Encoding utf8 }

  Start-Sleep -Seconds 2
  Write-All "INFO" "starting $ServiceName"
  $nssmStart = & nssm start $ServiceName 2>&1
  if ($nssmStart) { Add-Content -Path $JobLog -Value ($nssmStart -join [Environment]::NewLine) -Encoding utf8 }

  Write-All "OK  " "restart complete"
} catch {
  Write-All "ERR " $_.Exception.Message
  exit 1
}
