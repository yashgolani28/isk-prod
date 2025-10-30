param([string]$Root="C:\ESSI\Projects\isk6843",[int]$KeepDays=3)
$now=Get-Date
$log=(Join-Path $Root "system-logs\log_rotate_and_prune.ps1.log")
function L([string]$m){ "2025-10-30 11:44:09 $m" | Out-File $log -Append -Encoding UTF8 }
foreach($d in @("system-logs","radar-logs")){
  $p=Join-Path $Root $d
  if(-not (Test-Path $p)){continue}
  # delete rotated chunks, zips, and dated CSVs older than KeepDays
  Get-ChildItem $p -File -Recurse -Include *.log.*,*.stdout.log.*,*.stderr.log.*,*.zip,violations_*.csv |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$KeepDays) } |
    ForEach-Object { try{ Remove-Item $_.FullName -Force; L "[prune] $($_.FullName)" }catch{ L "[prune-fail] $($_.FullName): $($_.Exception.Message)" } }
}
