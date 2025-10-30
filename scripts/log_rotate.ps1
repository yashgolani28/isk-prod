$dir="C:\ESSI\Projects\isk6843\system-logs"; $max=20MB; $keep=14
Get-ChildItem $dir -File | %{
  if ($_.Length -gt $max) {
    $ts=(Get-Date -Format yyyyMMdd_HHmmss)
    $bak="$($_.FullName).$ts"
    Copy-Item $_.FullName $bak -Force
    Clear-Content $_.FullName
  }
}
Get-ChildItem $dir -File | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-$keep)} | Remove-Item -Force
