<#
  write_wslconfig.ps1: cap the WSL2 VM at 24 GB of the PC's 32 GB (RUNBOOK step 4).
  Default WSL gets half the RAM (16 GB); Planck's data prep and vLLM want more, and the
  remaining 8 GB keeps Windows responsive. Existing lines in %USERPROFILE%\.wslconfig are
  kept; only memory= and swap= under [wsl2] are set. A dated backup is written first.

  Run (no admin needed):
    powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\<winuser>\planck-kit\pc\win\write_wslconfig.ps1
  Takes effect after `wsl --shutdown` (this script does NOT run it: that would kill any job
  running in WSL). Check inside WSL afterwards: free -g  (total about 23-24).
#>
param([string]$Memory = "24GB", [string]$Swap = "8GB")
$ErrorActionPreference = "Stop"
$path = Join-Path $env:USERPROFILE ".wslconfig"
$want = [ordered]@{ "memory" = $Memory; "swap" = $Swap }

$lines = @()
if (Test-Path $path) {
    Copy-Item $path ("{0}.bak-{1}" -f $path, (Get-Date -Format "yyyyMMdd-HHmmss"))
    $lines = @(Get-Content $path)
}

$out = New-Object System.Collections.Generic.List[string]
$inWsl2 = $false
$seenWsl2 = $false
$done = @{}
foreach ($ln in $lines) {
    $t = $ln.Trim()
    if ($t -match '^\[(.+)\]$') {
        if ($inWsl2) {                           # leaving [wsl2]: add what was missing
            foreach ($k in $want.Keys) { if (-not $done[$k]) { $out.Add("$k=$($want[$k])"); $done[$k] = $true } }
        }
        $inWsl2 = ($Matches[1].Trim().ToLower() -eq "wsl2")
        if ($inWsl2) { $seenWsl2 = $true }
        $out.Add($ln)
        continue
    }
    if ($inWsl2 -and $t -match '^([A-Za-z]+)\s*=') {
        $key = $Matches[1].ToLower()
        if ($want.Contains($key)) {
            $out.Add("$key=$($want[$key])"); $done[$key] = $true
            continue
        }
    }
    $out.Add($ln)
}
if ($inWsl2) {
    foreach ($k in $want.Keys) { if (-not $done[$k]) { $out.Add("$k=$($want[$k])"); $done[$k] = $true } }
}
if (-not $seenWsl2) {
    $out.Add("# Planck (pc/win/write_wslconfig.ps1): WSL2 VM limits")
    $out.Add("[wsl2]")
    foreach ($k in $want.Keys) { $out.Add("$k=$($want[$k])") }
}

# WSL reads .wslconfig as UTF-8 or ASCII; write ASCII with no BOM.
[System.IO.File]::WriteAllLines($path, $out, (New-Object System.Text.ASCIIEncoding))
"== $path"
Get-Content $path
""
"Now, when no WSL job is running: wsl --shutdown   (then start WSL again and check: free -g)"
