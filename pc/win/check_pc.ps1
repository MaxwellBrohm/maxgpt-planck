<#
  check_pc.ps1: read-only inventory of the PC for Planck (RUNBOOK step 1). Changes nothing.
  Run: powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\<winuser>\planck-kit\pc\win\check_pc.ps1
#>
$ErrorActionPreference = "Continue"
$env:WSL_UTF8 = "1"            # wsl.exe prints UTF-16 otherwise, which reads as s p a c e d text

"== drives (PLAN wants about 150 GB free for Planck; WSL's disk lives on C: by default)"
Get-PSDrive -PSProvider FileSystem | Where-Object { $null -ne $_.Used } | ForEach-Object {
    "{0}:  free {1,7:N0} GB of {2,7:N0} GB" -f $_.Name, ($_.Free / 1GB), (($_.Used + $_.Free) / 1GB)
}
$c = Get-PSDrive C
if (($c.Free / 1GB) -lt 150) { "WARNING: C: has under 150 GB free; see RUNBOOK step 1 (install WSL on another drive)" }

"== memory and OS"
$cs = Get-CimInstance Win32_ComputerSystem
"{0:N1} GB RAM" -f ($cs.TotalPhysicalMemory / 1GB)
$os = Get-CimInstance Win32_OperatingSystem
"{0} {1} build {2}, last boot {3}" -f $os.Caption, $os.Version, $os.BuildNumber, $os.LastBootUpTime

"== GPU (Windows nvidia-smi)"
& nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.limit --format=csv

"== WSL"
& wsl.exe --version
& wsl.exe --list --verbose

"== power plan (AC sleep and hibernate should be 0 = never after step 3)"
& powercfg /getactivescheme
& powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String "Current AC"
& powercfg /query SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE | Select-String "Current AC"

"== Windows Update pause and active hours"
Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings" -ErrorAction SilentlyContinue |
    Select-Object PauseUpdatesExpiryTime, ActiveHoursStart, ActiveHoursEnd | Format-List

"== .wslconfig"
$wc = Join-Path $env:USERPROFILE ".wslconfig"
if (Test-Path $wc) { Get-Content $wc } else { "(none: WSL uses half the RAM)" }

"== Planck runner task"
Get-ScheduledTask -TaskName "PlanckRunner" -ErrorAction SilentlyContinue |
    Select-Object TaskName, State | Format-List
Get-ScheduledTaskInfo -TaskName "PlanckRunner" -ErrorAction SilentlyContinue |
    Select-Object LastRunTime, LastTaskResult | Format-List
