<#
  pause_updates.ps1: pause Windows Update for up to 35 days and set wide active hours
  (RUNBOOK step 5). Windows caps a pause at 35 days (5 weeks), then forces an install and a
  reboot, so PLAN.md books update windows at checkpoint boundaries: install, reboot, then
  run this again (Sep 25, Oct 24-25, Nov 28-29, Jan 2-3, Feb 6-7).

  Run in an ADMIN PowerShell AFTER installing whatever is pending:
    powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\<winuser>\planck-kit\pc\win\pause_updates.ps1
  Then LOOK at Settings > Windows Update: it must say "Updates paused until <date>". If it
  does not (some builds ignore these keys), click "Pause for 5 weeks" there by hand.
#>
#Requires -RunAsAdministrator
param([int]$Days = 35, [int]$ActiveStart = 6, [int]$ActiveEnd = 23)
$ErrorActionPreference = "Stop"
if ($Days -lt 1 -or $Days -gt 35) { throw "Days must be 1..35 (Windows allows at most 35)" }
if ((($ActiveEnd - $ActiveStart + 24) % 24) -gt 18) { throw "active hours may span at most 18 hours" }

$k = "HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings"
if (-not (Test-Path $k)) { New-Item -Path $k -Force | Out-Null }
$fmt = "yyyy-MM-ddTHH:mm:ssZ"
$start = (Get-Date).ToUniversalTime()
$end = $start.AddDays($Days)
$s, $e = $start.ToString($fmt), $end.ToString($fmt)

Set-ItemProperty -Path $k -Name PauseUpdatesStartTime -Value $s
Set-ItemProperty -Path $k -Name PauseUpdatesExpiryTime -Value $e
Set-ItemProperty -Path $k -Name PauseFeatureUpdatesStartTime -Value $s
Set-ItemProperty -Path $k -Name PauseFeatureUpdatesEndTime -Value $e
Set-ItemProperty -Path $k -Name PauseQualityUpdatesStartTime -Value $s
Set-ItemProperty -Path $k -Name PauseQualityUpdatesEndTime -Value $e

# Active hours: Windows does not restart for updates inside them. Fixed, not "smart".
Set-ItemProperty -Path $k -Name SmartActiveHoursState -Value 0 -Type DWord
Set-ItemProperty -Path $k -Name ActiveHoursStart -Value $ActiveStart -Type DWord
Set-ItemProperty -Path $k -Name ActiveHoursEnd -Value $ActiveEnd -Type DWord

Get-ItemProperty -Path $k | Select-Object Pause*, ActiveHours*, SmartActiveHoursState | Format-List
"Paused until {0} (local). Next update window must come before then." -f $end.ToLocalTime()
