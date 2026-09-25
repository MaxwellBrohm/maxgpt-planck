<#
  setup_power.ps1: the PC never sleeps or hibernates on AC power (RUNBOOK step 3).
  Run in an ADMIN PowerShell:
    powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\<winuser>\planck-kit\pc\win\setup_power.ps1
  The monitor may still turn off (20 min); that does not pause jobs.
  Undo: Settings > System > Power, or powercfg /restoredefaultschemes.
#>
#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"

powercfg /change standby-timeout-ac 0      # sleep: never
powercfg /change hibernate-timeout-ac 0    # hibernate: never
powercfg /change disk-timeout-ac 0         # disks: never spin down
powercfg /change monitor-timeout-ac 20     # screen off after 20 min (harmless)

# PCI Express link state power management off on AC, so the GPU link never naps.
# (powercfg is a native command: failures show in $LASTEXITCODE, not as exceptions)
powercfg /setacvalueindex SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0
if ($LASTEXITCODE -ne 0) { Write-Warning "could not set PCIe ASPM (alias missing on this build?)" }
powercfg /setactive SCHEME_CURRENT

"== result (Current AC Power Setting Index 0x00000000 = never)"
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String "Current AC"
powercfg /query SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE | Select-String "Current AC"
powercfg /query SCHEME_CURRENT SUB_PCIEXPRESS ASPM | Select-String "Current AC"
