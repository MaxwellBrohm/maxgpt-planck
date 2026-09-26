<#
  register_runner_task.ps1: a Task Scheduler task that starts the Planck queue runner in WSL
  at boot, with no stored password (RUNBOOK step 11).

  Run in an ADMIN PowerShell:
    powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\<winuser>\planck-kit\pc\win\register_runner_task.ps1 -LinuxUser max
    ... -AtLogon        # fallback if the boot task cannot start WSL (see below)

  What the task runs, in the foreground (the task's wsl.exe process is what keeps the WSL VM
  alive, so it must not exit while the runner runs):
    wsl.exe -d Ubuntu-24.04 -u <LinuxUser> --exec /bin/bash -lc "bash ~/planck/repo/pc/wsl/runner_start.sh"

  Boot mode uses LogonType S4U: "run whether the user is logged on or not" WITHOUT storing
  the Windows password. Some WSL builds refuse to start from such a session. Test it: reboot,
  do NOT log in, and after 3 minutes check from the Mac that ~/planck/logs/runner.log in WSL
  has a new "boot" line (or that a heartbeat arrived). If not, re-run with -AtLogon: the task
  then starts when Max logs in, so a forced reboot waits for him (the missing heartbeat
  shows it). No autologon is configured: that would store a password.

  Settings that matter: no execution time limit (the default 72 h would kill the runner),
  restart every minute on failure, start when available, ignore a second instance.
  Start it now without rebooting:   Start-ScheduledTask -TaskName PlanckRunner
  Remove it:                        Unregister-ScheduledTask -TaskName PlanckRunner -Confirm:$false
#>
#Requires -RunAsAdministrator
param([string]$LinuxUser = "", [string]$Distro = "Ubuntu-24.04", [switch]$AtLogon,
      [string]$TaskName = "PlanckRunner")
$ErrorActionPreference = "Stop"
$env:WSL_UTF8 = "1"

$distros = (& wsl.exe --list --quiet) -join "`n"
if ($distros -notmatch [regex]::Escape($Distro)) { throw "WSL distro $Distro not found (RUNBOOK step 6)" }

$wsl = Join-Path $env:WINDIR "System32\wsl.exe"
$userPart = ""
if ($LinuxUser) { $userPart = " -u $LinuxUser" }
$argLine = ('-d {0}{1} --exec /bin/bash -lc "bash ~/planck/repo/pc/wsl/runner_start.sh"' -f $Distro, $userPart)

# Over SSH, USERDOMAIN reads WORKGROUP, which maps to no account (measured 2026-09-25); the
# token's own name is always right (PCNAME\user).
$me = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $wsl -Argument $argLine
if ($AtLogon) {
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $me
    $principal = New-ScheduledTaskPrincipal -UserId $me -LogonType Interactive -RunLevel Limited
} else {
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId $me -LogonType S4U -RunLevel Limited
}
$trigger.Delay = "PT1M"         # let networking and the GPU driver come up first
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force `
    -Description "MaxGPT-Planck GPU queue runner in WSL (maxgpt-planck/pc/RUNBOOK.txt)" | Out-Null

$mode = "boot (S4U)"
if ($AtLogon) { $mode = "logon of $me" }
"registered task '$TaskName' at $mode"
"  $wsl $argLine"
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State | Format-List
