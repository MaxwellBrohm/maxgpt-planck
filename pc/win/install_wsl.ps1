<#
  install_wsl.ps1: current WSL2 plus Ubuntu 24.04 (RUNBOOK step 6).
  Run in an ADMIN PowerShell:
    powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\<winuser>\planck-kit\pc\win\install_wsl.ps1
    ... -Location D:\WSL\Ubuntu-24.04     # put the distro's disk on another drive if C: is short
  If it says a reboot is needed: reboot, then run it again.

  After it finishes, ONE step needs Max at the PC's keyboard: open "Ubuntu 24.04" from the
  Start menu once and create the Linux user and password when it asks. Pick them yourself;
  no script or agent types a password. Then continue with RUNBOOK step 7.
  Do NOT install an NVIDIA driver inside Ubuntu: the Windows driver provides the GPU to WSL.
#>
#Requires -RunAsAdministrator
param([string]$Distro = "Ubuntu-24.04", [string]$Location = "")
$ErrorActionPreference = "Continue"
$env:WSL_UTF8 = "1"

"== updating WSL itself"
& wsl.exe --update
& wsl.exe --version
if ($LASTEXITCODE -ne 0) {
    "wsl.exe --version failed: WSL is not installed or too old. Installing the platform:"
    & wsl.exe --install --no-distribution
    "Reboot now, then run this script again."
    exit 1
}

$have = (& wsl.exe --list --quiet) -join "`n"
if ($have -match [regex]::Escape($Distro)) {
    "$Distro is already installed:"
    & wsl.exe --list --verbose
    exit 0
}

"== installing $Distro (no first launch; Max creates the user at the keyboard)"
$wslArgs = @("--install", "-d", $Distro, "--no-launch")
if ($Location) { $wslArgs += @("--location", $Location) }
& wsl.exe @wslArgs
if ($LASTEXITCODE -ne 0) {
    "install returned $LASTEXITCODE. If it asks for a reboot, reboot and run again."
    exit $LASTEXITCODE
}
& wsl.exe --set-default $Distro
& wsl.exe --list --verbose
""
"Next (Max, at the keyboard): Start menu > $Distro, create your Linux user. Then RUNBOOK step 7."
