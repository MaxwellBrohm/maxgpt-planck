<#
  start_detached.ps1: start a long job that survives the SSH session.

  Over SSH, a child started with Start-Process (even -WindowStyle Hidden) is killed the
  moment the session ends (measured 2026-09-21: a 240 s ping died after 4 lines). A process
  created through WMI (Win32_Process.Create) is not a child of the SSH session and keeps
  running. Output goes to a log file; poll it over SSH.

  From the Mac (cmd.exe is the SSH shell, so never nest PowerShell quoting through it):
    ssh ... $PLANCK_PC_HOST "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\<winuser>\planck-kit\pc\win\start_detached.ps1 -Script C:\Users\<winuser>\planck-kit\job.ps1 -Log C:\Users\<winuser>\planck-logs\job.log"

  Two forms:
    -Script <file.ps1> [-Log <file>]   runs the script in a fresh PowerShell; all streams to the log
                                       (default log: the script path with .log)
    -Command "<command line>" [-Log]   any command line, e.g. starting the runner by hand:
        -Command 'wsl.exe -d Ubuntu-24.04 --exec /bin/bash -lc "bash ~/planck/repo/pc/wsl/runner_start.sh"'
      (normally the PlanckRunner task starts the runner: Start-ScheduledTask PlanckRunner)
  Prints the new process id. Stop it later with: Stop-Process -Id <pid>
#>
param([string]$Script = "", [string]$Command = "", [string]$Log = "",
      [string]$WorkDir = $env:USERPROFILE)
$ErrorActionPreference = "Stop"
if ((-not $Script) -eq (-not $Command)) { throw "give exactly one of -Script <file.ps1> or -Command <line>" }

if ($Script) {
    $Script = (Resolve-Path $Script).Path
    if (-not $Log) { $Log = [IO.Path]::ChangeExtension($Script, ".log") }
    # *>> would write UTF-16 in Windows PowerShell 5.1; pipe every stream to UTF-8 instead.
    $inner = "& '$Script' *>&1 | Out-File -FilePath '$Log' -Append -Encoding utf8"
    $cmdLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -Command `"$inner`""
} elseif ($Log) {
    $cmdLine = "cmd.exe /c `"$Command >> `"$Log`" 2>&1`""
} else {
    $cmdLine = $Command
}
if ($Log) {
    $dir = Split-Path -Parent $Log
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Add-Content -Path $Log -Encoding utf8 -Value ("=== start_detached {0}: {1}" -f (Get-Date -Format s), $cmdLine)
}

$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $cmdLine; CurrentDirectory = $WorkDir }
if ($r.ReturnValue -ne 0) {
    Write-Error ("Win32_Process.Create failed, ReturnValue {0}" -f $r.ReturnValue)
    exit 1
}
"started pid {0}" -f $r.ProcessId
"command: $cmdLine"
if ($Log) { "log: $Log" }
