$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$scriptsDir = Join-Path $workspaceRoot "agent_server\.venv\Scripts"
$managedExecutables = @(
    (Join-Path $scriptsDir "uvicorn.exe"),
    (Join-Path $scriptsDir "python.exe")
)

$managedProcesses = Get-CimInstance Win32_Process |
    Where-Object {
        $_.ExecutablePath -and ($managedExecutables -contains $_.ExecutablePath)
    }

foreach ($process in $managedProcesses) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host ("Stopped {0} existing managed agent process(es)." -f $managedProcesses.Count)
