$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $PSScriptRoot

& (Join-Path $PSScriptRoot "stop_managed_agent_processes.ps1")

$venvDir = Join-Path $workspaceRoot "agent_server\.venv"
$pyvenvConfigPath = Join-Path $venvDir "pyvenv.cfg"
if ((Test-Path $venvDir) -and -not (Test-Path $pyvenvConfigPath)) {
    @'
import pathlib
import shutil
import sys
import time

target = pathlib.Path(sys.argv[1])
for _ in range(5):
    if not target.exists():
        break
    shutil.rmtree(target, ignore_errors=True)
    time.sleep(0.5)

if target.exists():
    raise SystemExit(f"Failed to remove invalid virtualenv at {target}")
'@ | python - $venvDir

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Push-Location $workspaceRoot
try {
    uv sync --directory agent_server --python 3.12.2
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    npm --prefix client install --include=dev
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
