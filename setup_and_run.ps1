# setup_and_run.ps1: one-shot setup + full first pass.
#
# Run from the project folder in PowerShell:
#     powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1
#
# What it does, in order (stops at the first failure):
#   1. Creates .venv with uv using Python 3.11 (uv downloads 3.11 if you don't have it)
#   2. Installs requirements.txt into .venv
#   3. Runs the unit tests (pytest)
#   4. Runs scripts/preview_sequences.py (downloads gpt2-small on first run, ~500 MB)
#   5. Runs the first pass: 2 clauses x 200 trials, run name "first_pass"
#   6. Plots land in figures\first_pass\ as <date>_first_pass_<plot>.png
#
# Everything printed is also saved to logs\setup_and_run_<timestamp>.log

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot   # always run relative to the project folder

New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = "logs\setup_and_run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Start-Transcript -Path $log | Out-Null

# Helper: run a native command, fail loudly if its exit code isn't 0.
function Step($title, [scriptblock]$cmd) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $title (exit code $LASTEXITCODE)" }
}

try {
    # 0. uv must be installed. Install with:  winget install astral-sh.uv   (or: pip install uv)
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv not found. Install it with 'winget install astral-sh.uv' (or 'pip install uv'), open a new terminal, and rerun."
    }

    # Only create the venv if it doesn't exist yet, so the script is safe to rerun.
    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        Step "1/5 Create venv (Python 3.11)" { uv venv --python 3.11 .venv }
    } else {
        Write-Host "`n=== 1/5 .venv already exists, reusing it ===" -ForegroundColor Cyan
    }
    $py = ".\.venv\Scripts\python.exe"

    Step "2/5 Install requirements" { uv pip install --python $py -r requirements.txt }
    Step "    Versions" { & $py -c "import sys, torch, transformer_lens as tl, importlib.metadata as m; print('python', sys.version.split()[0]); print('torch', torch.__version__, '| cuda:', torch.cuda.is_available()); print('transformer_lens', m.version('transformer_lens'))" }

    Step "3/5 Unit tests" { & $py -m pytest tests -q }
    Step "4/5 Preview sequences" { & $py scripts\preview_sequences.py }
    Step "5/5 First pass" { & $py scripts\run_delta_experiment.py --run-name first_pass }

    Write-Host "`nAll done. Figures: figures\first_pass\   Raw results: results\first_pass\   Log: $log" -ForegroundColor Green
}
finally {
    Stop-Transcript | Out-Null
}
