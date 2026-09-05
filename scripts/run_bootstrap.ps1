param(
    [string]$GdeltStart,
    [string]$GdeltEnd,
    [switch]$SkipGdelt,
    [switch]$SkipFred
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path .venv)) {
    python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install -e .

if (-not (Test-Path .env)) {
    & powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
}

$env:TI_DATA_ROOT = if ($env:TI_DATA_ROOT) { $env:TI_DATA_ROOT } else { Join-Path $HOME "TradingIntelligenceData" }

$args = @()
if ($SkipGdelt) { $args += "--skip-gdelt" }
if ($SkipFred) { $args += "--skip-fred" }
if ($GdeltStart) { $args += @("--gdelt-start", $GdeltStart) }
if ($GdeltEnd) { $args += @("--gdelt-end", $GdeltEnd) }

& .\.venv\Scripts\python.exe .\scripts\bootstrap_research_data.py @args
if ($LASTEXITCODE -ne 0) { throw "Research-data bootstrap failed with exit code $LASTEXITCODE" }
