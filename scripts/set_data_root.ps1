param(
  [string]$DataRoot = "$env:USERPROFILE\TradingIntelligenceData"
)

if (-not (Test-Path $DataRoot)) {
  New-Item -ItemType Directory -Path $DataRoot -Force | Out-Null
}
$env:TI_DATA_ROOT = (Resolve-Path $DataRoot).Path
Write-Host "TI_DATA_ROOT set to $env:TI_DATA_ROOT"
