param(
    [switch]$PersistUserKey
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example."
}

$key = Read-Host "Enter your FRED API key"
if ([string]::IsNullOrWhiteSpace($key)) {
    throw "FRED API key cannot be empty."
}

# Write only the local .env file; .gitignore prevents accidental commits.
$lines = Get-Content .env | Where-Object { $_ -notmatch '^FRED_API_KEY=' }
$lines += "FRED_API_KEY=$key"
Set-Content -Path .env -Value $lines

$env:FRED_API_KEY = $key

if ($PersistUserKey) {
    [Environment]::SetEnvironmentVariable("FRED_API_KEY", $key, "User")
    Write-Host "FRED_API_KEY also saved as a Windows User environment variable."
}

Write-Host "FRED_API_KEY configured for this PowerShell session."
