# TradingIntelligence automatic GitHub sync
# Repository: C:\Users\willi\TradingIntelligence

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\willi\TradingIntelligence"
$LogDir = Join-Path $Repo "logs"
$LogFile = Join-Path $LogDir "auto_sync.log"

# Make sure the log directory exists
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$Timestamp] $Message"
}

try {
    Set-Location $Repo

    Write-Log "---- Automatic sync started ----"

    # Safety checks
    if (-not (Test-Path (Join-Path $Repo ".git"))) {
        throw "Git repository not found at $Repo"
    }

    $Remote = (git remote get-url origin 2>&1).ToString().Trim()

    if ($Remote -notmatch "github\.com[:/]Willyszn/TradingIntelligence(\.git)?$") {
        throw "Unexpected GitHub remote: $Remote"
    }

    $Branch = (git branch --show-current 2>&1).ToString().Trim()

    if ($Branch -ne "main") {
        throw "Repository is on '$Branch', not 'main'. No changes pushed."
    }

    # Refresh remote information
    git fetch origin main --quiet

    # Refuse to operate if local main has diverged from origin/main.
    $AheadBehind = git rev-list --left-right --count "origin/main...HEAD"

    if ($LASTEXITCODE -ne 0) {
        throw "Unable to determine local/remote relationship."
    }

    $Parts = $AheadBehind -split "\s+"

    if ($Parts.Count -lt 2) {
        throw "Could not parse ahead/behind status."
    }

    $Behind = [int]$Parts[0]
    $Ahead  = [int]$Parts[1]

    if ($Behind -gt 0 -and $Ahead -gt 0) {
        throw "Repository has diverged from origin/main. Manual intervention required."
    }

    if ($Behind -gt 0 -and $Ahead -eq 0) {
        throw "Local repository is behind origin/main. No automatic pull performed."
    }

    # Check for local changes.
    $Status = @(git status --porcelain)

    if ($Status.Count -eq 0) {
        Write-Log "No local changes. Nothing to commit."

        # Push any local commits that haven't reached GitHub.
        if ($Ahead -gt 0) {
            git push origin main

            if ($LASTEXITCODE -ne 0) {
                throw "git push failed."
            }

            Write-Log "Pushed $Ahead existing commit(s) to GitHub."
        }

        Write-Log "---- Automatic sync finished ----"
        exit 0
    }

    Write-Log "Changes detected: $($Status.Count) changed path(s)."

    # Stage tracked/untracked files according to .gitignore.
    git add -A

    if ($LASTEXITCODE -ne 0) {
        throw "git add failed."
    }

    # Show what will actually be committed in the log.
    $Staged = @(git diff --cached --name-status)

    if ($Staged.Count -eq 0) {
        Write-Log "Nothing staged after git add. Stopping."
        exit 0
    }

    foreach ($Line in $Staged) {
        Write-Log "Staged: $Line"
    }

    # Commit
    $CommitMessage = "Auto-sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    git commit -m $CommitMessage

    if ($LASTEXITCODE -ne 0) {
        throw "git commit failed."
    }

    Write-Log "Commit created: $CommitMessage"

    # Push
    git push origin main

    if ($LASTEXITCODE -ne 0) {
        throw "git push failed."
    }

    Write-Log "Successfully pushed to GitHub."
    Write-Log "---- Automatic sync finished ----"
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    Write-Log "---- Automatic sync stopped ----"
    exit 1
}