@echo off
setlocal

cd /d "C:\Users\willi\TradingIntelligence"

powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "C:\Users\willi\TradingIntelligence\scripts\auto_sync.ps1"

exit /b %ERRORLEVEL%