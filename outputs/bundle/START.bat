@echo off
REM NIHHA Dental Dataset Downloader - one-click launcher
REM Double-click this file. It hands off to START.ps1 in the same folder.

setlocal
set "HERE=%~dp0"
cd /d "%HERE%"

where pwsh >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%HERE%START.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%START.ps1"
)

echo.
echo Press any key to close this window...
pause >nul
endlocal
