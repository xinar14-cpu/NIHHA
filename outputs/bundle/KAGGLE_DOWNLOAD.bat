@echo off
REM NIHHA Kaggle-only downloader
REM Double-click. Pulls 8 Kaggle datasets (~30k photos) into D:\DATASETS by default.

setlocal
set "HERE=%~dp0"
cd /d "%HERE%"

where pwsh >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%HERE%KAGGLE_DOWNLOAD.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%KAGGLE_DOWNLOAD.ps1"
)

echo.
echo Press any key to close this window...
pause >nul
endlocal
