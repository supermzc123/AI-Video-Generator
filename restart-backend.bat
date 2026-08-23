@echo off
setlocal

cd /d "%~dp0"
title AI Video Generator - Restart Backend

echo Restarting AI Video Generator backend...
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\restart-api.ps1"
set "RESTART_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%RESTART_EXIT_CODE%"=="0" (
    echo Backend restart failed. Review the error above.
) else (
    echo Backend restart completed successfully.
    echo This script only restarts the API. Use start-app.bat to open the web interface.
)

if /I not "%~1"=="--no-pause" pause
exit /b %RESTART_EXIT_CODE%
