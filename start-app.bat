@echo off
setlocal

cd /d "%~dp0"
title AI Video Generator - Start Application

if /I "%~1"=="--no-pause" (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-app.ps1" -NoBrowser
) else (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-app.ps1"
)
set "START_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%START_EXIT_CODE%"=="0" (
    echo Application startup failed. Review the error above.
) else (
    echo The application is running at http://127.0.0.1:1420/
)

if /I not "%~1"=="--no-pause" pause
exit /b %START_EXIT_CODE%
