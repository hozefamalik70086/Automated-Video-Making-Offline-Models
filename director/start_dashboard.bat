@echo off
setlocal enabledelayedexpansion
title Director Control Room
cd /d "%~dp0"

REM ============================================================
REM  Director Control Room launcher
REM  Double-click this file to start the server and open the
REM  dashboard in your browser automatically.
REM  If the server is already running, it just opens the page.
REM ============================================================

REM ---- pick a Python that can run the app ----
set "PY="
if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\.venv-1\Scripts\python.exe" set "PY=%~dp0..\.venv-1\Scripts\python.exe"
if not defined PY set "PY=python"

REM ---- if the server is already running, just open it ----
powershell -NoProfile -Command "try{$r=Invoke-WebRequest -Uri 'http://127.0.0.1:8765/api/health' -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -eq 200){exit 0}}catch{exit 1}" >nul 2>&1
if "!errorlevel!"=="0" (
  echo Control Room is already running - opening the dashboard...
  start "" "http://127.0.0.1:8765/?v=2"
  exit /b 0
)

echo.
echo  Director Control Room  -  http://127.0.0.1:8765
echo  (keep this window open; closing it stops the server)
echo.
"%PY%" dashboard_server.py --open
echo.
echo Server stopped.
pause
