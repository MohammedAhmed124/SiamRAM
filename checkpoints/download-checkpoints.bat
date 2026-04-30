@echo off
setlocal
set SCRIPT_DIR=%~dp0

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%SCRIPT_DIR%download_checkpoints.py" %*
  exit /b %errorlevel%
)

where python >nul 2>nul
if %errorlevel%==0 (
  python "%SCRIPT_DIR%download_checkpoints.py" %*
  exit /b %errorlevel%
)

echo Python was not found. Install Python 3 and retry.
exit /b 1
