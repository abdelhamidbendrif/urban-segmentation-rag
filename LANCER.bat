@echo off
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Installer l'environnement comme indique dans README.md.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0interface.py"
if errorlevel 1 pause
