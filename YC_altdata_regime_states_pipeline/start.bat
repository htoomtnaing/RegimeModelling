@echo off
cd /d "%~dp0"
title AltData Regime States - %~dp0
echo %~dp0
echo.

if not exist ".venv\Scripts\activate" (
    echo [ERROR] No .venv found in this folder. Run setup.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment ...
call ".venv\Scripts\activate"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Could not activate the venv. Run setup.bat first.
    pause
    exit /b 1
)

echo.
echo Launching Jupyter Lab - a browser tab will open.
echo Open altdata_regime_states.ipynb and pick the
echo "AltData Regime States" kernel if prompted.
echo Press Ctrl+C in this window to stop the server.
echo.
jupyter lab
pause
