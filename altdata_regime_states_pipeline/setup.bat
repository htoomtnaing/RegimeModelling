@echo off
cd /d "%~dp0"
title setup - AltData Regime States
echo ============================================================
echo  AltData Regime States pipeline - one-time setup
echo  %~dp0
echo ============================================================
echo.

REM 1) Create the virtual environment (skip if it already exists).
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo.
        echo [ERROR] Could not create the venv. Make sure Python 3.11+ is on your PATH
        echo         ^(this bundle targets Python 3.14^).
        pause
        exit /b 1
    )
) else (
    echo Virtual environment .venv already exists - reusing it.
)

echo Activating virtual environment ...
call ".venv\Scripts\activate"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Could not activate the venv.
    pause
    exit /b 1
)

echo Upgrading pip ...
python -m pip install --upgrade pip

echo.
echo Installing dependencies from requirements.txt (wheels only) ...
pip install --only-binary=:all: -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Dependency install failed - see the messages above.
    echo         If a package has no Python 3.14 wheel, remove it from requirements.txt.
    echo         Only numpy / pandas / scipy / matplotlib / scikit-learn plus the jupyter
    echo         stack ^(jupyter, jupyterlab, ipykernel, nbconvert, nbformat^) are actually
    echo         required by the notebook.
    pause
    exit /b 1
)

echo.
echo Registering the Jupyter kernel "altdata-regime-states" ...
python -m ipykernel install --user --name altdata-regime-states --display-name "AltData Regime States"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Kernel registration failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete. Double-click start.bat to launch Jupyter Lab.
echo ============================================================
pause
