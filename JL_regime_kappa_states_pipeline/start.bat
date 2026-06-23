@echo off
REM -- Launch Jupyter Lab to view regime_kappa_states.ipynb --
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo No virtual environment found. Run setup.bat first.
  pause & exit /b 1
)
call .venv\Scripts\activate.bat
jupyter lab regime_kappa_states.ipynb
