@echo off
REM -- Execute the notebook end-to-end with live per-stage progress (regenerates outputs\) --
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo No virtual environment found. Run setup.bat first.
  pause & exit /b 1
)
call .venv\Scripts\activate.bat
if not exist "regime_metrics_states.ipynb" python build_nb_regime_metrics_states.py
echo Executing the notebook (live progress below; ~2-3 min) ...
echo ------------------------------------------------------------
python run_notebook.py
echo ------------------------------------------------------------
echo Done. Outputs written to outputs\
pause
