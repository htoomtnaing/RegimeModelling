@echo off
REM -- Build a standalone virtual environment for this pipeline --
cd /d "%~dp0"
echo Creating virtual environment (.venv) ...
python -m venv .venv 2>nul || py -m venv .venv
if not exist ".venv\Scripts\activate.bat" (
  echo ERROR: could not create venv. Is Python on your PATH?  ^(try: py --version^)
  pause & exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo ============================================================
echo Setup complete.  Now run:   run.bat   (compute everything)
echo               or:   start.bat (open the notebook in Jupyter)
echo ============================================================
pause
