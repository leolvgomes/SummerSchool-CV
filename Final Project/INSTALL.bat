@echo off
setlocal
cd /d "%~dp0"
py -3.12 --version >nul 2>&1
if errorlevel 1 (
 echo Install Python 3.12 64-bit from python.org, with the Python launcher.
 pause
 exit /b 1
)
py -3.12 -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo Installation complete. Open RUN_VIDEO.bat or RUN_WEBCAM.bat.
pause
exit /b 0
:fail
echo Installation failed. Check the error above and your internet connection.
pause
exit /b 1
