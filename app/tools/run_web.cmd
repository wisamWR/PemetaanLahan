@echo off
rem Satu perintah untuk UI web (lihat run_web.py): bangun frontend bila perlu, sajikan API + web di port 8000.
rem Contoh: app\tools\run_web.cmd --backend cli --open
setlocal
set "RW_PY=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%RW_PY%" set "RW_PY=python"
"%RW_PY%" "%~dp0run_web.py" %*
exit /b %ERRORLEVEL%
