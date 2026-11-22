@echo off
rem Pembungkus WSL untuk landcover (lihat lc_wsl.py). Pakai sebagai LC_CORE_BIN dari Windows.
rem Python: .venv repo bila ada, selain itu "python" dari PATH.
setlocal
set "LC_PY=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%LC_PY%" set "LC_PY=python"
"%LC_PY%" "%~dp0lc_wsl.py" %*
exit /b %ERRORLEVEL%
