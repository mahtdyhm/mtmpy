@echo off
chcp 65001 >nul
echo ========================================================
echo    KinetiQ OS - Sports Vision AI Platform
echo    Offline-First & Zero-Config Athletic Scouting Engine
echo ========================================================
echo.
echo Activating Conda Environment (pull_ups)...
call "C:\Users\hp\miniconda3\condabin\conda.bat" activate pull_ups

echo.
echo Starting KinetiQ OS Web Server on http://localhost:8000 ...
echo Press Ctrl+C to stop the server.
echo.

python run_server.py
pause
