@echo off
chcp 65001 >nul
title KinetiQ OS - Portable Runner

echo ========================================================
echo       KinetiQ OS - Sports Vision AI Platform
echo       Zero-Config Portable Launcher for Any Laptop
echo ========================================================
echo.

set PYTHON_CMD=

:: ۱. بررسی وجود پایتون پورتابل در پوشه محلی برنامه
if exist "%~dp0python\python.exe" (
    set PYTHON_CMD="%~dp0python\python.exe"
    goto :FOUND
)
if exist "%~dp0venv\Scripts\python.exe" (
    set PYTHON_CMD="%~dp0venv\Scripts\python.exe"
    goto :FOUND
)

:: ۲. بررسی پایتون عمومی سیستم
where python >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    goto :FOUND
)

:: ۳. بررسی کندای محلی
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set PYTHON_CMD="%USERPROFILE%\miniconda3\python.exe"
    goto :FOUND
)
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set PYTHON_CMD="%USERPROFILE%\anaconda3\python.exe"
    goto :FOUND
)

:NOT_FOUND
echo [خطا] پایتون روی این سیستم یافت نشد!
echo لطفاً پایتون را نصب کنید یا پوشه پایتون پورتابل را کنار این فایل قرار دهید.
pause
exit /b 1

:FOUND
echo [OK] پایتون شناسایی شد: %PYTHON_CMD%
echo.
echo در حال اجرای سامانه و باز کردن خودکار داشبورد در مرورگر...
echo برای توقف برنامه کلیدهای Ctrl+C را فشار دهید.
echo.

%PYTHON_CMD% "%~dp0run_server.py"

pause
