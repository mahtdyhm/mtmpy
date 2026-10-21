"""
KinetiQ OS - Distribution Pack Builder
تولید پکیج آماده تحویل پورتابل در پوشه pack-exe-win به همراه فایل فشرده ZIP و راهنما
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

# تنظیم خروجی یونیکد برای ویندوز
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
PACK_DIR = ROOT_DIR / "pack-exe-win"
VERSION = "1.0.0"
DIST_NAME = f"KinetiQ-OS-v{VERSION}"
DIST_FOLDER = PACK_DIR / DIST_NAME


def copy_clean_dir(src: Path, dst: Path):
    """کپی کردن پوشه بدون فایل‌های کش __pycache__"""
    def ignore_patterns(folder, files):
        ignored = []
        for f in files:
            if f == "__pycache__" or f.endswith(".pyc") or f == ".pytest_cache":
                ignored.append(f)
        return ignored

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore_patterns)


def build_pack():
    print("=" * 60)
    print(f"  Building KinetiQ OS Portable Package (v{VERSION})")
    print("=" * 60)

    # ساخت دایرکتوری هدف
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    if DIST_FOLDER.exists():
        shutil.rmtree(DIST_FOLDER)
    DIST_FOLDER.mkdir(parents=True, exist_ok=True)

    # ۱. کپی پوشه‌های اصلی پروژه
    folders_to_copy = ["api", "config", "core", "db", "plugins", "security", "static"]
    for folder in folders_to_copy:
        src_path = ROOT_DIR / folder
        dst_path = DIST_FOLDER / folder
        if src_path.is_dir():
            print(f"-> Copying {folder}...")
            copy_clean_dir(src_path, dst_path)

    # ۲. ساخت پوشه data با زیرپوشه‌های خالی استاندارد
    data_dir = DIST_FOLDER / "data"
    for sub in ["output", "uploads", "logs"]:
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    # ۳. کپی فایل‌های اصلی
    files_to_copy = [
        "run_server.py",
        "yolo11n-pose.pt",
        "README.md",
    ]
    for file_name in files_to_copy:
        src_file = ROOT_DIR / file_name
        if src_file.is_file():
            print(f"-> Copying {file_name}...")
            shutil.copy2(src_file, DIST_FOLDER / file_name)

    # ۴. ساخت requirements.txt دقیق و بهینه
    req_file = DIST_FOLDER / "requirements.txt"
    req_content = """# KinetiQ OS - Dependencies
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
sqlmodel>=0.0.16
pydantic>=2.6.0
pydantic-settings>=2.2.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.9
ultralytics>=8.1.0
torch>=2.0.0
torchvision>=0.15.0
mediapipe>=0.10.9
opencv-python>=4.8.0
numpy>=1.24.0
websockets>=12.0
aiofiles>=23.2.0
"""
    req_file.write_text(req_content, encoding="utf-8")
    print("-> Generated requirements.txt")

    # ۵. ساخت لانچر کاربرپسند ویندوز (Launch-KinetiQ.bat)
    launcher_bat = DIST_FOLDER / "Launch-KinetiQ.bat"
    launcher_content = """@echo off
chcp 65001 >nul
title KinetiQ OS v1.0.0 - Sports Vision AI Platform

echo ======================================================================
echo          KinetiQ OS - Sports Vision AI Platform (v1.0.0)
echo ======================================================================
echo.
echo در حال راه‌اندازی سرور محلی و باز شدن خودکار داشبورد در مرورگر...
echo لطفاً چند لحظه صبر کنید...
echo.
echo آدرس داشبورد: http://localhost:8000
echo نام کاربری پیش‌فرض: admin
echo کلمه عبور پیش‌فرض: admin123
echo.
echo برای بستن برنامه، این پنجره را ببندید یا کلیدهای Ctrl+C را بزنید.
echo ======================================================================
echo.

python run_server.py

if %errorlevel% neq 0 (
    echo.
    echo ======================================================================
    echo [خطا در اجرا] پایتون یا کتابخانه‌ها روی این سیستم یافت نشدند!
    echo لطفاً در صورت عدم نصب کتابخانه‌ها، فایل Install-Dependencies.bat را اجرا کنید.
    echo ======================================================================
    pause
)
"""
    launcher_bat.write_text(launcher_content, encoding="utf-8")
    print("-> Generated Launch-KinetiQ.bat")

    # ۶. ساخت فایل نصب خودکار وابستگی‌ها (Install-Dependencies.bat)
    install_bat = DIST_FOLDER / "Install-Dependencies.bat"
    install_content = """@echo off
chcp 65001 >nul
title KinetiQ OS - Dependency Installer

echo ======================================================================
echo   KinetiQ OS - نصب خودکار پیش‌نیازها و کتابخانه‌های هوش مصنوعی
echo ======================================================================
echo.
echo در حال نصب کتابخانه‌ها از requirements.txt (ممکن است چند دقیقه زمان ببرد)...
echo.

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ======================================================================
if %errorlevel% equ 0 (
    echo [موفق] تمام کتابخانه‌ها با موفقیت نصب شدند!
    echo اکنون می‌توانید با دابل‌کلیک روی Launch-KinetiQ.bat برنامه را اجرا کنید.
) else (
    echo [خطا] در حین نصب کتابخانه‌ها خطایی رخ داد. اتصال اینترنت خود را بررسی کنید.
)
echo ======================================================================
pause
"""
    install_bat.write_text(install_content, encoding="utf-8")
    print("-> Generated Install-Dependencies.bat")

    # ۷. فایل راهنمای فارسی (راهنمای_اجرا.txt)
    guide_txt = DIST_FOLDER / "راهنمای_اجرا.txt"
    guide_content = f"""======================================================================
  راهنمای سریع اجرای KinetiQ OS (نسخه {VERSION})
======================================================================

سلام! برای اجرای این سامانه روی لپ‌تاپ خود کافیست مراحل زیر را انجام دهید:

مرحله ۱: (فقط بار اول)
- مطمئن شوید پایتون (Python 3.10 یا 3.11) روی ویندوز نصب است و تیک "Add Python to PATH" را در هنگام نصب زده‌اید.
- اگر اولین بار است که برنامه را روی این لپ‌تاپ اجرا می‌کنید، روی فایل زیر دابل‌کلیک کنید تا کتابخانه‌ها نصب شوند:
    Install-Dependencies.bat

مرحله ۲: (اجرای برنامه)
- روی فایل زیر دابل‌کلیک کنید:
    Launch-KinetiQ.bat

- بلافاصله پنجره مشکی سرور باز شده و ظرف ۲ ثانیه مرورگر لپ‌تاپ شما به طور خودکار باز می‌شود و وارد برنامه می‌شوید!
- آدرس ورود در مرورگر: http://localhost:8000
- نام کاربری پیش‌فرض: admin
- کلمه عبور پیش‌فرض: admin123

نکات مهم:
۱. برای فعال‌سازی دوربین در بخش آزمون‌ها، دکمه «فعال‌سازی دوربین» را بزنید و در بالای مرورگر روی Allow کلیک کنید.
۲. برنامه کاملاً آفلاین بوده و نیازی به اینترنت ندارد (مدل هوش مصنوعی YOLO11 در بسته قرار داده شده است).
۳. برای خروج، پنجره مشکی را ببندید.
======================================================================
"""
    guide_txt.write_text(guide_content, encoding="utf-8")
    print("-> Generated راهنمای_اجرا.txt")

    # ۸. فایل مشخصات نسخه VERSION.json
    version_file = DIST_FOLDER / "VERSION.json"
    version_info = f"""{{
  "app_name": "KinetiQ OS",
  "version": "{VERSION}",
  "release_type": "Portable Windows Package",
  "ai_model": "yolo11n-pose.pt",
  "target_platform": "Windows 10 / 11 (x64)",
  "build_date": "2026-10-19"
}}
"""
    version_file.write_text(version_info, encoding="utf-8")
    print("-> Generated VERSION.json")

    # ۹. فشرده‌سازی خودکار و تولید فایل ZIP آماده ارسال در pack-exe-win
    zip_output_path = PACK_DIR / f"{DIST_NAME}-Portable"
    print("\n-> Creating ZIP archive for easy sharing...")
    shutil.make_archive(str(zip_output_path), "zip", PACK_DIR, DIST_NAME)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Package created in: {DIST_FOLDER}")
    print(f"[SUCCESS] Ready-to-send ZIP archive: {zip_output_path}.zip")
    print("=" * 60)


if __name__ == "__main__":
    build_pack()
