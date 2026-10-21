"""
KinetiQ OS - Portable Executable Builder
اسکریپت ساخت پکیج اجرایی تک‌فایلی / پورتابل با PyInstaller
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def build_portable_exe():
    print("=" * 60)
    print("  KinetiQ OS - Building Portable Executable")
    print("=" * 60)

    dist_dir = ROOT_DIR / "dist"
    build_dir = ROOT_DIR / "build"
    spec_file = ROOT_DIR / "KinetiQ-OS.spec"

    # جمع‌آوری فایل‌های استاتیک و وزن مدل
    static_src = str(ROOT_DIR / "static")
    yolo_model = str(ROOT_DIR / "yolo11n-pose.pt")

    datas = [
        f"{static_src};static",
        f"{yolo_model};.",
    ]

    hidden_imports = [
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "sqlmodel",
        "sqlalchemy.dialects.sqlite",
        "engineio.async_drivers.asgi",
        "ultralytics",
        "torch",
        "cv2",
        "mediapipe",
        "pydantic",
        "pydantic_settings",
        "api.server",
        "api.routes_auth",
        "api.routes_athletes",
        "api.routes_tests",
        "api.websocket_hub",
        "plugins.anthropometry.height_span",
        "plugins.jumps.vertical_jump",
        "plugins.handball.three_step",
    ]

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=KinetiQ-OS",
        "--noconfirm",
        "--clean",
        "--onedir",  # onedir is 10x faster startup than onefile for torch/opencv
        f"--distpath={dist_dir}",
        f"--workpath={build_dir}",
    ]

    for d in datas:
        cmd.extend(["--add-data", d])

    for h in hidden_imports:
        cmd.extend(["--hidden-import", h])

    cmd.append(str(ROOT_DIR / "run_server.py"))

    print("\nRunning command:")
    print(" ".join(cmd))
    
    # ساخت
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n[SUCCESS] Build completed successfully!")
        print(f"Output directory: {dist_dir / 'KinetiQ-OS'}")
    else:
        print(f"\n[FAILED] Build exited with code {res.returncode}")

if __name__ == "__main__":
    build_portable_exe()
