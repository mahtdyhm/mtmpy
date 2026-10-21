"""
KinetiQ OS - Server Launcher
نقطه شروع اجرای سرور بک‌اند با پورت و آدرس تعریف‌شده
"""

import sys
import threading
import time
import webbrowser
import uvicorn
from config.settings import settings
from api.server import app


def open_browser():
    """باز کردن خودکار مرورگر پیش‌فرض کاربر پس از آماده‌سازی سرور"""
    time.sleep(1.5)
    url = f"http://localhost:{settings.PORT}"
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("   KinetiQ OS - Sports Vision AI Platform")
    print(f"   Opening web interface on http://localhost:{settings.PORT} ...")
    print("=" * 60)

    # باز کردن خودکار داشبورد در مرورگر (فقط در سیستم دسکتاپ محلی)
    if sys.platform == "win32" and not getattr(sys, "frozen", False):
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        workers=1,
    )
