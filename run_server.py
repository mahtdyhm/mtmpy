"""
KinetiQ OS - Server Launcher
نقطه شروع اجرای سرور بک‌اند با پورت و آدرس تعریف‌شده
"""

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,
    )
