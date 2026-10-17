"""
KinetiQ OS - Configuration & Settings Manager
مدیریت متمرکز تنظیمات، امنیت، مسیرها و وضعیت هیبریدی (آنلاین/آفلاین)
"""

from __future__ import annotations
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# دایرکتوری ریشه پروژه
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """تنظیمات سامانه KinetiQ"""

    # اطلاعات پایه برنامه
    APP_NAME: str = "KinetiQ OS"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # سرور و پورت
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # مسیرهای داده و دایرکتوری‌ها
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = BASE_DIR / "data" / "output"
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    LOGS_DIR: Path = BASE_DIR / "data" / "logs"

    # دیتابیس لوکال SQLite
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'kinetiq.db'}"

    # امنیت و احراز هویت
    SECRET_KEY: str = os.getenv("KINETIQ_SECRET_KEY", "kinetiq-super-secret-key-32bytes-min-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # ۷ روز اعتبار توکن برای سهولت در سالن ورزشی

    # حالت هیبریدی هوش مصنوعی (Online / Offline)
    DEFAULT_AI_MODE: str = "offline"  # 'offline' (محلی) یا 'online' (پردازش ابری)
    ALLOW_ONLINE_FALLBACK: bool = True
    ONLINE_GATEWAY_URL: str = "https://gateway.vlm.run/v1/openai"
    ONLINE_API_KEY: str = os.getenv("VLMRUN_API_KEY", "")

    # نرخ فریم و پردازش
    TARGET_FPS: int = 30
    MAX_BUFFER_FRAMES: int = 300

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


# نمونه سراسری تنظیمات
settings = Settings()

# اطمینان از وجود پوشه‌های پایه
for directory in [settings.DATA_DIR, settings.OUTPUT_DIR, settings.UPLOADS_DIR, settings.LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
