"""
KinetiQ OS - Core Structured Logging Engine
سیستم لاگ ساختاریافته کنسولی و فایلی با چرخشی بودن روزانه
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config.settings import settings


def setup_logger(name: str = "kinetiq") -> logging.Logger:
    """ایجاد یا بازیابی لاگر اختصاصی سیستم"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # هندلر کنسول
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # هندلر فایل چرخشی (حداکثر ۱۰ مگابایت، با ۵ نسخه پشتیبان)
    log_file = settings.LOGS_DIR / "kinetiq.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger("kinetiq")
