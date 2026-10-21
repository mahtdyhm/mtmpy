"""
KinetiQ OS - Security & Authentication Engine
مدیریت هش کردن رمز عبور با Bcrypt و تولید/اعتبارسنجی توکن‌های JWT
"""

import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Any, Dict
from jose import jwt, JWTError
from config.settings import settings
from core.logging import logger


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """اعتبارسنجی رمز عبور خام در برابر هش ذخیره‌شده با bcrypt مستقیم"""
    try:
        pw_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pw_bytes, hash_bytes)
    except Exception as e:
        logger.error(f"خطا در بررسی رمز عبور: {e}")
        return False


def get_password_hash(password: str) -> str:
    """تولید هش استاندارد bcrypt بدون وابستگی به باگ‌های قدیمی passlib"""
    pw_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """تولید توکن JWT رمزنگاری‌شده"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """رمزگشایی و اعتبارسنجی توکن JWT"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"خطای اعتبارسنجی توکن امنیتی JWT: {e}")
        return None
