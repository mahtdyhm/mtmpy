"""
KinetiQ OS - Security & Authentication Engine
مدیریت هش کردن رمز عبور با Bcrypt و تولید/اعتبارسنجی توکن‌های JWT
"""

from datetime import datetime, timedelta
from typing import Optional, Any, Dict
from passlib.context import CryptContext
from jose import jwt, JWTError
from config.settings import settings
from core.logging import logger

# تنظیم کانتکست هش رمز عبور با استفاده از الگوریتم ایمن Bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """اعتبارسنجی رمز عبور خام در برابر هش ذخیره‌شده"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """تولید هش غیرقابل بازگشت برای رمز عبور"""
    return pwd_context.hash(password)


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
