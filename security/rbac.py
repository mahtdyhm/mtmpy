"""
KinetiQ OS - Role-Based Access Control (RBAC)
سیستم تفکیک سطوح دسترسی برای مدیران، مربیان و ورزشکاران
"""

from typing import List
from fastapi import HTTPException, status
from db.models import User, UserRole


def require_roles(allowed_roles: List[str]):
    """گارد احراز سطح دسترسی بر اساس نقش‌های مجاز"""
    def role_checker(current_user: User) -> User:
        if not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="حساب کاربری شما غیرفعال شده است.",
            )
        # ادمین ارشد به همه جا دسترسی دارد
        if current_user.role == UserRole.SUPERADMIN:
            return current_user

        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="شما مجوز لازم برای دسترسی به این بخش را ندارید.",
            )
        return current_user

    return role_checker
