"""
KinetiQ OS - Database Session & Initialization
اتصال پایگاه داده، نشست‌ها و ساخت خودکار جداول و کاربر پیش‌فرض ادمین
"""

from sqlmodel import SQLModel, create_engine, Session, select
from config.settings import settings
from core.logging import logger

# اتصال به دیتابیس SQLite لوکال
connect_args = {"check_same_thread": False}
engine = create_engine(settings.DATABASE_URL, echo=settings.DEBUG, connect_args=connect_args)


def init_db() -> None:
    """ساخت خودکار تمام جدول‌ها و مقداردهی اولیه ادمین ارشد"""
    from db.models import User, UserRole
    from security.auth import get_password_hash

    SQLModel.metadata.create_all(engine)
    logger.info("ساختار جداول پایگاه داده در SQLite با موفقیت بررسی و ایجاد شد.")

    with Session(engine) as session:
        # ایجاد ادمین پیش‌فرض در صورت عدم وجود
        admin = session.exec(select(User).where(User.username == "admin")).first()
        if not admin:
            admin_user = User(
                username="admin",
                hashed_password=get_password_hash("admin123"),
                full_name="مدیر ارشد سامانه",
                role=UserRole.SUPERADMIN,
                is_active=True,
            )
            session.add(admin_user)
            session.commit()
            logger.info("کاربر پیش‌فرض SuperAdmin با نام کاربری 'admin' و رمز 'admin123' ایجاد شد.")


def get_db():
    """Generator دریافت نشست دیتابیس برای تزریق وابستگی در FastAPI"""
    with Session(engine) as session:
        yield session
