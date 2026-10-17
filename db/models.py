"""
KinetiQ OS - Database Models
موجودیت‌های پایگاه داده برای کاربران، ورزشکاران، سشن‌های تست و رکوردهای متریک
"""

from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class UserRole:
    SUPERADMIN = "superadmin"
    COACH = "coach"
    ATHLETE = "athlete"


class User(SQLModel, table=True):
    """کاربران سامانه با نقش‌های امنیتی مختلف"""
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str
    full_name: str
    role: str = Field(default=UserRole.COACH)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=now_utc)

    # روابط
    sessions: List["TestSession"] = Relationship(back_populates="coach")


class Athlete(SQLModel, table=True):
    """پرونده جامع بیومتریک و استعدادیابی ورزشکار"""
    __tablename__ = "athletes"

    id: Optional[int] = Field(default=None, primary_key=True)
    first_name: str = Field(index=True)
    last_name: str = Field(index=True)
    national_id: Optional[str] = Field(default=None, unique=True, index=True)
    gender: str = Field(default="male")  # 'male' یا 'female'
    birth_date: Optional[str] = None     # YYYY-MM-DD
    sport: str = Field(default="handball", index=True)
    position: Optional[str] = None       # e.g., 'goalkeeper', 'wing', 'pivot', 'back'
    standing_height_cm: Optional[float] = None
    arm_span_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    photo_path: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=now_utc)

    # روابط
    test_sessions: List["TestSession"] = Relationship(back_populates="athlete")


class MetricLog(SQLModel, table=True):
    """رکوردهای ثبت‌شده کینماتیک و آماری برای هر سشن"""
    __tablename__ = "metric_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="test_sessions.id", index=True)
    metric_name: str = Field(index=True)
    metric_value: float
    metric_unit: str
    details_json: Optional[str] = None  # داده‌های تکمیلی فریم‌به‌فریم به‌صورت JSON

    # روابط
    session: Optional["TestSession"] = Relationship(back_populates="metrics")


class TestSession(SQLModel, table=True):
    """جلسه ثبت شده از یک آزمون ورزشی"""
    __tablename__ = "test_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    test_id: str = Field(index=True)  # مثلا 'cmj_jump', 'height_span', 'handball_3step'
    athlete_id: int = Field(foreign_key="athletes.id", index=True)
    coach_id: Optional[int] = Field(default=None, foreign_key="users.id")
    status: str = Field(default="completed")  # 'completed', 'in_progress', 'cancelled'
    ai_mode: str = Field(default="offline")   # 'offline' یا 'online'
    raw_video_path: Optional[str] = None
    rendered_video_path: Optional[str] = None
    summary_report_path: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=now_utc)

    # روابط
    athlete: Optional[Athlete] = Relationship(back_populates="test_sessions")
    coach: Optional[User] = Relationship(back_populates="sessions")
    metrics: List[MetricLog] = Relationship(back_populates="session")
