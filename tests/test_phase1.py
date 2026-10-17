"""
KinetiQ OS - Unit Tests for Phase 1
تست‌های جامع اعتبارسنجی فاز ۱ (تسک‌های ۱.۱ تا ۱.۷)
"""

import pytest
import uuid
from fastapi.testclient import TestClient
from sqlmodel import Session, select
from config.settings import settings
from db.session import engine, init_db
from db.models import User, Athlete, UserRole
from security.auth import get_password_hash, verify_password, create_access_token, decode_access_token
from core.registry import BaseSportTest, TestRegistry
from core.schemas import SkeletonFrame, MetricSummary
from api.server import app

client = TestClient(app)


def setup_module():
    """راه‌اندازی دیتابیس تست"""
    init_db()


def test_1_1_and_1_5_config_and_directories():
    """تست تسک ۱.۱ و ۱.۵: بررسی تنظیمات و ساخته شدن دایرکتوری‌ها"""
    assert settings.DATA_DIR.exists()
    assert settings.OUTPUT_DIR.exists()
    assert settings.UPLOADS_DIR.exists()
    assert settings.LOGS_DIR.exists()
    assert settings.APP_NAME == "KinetiQ OS"


def test_1_2_plugin_registry():
    """تست تسک ۱.۲: رجیستری ماژولار آزمون‌های ورزشی"""
    registry = TestRegistry()

    class MockJumpTest(BaseSportTest):
        test_id = "mock_jump"
        title = "Mock Jump Test"
        category = "jumps"
        description = "تست آزمایشی پرش"

        def start(self):
            self.is_active = True

        def process_frame(self, frame: SkeletonFrame):
            return {"height": 50.0}

        def finish(self) -> MetricSummary:
            return MetricSummary(
                test_id=self.test_id,
                athlete_id=self.athlete_id,
                primary_metric_name="jump_height",
                primary_metric_value=50.0,
                primary_metric_unit="cm",
            )

    # ثبت پلاگین
    registry.register(MockJumpTest)
    assert "mock_jump" in [t["test_id"] for t in registry.list_available_tests()]

    # ساخت نمونه و اجرای چرخه تست
    instance = registry.create_instance("mock_jump", athlete_id=1)
    instance.start()
    assert instance.is_active is True
    summary = instance.finish()
    assert summary.primary_metric_value == 50.0

    # حذف پلاگین
    registry.unregister("mock_jump")
    assert registry.get_test_class("mock_jump") is None


def test_1_3_and_1_4_auth_and_db():
    """تست تسک ۱.۳ و ۱.۴: دیتابیس، هش رمز و توکن‌های JWT"""
    password = "secret_password_123"
    hashed = get_password_hash(password)
    assert verify_password(password, hashed) is True
    assert verify_password("wrong_password", hashed) is False

    # تست تولید و دیکود توکن
    token = create_access_token({"sub": "admin", "role": "superadmin"})
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "admin"
    assert payload["role"] == "superadmin"


def test_1_4_api_auth_login():
    """تست لاگین ادمین از طریق API"""
    response = client.post(
        "/api/auth/login",
        data={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == UserRole.SUPERADMIN


def test_1_4_athlete_crud_with_auth():
    """تست کامل مدیریت پرونده ورزشکاران با توکن احراز هویت"""
    # لاگین جهت دریافت توکن
    login_res = client.post(
        "/api/auth/login",
        data={"username": "admin", "password": "admin123"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # ایجاد ورزشکار جدید
    new_athlete = {
        "first_name": "علی",
        "last_name": "محمدی",
        "national_id": f"ID_{uuid.uuid4().hex[:8]}",
        "gender": "male",
        "sport": "handball",
        "position": "wing",
        "standing_height_cm": 185.0,
        "arm_span_cm": 192.0,
        "weight_kg": 82.5,
    }
    create_res = client.post("/api/athletes", json=new_athlete, headers=headers)
    assert create_res.status_code == 201
    athlete_data = create_res.json()
    athlete_id = athlete_data["id"]
    assert athlete_data["first_name"] == "علی"

    # بازیابی اطلاعات
    get_res = client.get(f"/api/athletes/{athlete_id}", headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["arm_span_cm"] == 192.0

    # جستجو در لیست
    list_res = client.get("/api/athletes?search=محمدی", headers=headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1


def test_1_7_health_check_endpoint():
    """تست بررسی سلامت سرور"""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["app"] == "KinetiQ OS"
