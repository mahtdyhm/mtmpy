"""
KinetiQ OS - Unit Tests for Phase 6 & Phase 7
تست‌های اعتبارسنجی رابط کاربری، رندرینگ ویدیویی، گزارش استعدادیابی و تست جامع یکپارچگی (End-to-End)
"""

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.server import app
from core.export.telemetry_renderer import VideoTelemetryRenderer, ScoutReportExporter
from core.schemas import SkeletonFrame, Landmark3D, MetricSummary
from core.registry import plugin_registry

client = TestClient(app)


def test_6_1_and_6_2_serve_ui_and_manifest():
    """تست تسک ۶.۱ و ۶.۲: ارائه رابط کاربری وب و فایل مانیفست PWA"""
    res = client.get("/")
    assert res.status_code == 200
    assert "KinetiQ OS" in res.text
    assert "سنجش قد و اسپن" in res.text

    manifest_res = client.get("/static/manifest.json")
    assert manifest_res.status_code == 200
    assert manifest_res.json()["short_name"] == "KinetiQ"


def test_6_3_auto_registered_plugins():
    """تست تسک ۶.۳: بررسی رجیستر شدن خودکار ۳ پلاگین اصلی در زمان لود سیستم"""
    plugins = plugin_registry.list_available_tests()
    test_ids = [p["test_id"] for p in plugins]
    assert "height_span" in test_ids
    assert "vertical_jump" in test_ids
    assert "handball_three_step" in test_ids


def test_7_1_and_7_3_scout_report_exporter(tmp_path: Path):
    """تست تسک ۷.۱ و ۷.۳: صدور فایل‌های گزارش CSV و JSON استعدادیابی"""
    csv_file = tmp_path / "test_telemetry.csv"
    data = [
        {"frame": 1, "knee_angle": 175.0, "height_cm": 0.0},
        {"frame": 2, "knee_angle": 140.0, "height_cm": 15.0},
    ]
    ScoutReportExporter.export_csv(data, csv_file)
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "frame,knee_angle,height_cm" in content

    # صدور کارت JSON استعدادیابی
    json_file = tmp_path / "scout_card.json"
    summary = MetricSummary(
        test_id="height_span",
        athlete_id=1,
        primary_metric_name="standing_height_cm",
        primary_metric_value=185.0,
        primary_metric_unit="cm",
        detailed_metrics={"arm_span_cm": 192.0, "ape_index": 1.04},
        feedback_notes=["پتانسیل بالا برای هندبال."],
    )
    athlete_info = {"name": "علی محمدی", "sport": "handball", "position": "wing"}
    ScoutReportExporter.export_scout_card_json(summary, athlete_info, json_file)
    assert json_file.exists()
    json_text = json_file.read_text(encoding="utf-8")
    assert "علی محمدی" in json_text
    assert "185.0" in json_text


def test_7_2_video_telemetry_renderer():
    """تست تسک ۷.۲: رندرینگ فریم ویدیویی با اسکلت مفاصل و HUD"""
    renderer = VideoTelemetryRenderer(output_width=640, output_height=360)
    blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    lms = {
        "left_shoulder": Landmark3D(name="left_shoulder", x=0.4, y=0.3, z=0.0, visibility=0.9),
        "right_shoulder": Landmark3D(name="right_shoulder", x=0.6, y=0.3, z=0.0, visibility=0.9),
        "left_elbow": Landmark3D(name="left_elbow", x=0.3, y=0.5, z=0.0, visibility=0.9),
        "right_elbow": Landmark3D(name="right_elbow", x=0.7, y=0.5, z=0.0, visibility=0.9),
    }
    skeleton = SkeletonFrame(frame_index=1, timestamp_ms=33.3, landmarks=lms, detected_person=True)
    telemetry = {"قد": "185.0 cm", "وضعیت": "TRACKING"}

    rendered = renderer.render_overlay_frame(blank_frame, skeleton, telemetry, test_title="آنتروپومتری")
    assert rendered.shape == (360, 640, 3)
    # اطمینان از اینکه فریم دیگر تماماً سیاه نیست و خطوط HUD و اسکلت روی آن رسم شده است
    assert np.sum(rendered) > 0


def test_7_6_end_to_end_system_pipeline():
    """تست تسک ۷.۶: تست جامع یکپارچه کل چرخه (ورود -> تعریف ورزشکار -> شروع سشن -> دریافت جزئیات)"""
    # ۱. لاگین ادمین
    login_res = client.post("/api/auth/login", data={"username": "admin", "password": "admin123"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # ۲. ایجاد ورزشکار
    athlete_res = client.post(
        "/api/athletes",
        json={"first_name": "رضا", "last_name": "کریمی", "national_id": "9988776655", "sport": "handball"},
        headers=headers,
    )
    assert athlete_res.status_code in [201, 400]  # در صورت تکرار شناسه
    athlete_id = athlete_res.json().get("id", 1)

    # ۳. شروع سشن آزمون پرش عمودی
    session_res = client.post(
        "/api/tests/sessions",
        json={"test_id": "vertical_jump", "athlete_id": athlete_id, "ai_mode": "offline"},
        headers=headers,
    )
    assert session_res.status_code == 201
    session_id = session_res.json()["id"]

    # ۴. استعلام سشن
    details_res = client.get(f"/api/tests/sessions/{session_id}", headers=headers)
    assert details_res.status_code == 200
    assert details_res.json()["session"]["test_id"] == "vertical_jump"
