"""
KinetiQ OS - Unit Tests for Scout PDF Report Generation
تست صدور فایل PDF کارنامه استعدادیابی و اندپوینت دانلود گزارش
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api.server import app
from core.export.pdf_generator import ScoutPDFReportGenerator
from core.schemas import MetricSummary
from db.session import init_db

client = TestClient(app)


def setup_module():
    init_db()


def test_pdf_report_direct_generation(tmp_path: Path):
    """تست تولید مستقیم فایل PDF استعدادیابی با نمودار رادار"""
    output_pdf = tmp_path / "test_scout.pdf"
    athlete_info = {
        "first_name": "آرش",
        "last_name": "کمالی",
        "national_id": "0012345678",
        "gender": "male",
        "sport": "Handball",
        "position": "Goalkeeper",
        "standing_height_cm": 194.0,
        "arm_span_cm": 202.0,
        "weight_kg": 91.0,
    }
    summary = MetricSummary(
        test_id="vertical_jump",
        athlete_id=1,
        primary_metric_name="max_jump_height_cm",
        primary_metric_value=62.5,
        primary_metric_unit="cm",
        detailed_metrics={
            "flight_time_s": 0.71,
            "rsi": 2.45,
            "takeoff_velocity_mps": 3.48,
        },
        feedback_notes=[
            "Explosive power and elastic rate of force development are exceptional.",
            "Takeoff symmetry balanced with minimal knee valgus.",
        ],
    )

    result_path = ScoutPDFReportGenerator.create_report(
        athlete_info=athlete_info,
        metric_summary=summary,
        output_path=output_pdf,
        session_info={"id": 99, "created_at": "2026-10-31 12:00", "ai_mode": "Offline YOLO11"},
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 5000  # حجم فایل به خاطر تصاویر و جداول باید معقول باشد
    with open(result_path, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-"  # فرمت استاندارد PDF


def test_pdf_download_api_endpoint():
    """تست اندپوینت دانلود فایل PDF از طریق API"""
    # ورود ادمین
    login_res = client.post("/api/auth/login", data={"username": "admin", "password": "admin123"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # ایجاد یک سشن تست
    session_res = client.post(
        "/api/tests/sessions",
        json={"test_id": "vertical_jump", "athlete_id": 1, "ai_mode": "offline"},
        headers=headers,
    )
    assert session_res.status_code == 201
    session_id = session_res.json()["id"]

    # ثبت خاتمه سشن
    finish_res = client.post(
        f"/api/tests/sessions/{session_id}/finish",
        json={
            "primary_metric_name": "max_jump_height_cm",
            "primary_metric_value": 55.4,
            "primary_metric_unit": "cm",
            "detailed_metrics": {"rsi": 2.1, "flight_time_s": 0.67},
            "feedback_notes": ["Good jump execution"],
        },
        headers=headers,
    )
    assert finish_res.status_code == 200

    # درخواست دانلود PDF
    pdf_res = client.get(f"/api/tests/sessions/{session_id}/report.pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF-")
