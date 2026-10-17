"""
KinetiQ OS - Unit Tests for Phase 3
تست‌های اعتبارسنجی کالیبراسیون فضایی، فیلتر نویز و آنتروپومتری (تسک‌های ۳.۱ تا ۳.۷)
"""

import math
import pytest
from core.biomechanics.filter import OneEuroFilter
from core.biomechanics.calibration import SpatialCalibrator
from core.biomechanics.anthropometry import AnthropometryCalculator
from core.schemas import Landmark3D, SkeletonFrame
from plugins.anthropometry.height_span import HeightSpanTest
from core.registry import plugin_registry


def test_3_1_one_euro_filter():
    """تست تسک ۳.۱: فیلتر کینماتیک وان-یورو برای حذف لرزش مفاصل"""
    filter_x = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    noisy_signal = [10.0, 10.5, 9.8, 10.2, 10.1, 9.9, 10.3]
    filtered_results = []
    for i, val in enumerate(noisy_signal):
        filtered_results.append(filter_x.filter(val, timestamp=i * 0.033))

    # میانگین تغییرات باید به سمت مقدار پایدار ۱۰ میل کند
    assert abs(filtered_results[-1] - 10.0) < 0.3


def test_3_2_spatial_calibrator():
    """تست تسک ۳.۲: کالیبراسیون پیکسل به سانتی‌متر"""
    calibrator = SpatialCalibrator()
    # کالیبراسیون با دو نقطه که فاصله‌شان روی تصویر ۵۰۰ پیکسل است و واقعاً ۱۰۰ سانتی‌متر است
    pt1 = (0.0, 0.0)
    pt2 = (0.5, 0.0)  # در تصویر ۱۰۰۰ پیکسلی، ۵۰۰ پیکسل فاصله است
    ppm = calibrator.calibrate_by_reference_distance(pt1, pt2, real_distance_cm=100.0, image_width=1000, image_height=1000)
    assert abs(ppm - 500.0) < 0.1

    # سنجش فاصله دو نقطه دیگر
    measured_cm = calibrator.distance_cm((0.0, 0.0), (0.25, 0.0), image_width=1000, image_height=1000)
    assert abs(measured_cm - 50.0) < 0.1


def test_3_3_and_3_4_arm_span_calculation():
    """تست تسک ۳.۳ و ۳.۴: محاسبه طول اسپن به روش قطعات اسکلتی"""
    calibrator = SpatialCalibrator(pixels_per_meter=1000.0)  # ۱۰۰۰ پیکسل = ۱ متر = ۱۰۰ سانتی‌متر
    calc = AnthropometryCalculator(calibrator=calibrator)

    # ساخت یک اسکلت تست که دست‌هایش باز است (در تصویر ۱۰۰۰x۱۰۰۰)
    landmarks = {
        "left_wrist": Landmark3D(name="left_wrist", x=0.1, y=0.3, z=0.0, visibility=0.9),
        "left_elbow": Landmark3D(name="left_elbow", x=0.25, y=0.3, z=0.0, visibility=0.9),
        "left_shoulder": Landmark3D(name="left_shoulder", x=0.4, y=0.3, z=0.0, visibility=0.9),
        "right_shoulder": Landmark3D(name="right_shoulder", x=0.6, y=0.3, z=0.0, visibility=0.9),
        "right_elbow": Landmark3D(name="right_elbow", x=0.75, y=0.3, z=0.0, visibility=0.9),
        "right_wrist": Landmark3D(name="right_wrist", x=0.9, y=0.3, z=0.0, visibility=0.9),
    }

    res = calc.calculate_arm_span(landmarks, image_width=1000, image_height=1000)
    assert res["valid"] == 1.0
    assert res["arm_span_cm"] > 80.0
    assert res["shoulder_width_cm"] == 20.0  # (0.6 - 0.4) * 1000px / 1000ppm * 100cm = 20cm


def test_3_5_dynamic_height_calculation():
    """تست تسک ۳.۵: محاسبه قد بر اساس زنجیره اسکلتی"""
    calibrator = SpatialCalibrator(pixels_per_meter=1000.0)
    calc = AnthropometryCalculator(calibrator=calibrator)

    landmarks = {
        "left_shoulder": Landmark3D(name="left_shoulder", x=0.45, y=0.2, z=0.0, visibility=0.9),
        "right_shoulder": Landmark3D(name="right_shoulder", x=0.55, y=0.2, z=0.0, visibility=0.9),
        "left_hip": Landmark3D(name="left_hip", x=0.48, y=0.5, z=0.0, visibility=0.9),
        "right_hip": Landmark3D(name="right_hip", x=0.52, y=0.5, z=0.0, visibility=0.9),
        "left_knee": Landmark3D(name="left_knee", x=0.48, y=0.75, z=0.0, visibility=0.9),
        "left_ankle": Landmark3D(name="left_ankle", x=0.48, y=0.95, z=0.0, visibility=0.9),
    }

    res = calc.calculate_dynamic_height(landmarks, image_width=1000, image_height=1000)
    assert res["valid"] == 1.0
    assert res["standing_height_cm"] > 70.0
    assert res["torso_cm"] == 30.0  # 0.5 - 0.2 = 0.3 * 100cm = 30cm


def test_3_6_posture_and_asymmetry():
    """تست تسک ۳.۶: تحلیل تقارن و شیب شانه"""
    calc = AnthropometryCalculator()
    landmarks_level = {
        "left_shoulder": Landmark3D(name="left_shoulder", x=0.4, y=0.3, z=0.0, visibility=0.9),
        "right_shoulder": Landmark3D(name="right_shoulder", x=0.6, y=0.3, z=0.0, visibility=0.9),
    }
    res = calc.calculate_posture_and_asymmetry(landmarks_level)
    assert res["is_posture_balanced"] is True
    assert res["shoulder_tilt_degrees"] == 0.0


def test_3_7_height_span_plugin_lifecycle():
    """تست تسک ۳.۷: چرخه حیات کامل پلاگین HeightSpanTest"""
    test_instance = HeightSpanTest(athlete_id=1, config={"pixels_per_meter": 1000.0})
    test_instance.start()
    assert test_instance.state == "COLLECTING"

    # ارسال چند فریم نمونه
    landmarks = {
        "left_shoulder": Landmark3D(name="left_shoulder", x=0.45, y=0.2, z=0.0, visibility=0.9),
        "right_shoulder": Landmark3D(name="right_shoulder", x=0.55, y=0.2, z=0.0, visibility=0.9),
        "left_hip": Landmark3D(name="left_hip", x=0.48, y=0.5, z=0.0, visibility=0.9),
        "right_hip": Landmark3D(name="right_hip", x=0.52, y=0.5, z=0.0, visibility=0.9),
        "left_knee": Landmark3D(name="left_knee", x=0.48, y=0.75, z=0.0, visibility=0.9),
        "left_ankle": Landmark3D(name="left_ankle", x=0.48, y=0.95, z=0.0, visibility=0.9),
        "left_wrist": Landmark3D(name="left_wrist", x=0.1, y=0.2, z=0.0, visibility=0.9),
        "left_elbow": Landmark3D(name="left_elbow", x=0.25, y=0.2, z=0.0, visibility=0.9),
        "right_elbow": Landmark3D(name="right_elbow", x=0.75, y=0.2, z=0.0, visibility=0.9),
        "right_wrist": Landmark3D(name="right_wrist", x=0.9, y=0.2, z=0.0, visibility=0.9),
    }

    for i in range(5):
        frame = SkeletonFrame(frame_index=i, timestamp_ms=i * 33.3, landmarks=landmarks, detected_person=True)
        live_telemetry = test_instance.process_frame(frame)
        assert live_telemetry["status"] == "TRACKING"

    summary = test_instance.finish()
    assert summary.test_id == "height_span"
    assert summary.primary_metric_name == "standing_height_cm"
    assert summary.primary_metric_value > 50.0
    assert len(summary.feedback_notes) >= 1
