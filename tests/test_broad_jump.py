"""
KinetiQ OS - Unit Tests for Standing Broad Jump Plugin
تست اعتبارسنجی آزمون پرش طول جفت‌پا و رجیستری آن در سامانه
"""

import pytest
from core.schemas import SkeletonFrame, Landmark3D
from plugins.jumps.broad_jump import StandingBroadJumpTest
from core.registry import plugin_registry


def test_broad_jump_lifecycle():
    """تست چرخه حیات کامل آزمون پرش طول جفت‌پا"""
    test_inst = StandingBroadJumpTest(athlete_id=5, config={"pixels_per_meter": 500.0})
    test_inst.start()
    assert test_inst.is_active is True
    assert test_inst.state == "STAND"

    # فاز ۱: ایستادن در مبدا x=0.2, y=0.85
    lms_stand = {
        "left_ankle": Landmark3D(name="left_ankle", x=0.2, y=0.85, z=0.0, visibility=0.9),
        "left_hip": Landmark3D(name="left_hip", x=0.2, y=0.5, z=0.0, visibility=0.9),
        "left_knee": Landmark3D(name="left_knee", x=0.2, y=0.7, z=0.0, visibility=0.9),
    }
    frame1 = SkeletonFrame(frame_index=1, timestamp_ms=0.0, landmarks=lms_stand, detected_person=True)
    res1 = test_inst.process_frame(frame1)
    assert res1["state"] == "STAND"

    # فاز ۲: آماده‌سازی و خمش
    lms_squat = {
        "left_ankle": Landmark3D(name="left_ankle", x=0.2, y=0.85, z=0.0, visibility=0.9),
        "left_hip": Landmark3D(name="left_hip", x=0.2, y=0.6, z=0.0, visibility=0.9),
        "left_knee": Landmark3D(name="left_knee", x=0.25, y=0.75, z=0.0, visibility=0.9),
    }
    frame2 = SkeletonFrame(frame_index=2, timestamp_ms=100.0, landmarks=lms_squat, detected_person=True)
    res2 = test_inst.process_frame(frame2)
    assert res2["state"] == "PREPARE"

    # فاز ۳: پرواز در هوا
    lms_flight = {
        "left_ankle": Landmark3D(name="left_ankle", x=0.45, y=0.65, z=0.0, visibility=0.9),
        "left_hip": Landmark3D(name="left_hip", x=0.45, y=0.45, z=0.0, visibility=0.9),
        "left_knee": Landmark3D(name="left_knee", x=0.45, y=0.55, z=0.0, visibility=0.9),
    }
    frame3 = SkeletonFrame(frame_index=3, timestamp_ms=300.0, landmarks=lms_flight, detected_person=True)
    res3 = test_inst.process_frame(frame3)
    assert res3["state"] == "FLIGHT"

    # فاز ۴: فرود در نقطه جدید x=0.7, y=0.88
    lms_land = {
        "left_ankle": Landmark3D(name="left_ankle", x=0.7, y=0.88, z=0.0, visibility=0.9),
        "left_hip": Landmark3D(name="left_hip", x=0.7, y=0.55, z=0.0, visibility=0.9),
        "left_knee": Landmark3D(name="left_knee", x=0.7, y=0.72, z=0.0, visibility=0.9),
    }
    frame4 = SkeletonFrame(frame_index=4, timestamp_ms=500.0, landmarks=lms_land, detected_person=True)
    res4 = test_inst.process_frame(frame4)
    assert res4["state"] == "LANDED"
    assert res4["jump_distance_cm"] > 100.0

    # صدور خلاصه نتایج آزمون
    summary = test_inst.finish()
    assert summary.test_id == "broad_jump"
    assert summary.primary_metric_name == "broad_jump_distance_cm"
    assert summary.primary_metric_value > 100.0
    assert summary.passed is True


def test_broad_jump_in_registry():
    """بررسی وجود آزمون پرش طول در رجیستری سراسری سیستم"""
    tests = plugin_registry.list_available_tests()
    test_ids = [t["test_id"] for t in tests]
    assert "broad_jump" in test_ids
