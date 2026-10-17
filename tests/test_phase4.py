"""
KinetiQ OS - Unit Tests for Phase 4
تست‌های اعتبارسنجی آزمون‌های پرش عمودی، توان بی‌هوازی و شاخص‌های دینامیک (تسک‌های ۴.۱ تا ۴.۷)
"""

import pytest
from core.biomechanics.jump_engine import JumpAnalyzer, GRAVITY
from core.schemas import Landmark3D, SkeletonFrame
from plugins.jumps.vertical_jump import VerticalJumpTest
from core.registry import plugin_registry


def test_4_1_and_4_2_flight_time_formula():
    """تست تسک ۴.۲: بررسی درستی فرمول فیزیک سقوط آزاد نیوتن"""
    flight_time = 0.5
    h_m = (1.0 / 8.0) * GRAVITY * (flight_time ** 2)
    h_cm = h_m * 100.0
    assert 30.0 < h_cm < 31.0


def test_4_3_single_cmj_state_machine():
    """تست تسک ۴.۱ و ۴.۳: شبیه‌سازی مراحل پرش CMJ و استیت ماشین"""
    analyzer = JumpAnalyzer(mode="cmj")

    def make_frame(idx: int, t_ms: float, hip_y: float, knee_x: float, ankle_y: float):
        lms = {
            "left_hip": Landmark3D(name="left_hip", x=0.5, y=hip_y, z=0.0, visibility=0.9),
            "left_knee": Landmark3D(name="left_knee", x=knee_x, y=hip_y + 0.25, z=0.0, visibility=0.9),
            "left_ankle": Landmark3D(name="left_ankle", x=0.5, y=ankle_y, z=0.0, visibility=0.9),
        }
        return SkeletonFrame(frame_index=idx, timestamp_ms=t_ms, landmarks=lms, detected_person=True)

    # ۱. فاز ایستادن (Standing - پای صاف)
    for i in range(5):
        analyzer.update_frame(make_frame(i, i * 33.3, hip_y=0.5, knee_x=0.5, ankle_y=0.95))
    assert analyzer.state == "STAND"

    # ۲. فاز خمش زانو (Squat - زانو خم شده به جلو x=0.62)
    for i in range(5, 10):
        analyzer.update_frame(make_frame(i, i * 33.3, hip_y=0.6, knee_x=0.62, ankle_y=0.95))
    assert analyzer.state == "SQUAT"

    # ۳. فاز پرواز در هوا (Flight - جدا شدن از زمین به مدت ۰.۵ ثانیه)
    takeoff_start_ms = 400.0
    for i in range(15):
        t = takeoff_start_ms + i * 33.3
        analyzer.update_frame(make_frame(10 + i, t, hip_y=0.35, knee_x=0.5, ankle_y=0.75))
    assert analyzer.in_flight is True
    assert analyzer.state == "FLIGHT"

    # ۴. فاز فرود روی زمین (Landing)
    landing_ms = takeoff_start_ms + 15 * 33.3
    for i in range(5):
        t = landing_ms + i * 33.3
        analyzer.update_frame(make_frame(25 + i, t, hip_y=0.55, knee_x=0.5, ankle_y=0.95))

    assert len(analyzer.reps) == 1
    assert analyzer.reps[0].jump_height_cm > 25.0
    assert analyzer.reps[0].is_valid is True


def test_4_4_and_4_5_repeated_jump_and_rsi():
    """تست تسک ۴.۴ و ۴.۵: شاخص توان کشسانی RSI و خستگی"""
    analyzer = JumpAnalyzer(mode="repeated_bosco")
    from core.biomechanics.jump_engine import JumpRepRecord
    analyzer.reps.append(JumpRepRecord(1, 1.0, 1.5, 0.5, 30.6, contact_time_s=0.25, rsi=2.0))
    analyzer.reps.append(JumpRepRecord(2, 1.8, 2.3, 0.5, 30.6, contact_time_s=0.25, rsi=2.0))

    summary = analyzer.compute_summary()
    assert summary["total_reps"] == 2
    assert summary["max_height_cm"] == 30.6
    assert summary["mean_rsi"] == 2.0


def test_4_7_vertical_jump_plugin_lifecycle():
    """تست تسک ۴.۷: چرخه حیات کامل پلاگین VerticalJumpTest"""
    plugin = VerticalJumpTest(athlete_id=2, config={"jump_mode": "cmj"})
    plugin.start()
    assert plugin.is_active is True

    lms = {
        "left_hip": Landmark3D(name="left_hip", x=0.5, y=0.5, z=0.0, visibility=0.9),
        "left_knee": Landmark3D(name="left_knee", x=0.5, y=0.75, z=0.0, visibility=0.9),
        "left_ankle": Landmark3D(name="left_ankle", x=0.5, y=0.95, z=0.0, visibility=0.9),
    }
    frame = SkeletonFrame(frame_index=0, timestamp_ms=0.0, landmarks=lms, detected_person=True)
    res = plugin.process_frame(frame)
    assert res["status"] == "TESTING"

    metric = plugin.finish()
    assert metric.test_id == "vertical_jump"
    assert metric.primary_metric_name == "max_jump_height_cm"
