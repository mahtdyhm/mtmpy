"""
KinetiQ OS - Unit Tests for Phase 5
تست‌های اعتبارسنجی آنالیزهای تاکتیکی، خطاهای داوری و بیومکانیک هندبال (تسک‌های ۵.۱ تا ۵.۷)
"""

import pytest
from core.sports.handball_engine import HandballEngine
from core.schemas import Landmark3D, SkeletonFrame
from plugins.handball.three_step import HandballThreeStepTest
from core.registry import plugin_registry


def test_5_1_ball_possession():
    """تست تسک ۵.۱: تشخیص مالکیت توپ بر اساس مجاورت با مچ دست"""
    engine = HandballEngine()
    lms = {
        "left_wrist": Landmark3D(name="left_wrist", x=0.4, y=0.5, z=0.0, visibility=0.9),
        "right_wrist": Landmark3D(name="right_wrist", x=0.6, y=0.5, z=0.0, visibility=0.9),
    }
    # توپ نزدیک دست چپ (x=0.42, y=0.51)
    ball_pos = (0.42, 0.51, 0.05)
    has_possession = engine.update_ball_possession(ball_pos, lms)
    assert has_possession is True


def test_5_2_and_5_3_three_step_and_traveling():
    """تست تسک ۵.۲ و ۵.۳: شمارش ۳ گام و تشخیص خطای رانینگ در گام چهارم"""
    engine = HandballEngine()
    engine.has_possession = True

    def make_foot_frame(idx: int, t_s: float, left_y: float, right_y: float):
        lms = {
            "left_ankle": Landmark3D(name="left_ankle", x=0.45, y=left_y, z=0.0, visibility=0.9),
            "right_ankle": Landmark3D(name="right_ankle", x=0.55, y=right_y, z=0.0, visibility=0.9),
            "right_shoulder": Landmark3D(name="right_shoulder", x=0.5, y=0.2, z=0.0, visibility=0.9),
            "right_elbow": Landmark3D(name="right_elbow", x=0.5, y=0.35, z=0.0, visibility=0.9),
            "right_wrist": Landmark3D(name="right_wrist", x=0.5, y=0.5, z=0.0, visibility=0.9),
        }
        return SkeletonFrame(frame_index=idx, timestamp_ms=t_s * 1000.0, landmarks=lms, detected_person=True)

    # شبیه‌سازی گام ۱ (پای چپ)
    engine.left_foot_down = False
    engine.process_step_rhythm(make_foot_frame(1, 0.5, left_y=0.92, right_y=0.80))
    assert engine.steps_count == 1

    # شبیه‌سازی گام ۲ (پای راست)
    engine.right_foot_down = False
    engine.process_step_rhythm(make_foot_frame(2, 1.0, left_y=0.80, right_y=0.92))
    assert engine.steps_count == 2

    # شبیه‌سازی گام ۳ (پای چپ)
    engine.left_foot_down = False
    engine.process_step_rhythm(make_foot_frame(3, 1.5, left_y=0.92, right_y=0.80))
    assert engine.steps_count == 3
    assert len(engine.violations) == 0  # تا گام ۳ قانونی است

    # شبیه‌سازی گام ۴ غیرمجاز (پای راست) -> باید خطای Traveling ثبت شود
    engine.right_foot_down = False
    engine.process_step_rhythm(make_foot_frame(4, 2.0, left_y=0.80, right_y=0.92))
    assert engine.steps_count == 4
    assert len(engine.violations) >= 1
    assert engine.violations[0].violation_type == "TRAVELING"


def test_5_4_double_dribble_violation():
    """تست تسک ۵.۴: خطای دبل در شروع مجدد دریبل"""
    engine = HandballEngine()
    engine.has_possession = True
    engine.dribble_stopped = True  # بازیکن توپ را با دو دست گرفته است

    lms = {
        "left_ankle": Landmark3D(name="left_ankle", x=0.45, y=0.85, z=0.0, visibility=0.9),
        "right_ankle": Landmark3D(name="right_ankle", x=0.55, y=0.85, z=0.0, visibility=0.9),
    }
    frame = SkeletonFrame(frame_index=1, timestamp_ms=500.0, landmarks=lms, detected_person=True)

    # اگر مجدداً دریبل بزند (is_dribbling=True)
    engine.process_step_rhythm(frame, is_dribbling=True)
    assert any(v.violation_type == "DOUBLE_DRIBBLE" for v in engine.violations)


def test_5_7_goalkeeper_coverage():
    """تست تسک ۵.۷: محاسبه درصد پوشش چارچوب توسط دروازه‌بان"""
    engine = HandballEngine()
    # دروازه‌بان دست‌ها و پاها را باز کرده است
    lms = {
        "left_wrist": Landmark3D(name="left_wrist", x=0.2, y=0.3, z=0.0, visibility=0.9),
        "right_wrist": Landmark3D(name="right_wrist", x=0.8, y=0.3, z=0.0, visibility=0.9),
        "left_ankle": Landmark3D(name="left_ankle", x=0.2, y=0.9, z=0.0, visibility=0.9),
        "right_ankle": Landmark3D(name="right_ankle", x=0.8, y=0.9, z=0.0, visibility=0.9),
    }

    res = engine.calculate_goalkeeper_coverage(lms)
    assert res["coverage_percent"] > 50.0  # پوشش گسترده


def test_5_plugin_lifecycle():
    """تست چرخه حیات کامل پلاگین HandballThreeStepTest"""
    plugin = HandballThreeStepTest(athlete_id=3)
    plugin.start()
    assert plugin.is_active is True

    lms = {
        "left_ankle": Landmark3D(name="left_ankle", x=0.45, y=0.85, z=0.0, visibility=0.9),
        "right_ankle": Landmark3D(name="right_ankle", x=0.55, y=0.85, z=0.0, visibility=0.9),
    }
    frame = SkeletonFrame(frame_index=0, timestamp_ms=0.0, landmarks=lms, detected_person=True)
    res = plugin.process_frame(frame)
    assert res["status"] == "ACTIVE"

    metric = plugin.finish()
    assert metric.test_id == "handball_three_step"
    assert metric.primary_metric_name == "steps_taken"
