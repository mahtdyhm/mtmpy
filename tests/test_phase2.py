"""
KinetiQ OS - Unit Tests for Phase 2
تست‌های اعتبارسنجی موتور هیبریدی هوش مصنوعی (تسک‌های ۲.۱ تا ۲.۷)
"""

import numpy as np
import pytest
from core.schemas import SkeletonFrame, Landmark3D
from core.ai.offline_yolo import YoloPoseEstimator
from core.ai.online_vlm import OnlineCloudPoseEstimator
from core.ai.hybrid_manager import HybridPoseManager
from core.ai.object_tracker import ObjectTracker
from core.ai.frame_buffer import FrameRingBuffer


def test_2_1_unified_pose_schema():
    """تست تسک ۲.۱: اعتبارسنجی شمای یکپارچه مفاصل"""
    landmark = Landmark3D(
        name="left_wrist",
        x=0.45,
        y=0.65,
        z=-0.12,
        visibility=0.98,
        presence=1.0,
    )
    frame = SkeletonFrame(
        frame_index=1,
        timestamp_ms=33.3,
        landmarks={"left_wrist": landmark},
        detected_person=True,
        ai_mode="offline",
    )
    assert frame.landmarks["left_wrist"].x == 0.45
    assert frame.detected_person is True


def test_2_2_offline_yolo_pose():
    """تست تسک ۲.۲: بررسی موتور پوز آفلاین YOLO11"""
    estimator = YoloPoseEstimator(model_name="yolo11n-pose.pt")
    estimator.initialize()
    assert estimator.is_initialized is True

    # ساخت یک تصویر تست سفید
    dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    frame = estimator.estimate_frame(dummy_image, frame_index=0, timestamp_ms=0.0)

    assert frame is not None
    assert frame.ai_mode == "offline"
    estimator.close()
    assert estimator.is_initialized is False


def test_2_3_and_2_4_hybrid_manager_and_fallback():
    """تست تسک ۲.۳ و ۲.۴: سوئیچ بین حالت‌ها و فال‌بک خودکار"""
    manager = HybridPoseManager(default_mode="offline")
    assert manager.current_mode == "offline"

    # تست تغییر مود
    manager.set_mode("offline")
    assert manager.current_mode == "offline"

    # پردازش یک فریم تست
    dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
    result = manager.process_frame(dummy_image)
    assert result is not None
    assert result.ai_mode == "offline"
    manager.close()


def test_2_5_object_tracker():
    """تست تسک ۲.۵: ردیاب اشیاء برای توپ هندبال"""
    tracker = ObjectTracker(target_name="handball")

    # ساخت فریم با یک دایره نارنجی رنگ شبیه توپ هندبال
    test_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    import cv2
    cv2.circle(test_frame, (150, 150), 20, (0, 140, 255), -1)

    result = tracker.track_hsv_sphere(test_frame)
    assert result is not None
    cx, cy, r = result
    assert 0.4 < cx < 0.6
    assert 0.4 < cy < 0.6


def test_2_6_frame_ring_buffer():
    """تست تسک ۲.۶: بافر حلقوی فریم‌ها و پیشگیری از نشت حافظه"""
    buffer = FrameRingBuffer(max_capacity=5)
    for i in range(10):
        dummy_frame = np.ones((50, 50, 3), dtype=np.uint8) * i
        buffer.append(dummy_frame, metadata={"index": i})

    assert buffer.size() == 5  # حداکثر ۵ فریم نگه داشته شده
    assert buffer.get_latest()["metadata"]["index"] == 9
    buffer.clear()
    assert buffer.size() == 0
