"""
KinetiQ OS - Offline MediaPipe Pose Estimator
موتور تخمین پوز ۱۰۰٪ آفلاین و محلی با ۳۳ مفصل سه‌بعدی و سرعت بیش از ۶۰ فریم بر ثانیه
"""

from typing import Optional, Dict
import cv2
import numpy as np
import mediapipe as mp
from core.ai.base_estimator import BasePoseEstimator
from core.schemas import SkeletonFrame, Landmark3D
from core.logging import logger

# تطبیق ۳۳ اندیس مفاصل مدیاپایپ با نام‌های استاندارد
MP_LANDMARK_MAP = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


class MediaPipePoseEstimator(BasePoseEstimator):
    """پیاده‌سازی موتور آفلاین مبتنی بر Google MediaPipe"""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        super().__init__(mode_name="offline_mediapipe")
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.pose = None

    def initialize(self) -> None:
        """مقداردهی اولیه پایپ‌لاین مدیاپایپ"""
        if self.is_initialized:
            return
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.is_initialized = True
        logger.info("موتور آفلاین MediaPipe Pose با موفقیت مقداردهی و فعال شد.")

    def estimate_frame(self, image_bgr: np.ndarray, frame_index: int = 0, timestamp_ms: float = 0.0) -> Optional[SkeletonFrame]:
        """پردازش تک‌فریم و استخراج اسکلت ۳۳ مفصلی"""
        if not self.is_initialized:
            self.initialize()

        if image_bgr is None or image_bgr.size == 0:
            return None

        # تبدیل BGR به RGB برای مدیاپایپ
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return SkeletonFrame(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                landmarks={},
                detected_person=False,
                ai_mode="offline",
            )

        landmarks_dict: Dict[str, Landmark3D] = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            name = MP_LANDMARK_MAP.get(idx, f"landmark_{idx}")
            landmarks_dict[name] = Landmark3D(
                name=name,
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
                presence=float(getattr(lm, "presence", 1.0)),
            )

        return SkeletonFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            landmarks=landmarks_dict,
            detected_person=True,
            ai_mode="offline",
        )

    def close(self) -> None:
        """بستن مدل و آزادسازی حافظه"""
        if self.pose:
            self.pose.close()
            self.pose = None
            self.is_initialized = False
            logger.info("موتور آفلاین MediaPipe با موفقیت بسته شد.")
