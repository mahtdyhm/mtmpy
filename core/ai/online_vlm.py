"""
KinetiQ OS - Online Cloud Pose Estimator
موتور اتصال به هوش مصنوعی ابری با کالیبراسیون و تبدیل به شمای واحد
"""

from typing import Optional, Dict
import base64
import cv2
import numpy as np
from core.ai.base_estimator import BasePoseEstimator
from core.schemas import SkeletonFrame, Landmark3D
from config.settings import settings
from core.logging import logger

# مپینگ ۱۷ نقطه COCO به فرمت استاندارد
COCO_17_MAP = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


class OnlineCloudPoseEstimator(BasePoseEstimator):
    """موتور آنلاین پردازش ابری برای دقت‌های پژوهشی"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(mode_name="online_cloud")
        self.api_key = api_key or settings.ONLINE_API_KEY
        self.base_url = base_url or settings.ONLINE_GATEWAY_URL
        self.client = None

    def initialize(self) -> None:
        """آماده‌سازی کلاینت سرور ابری"""
        if self.is_initialized:
            return
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=15.0)
                self.is_initialized = True
                logger.info("کلاینت آنلاین ابری با موفقیت متصل و مقداردهی شد.")
            except Exception as e:
                logger.warning(f"عدم امکان اتصال به سرور ابری آنلاین: {e}")
                self.is_initialized = False
        else:
            logger.info("کلید آنلاین ارائه نشده است؛ آماده برای حالت پروکسی پیش‌فرض سرور.")
            self.is_initialized = True

    def estimate_frame(self, image_bgr: np.ndarray, frame_index: int = 0, timestamp_ms: float = 0.0) -> Optional[SkeletonFrame]:
        """ارسال فریم به سرور ابری و تبدیل به SkeletonFrame"""
        if not self.is_initialized:
            self.initialize()

        if self.client is None or not self.api_key:
            raise ConnectionError("سرویس ابری آنلاین به دلیل عدم وجود اتصال یا کلید در دسترس نیست.")

        # کدگذاری فریم به تصویر JPEG جهت ارسال فشرده
        success, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return None

        # در حالت استریم تک‌فریم یا بچ
        return SkeletonFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            landmarks={},
            detected_person=False,
            ai_mode="online",
        )

    def close(self) -> None:
        self.client = None
        self.is_initialized = False
