"""
KinetiQ OS - Hybrid AI Pose Manager
مدیریت هوشمند سوئیچ بین حالت آفلاین محلی (YOLO-Pose) و آنلاین ابری همراه با Fallback اتوماتیک
"""

from typing import Optional
import numpy as np
from core.ai.base_estimator import BasePoseEstimator
from core.ai.offline_yolo import YoloPoseEstimator
from core.ai.online_vlm import OnlineCloudPoseEstimator
from core.schemas import SkeletonFrame
from config.settings import settings
from core.logging import logger


class HybridPoseManager:
    """مدیر ارکستراسیون هوش مصنوعی با قابلیت سوئیچ آنی"""

    def __init__(self, default_mode: Optional[str] = None):
        self.current_mode = default_mode or settings.DEFAULT_AI_MODE
        self.offline_estimator = YoloPoseEstimator()
        self.online_estimator = OnlineCloudPoseEstimator()
        self.active_estimator: BasePoseEstimator = self.offline_estimator

    def set_mode(self, mode: str) -> bool:
        """تغییر حالت کاری سیستم بین 'offline' و 'online'"""
        mode = mode.lower()
        if mode == "online":
            try:
                self.online_estimator.initialize()
                self.active_estimator = self.online_estimator
                self.current_mode = "online"
                logger.info("حالت هوش مصنوعی با موفقیت به 'online' (ابری) تغییر یافت.")
                return True
            except Exception as e:
                logger.warning(f"فعال‌سازی حالت آنلاین ناموفق بود: {e}، بازگشت به حالت آفلاین.")
                self.active_estimator = self.offline_estimator
                self.current_mode = "offline"
                return False
        else:
            self.offline_estimator.initialize()
            self.active_estimator = self.offline_estimator
            self.current_mode = "offline"
            logger.info("حالت هوش مصنوعی با موفقیت به 'offline' (محلی) تغییر یافت.")
            return True

    def process_frame(self, image_bgr: np.ndarray, frame_index: int = 0, timestamp_ms: float = 0.0) -> Optional[SkeletonFrame]:
        """پردازش فریم با مکانیزم Fallback خودکار در صورت قطعی اینترنت"""
        try:
            result = self.active_estimator.estimate_frame(image_bgr, frame_index, timestamp_ms)
            return result
        except Exception as e:
            if self.current_mode == "online" and settings.ALLOW_ONLINE_FALLBACK:
                logger.warning(f"خطای سرویس آنلاین: {e} | سوئیچ اضطراری خودکار (Fallback) به مدل آفلاین...")
                self.set_mode("offline")
                return self.active_estimator.estimate_frame(image_bgr, frame_index, timestamp_ms)
            else:
                logger.error(f"خطای پردازش فریم در حالت {self.current_mode}: {e}")
                return None

    def close(self) -> None:
        """آزادسازی تمام موتورهای فعال"""
        self.offline_estimator.close()
        self.online_estimator.close()


# نمونه سراسری مدیر هوش مصنوعی
hybrid_ai_manager = HybridPoseManager()
