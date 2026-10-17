"""
KinetiQ OS - Base AI Pose Estimator Interface
اینترفیس پایه برای تمامی موتورهای تخمین پوز (آفلاین محلی و آنلاین ابری)
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from core.schemas import SkeletonFrame


class BasePoseEstimator(ABC):
    """کلاس انتزاعی برای استخراج اسکلت مفاصل از فریم تصویر"""

    def __init__(self, mode_name: str):
        self.mode_name = mode_name
        self.is_initialized: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """آماده‌سازی مدل و بارگذاری وزن‌ها"""
        pass

    @abstractmethod
    def estimate_frame(self, image_bgr: np.ndarray, frame_index: int = 0, timestamp_ms: float = 0.0) -> Optional[SkeletonFrame]:
        """
        پردازش یک فریم BGR و استخراج مختصات استاندارد SkeletonFrame
        در صورت عدم تشخیص شخص در کادر، خروجی None یا اسکلت خالی برمی‌گرداند.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """آزادسازی منابع و مدل از حافظه"""
        pass
