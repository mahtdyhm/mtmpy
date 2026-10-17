"""
KinetiQ OS - Object Tracker (Ball & Gear Tracking)
ردیابی اشیاء محیطی (مانند توپ هندبال) بر پایه ویژگی‌های حرکتی و ابعاد
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class ObjectTracker:
    """ردیاب اپتیکال سبک برای تشخیص موقعیت توپ هندبال"""

    def __init__(self, target_name: str = "handball"):
        self.target_name = target_name
        self.last_position: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h)
        self.is_tracking: bool = False

    def track_hsv_sphere(
        self,
        frame_bgr: np.ndarray,
        lower_hsv: Tuple[int, int, int] = (5, 100, 100),
        upper_hsv: Tuple[int, int, int] = (25, 255, 255),
    ) -> Optional[Tuple[float, float, float]]:
        """
        تشخیص و ردیابی توپ بر اساس ماسک رنگی و فیلتر کرویت
        خروجی: (x_center, y_center, radius) در مقیاس نرمالایز شده [0, 1]
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

        # حذف نویز با عملیات مورفولوژی
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.is_tracking = False
            return None

        # پیدا کردن بزرگترین کانتور کروی
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 50:  # حداقل مساحت قابل قبول
            self.is_tracking = False
            return None

        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        if radius < 5:
            self.is_tracking = False
            return None

        self.is_tracking = True
        return float(cx / w), float(cy / h), float(radius / w)
