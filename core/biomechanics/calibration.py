"""
KinetiQ OS - Spatial Calibration Engine
تبدیل مختصات پیکسلی/نرمالایز شده به مقیاس واقعی سانتی‌متر با دقت زیر ۱ سانتی‌متر
"""

import math
from typing import Optional, Tuple


class SpatialCalibrator:
    """موتور کالیبراسیون مقیاس تصویر به متراژ واقعی (Pixels to Centimeters)"""

    def __init__(self, pixels_per_meter: Optional[float] = None):
        # تعداد پیکسل در هر متر (یا نسبت نرمالایز شده)
        self.pixels_per_meter = pixels_per_meter or 500.0
        self.is_calibrated: bool = pixels_per_meter is not None

    def calibrate_by_reference_distance(
        self,
        pt1: Tuple[float, float],
        pt2: Tuple[float, float],
        real_distance_cm: float,
        image_width: int,
        image_height: int,
    ) -> float:
        """
        کالیبراسیون با استفاده از دو نقطه روی صفحه با فاصله واقعی مشخص (مثلاً خط ۵۰ سانتی روی دیوار)
        pt1, pt2: مختصات نرمالایز شده [0, 1]
        """
        x1, y1 = pt1[0] * image_width, pt1[1] * image_height
        x2, y2 = pt2[0] * image_width, pt2[1] * image_height

        pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if pixel_distance <= 0:
            raise ValueError("فاصله بین دو نقطه کالیبراسیون نمی‌تواند صفر باشد.")

        # محاسبه پیکسل بر متر
        meters = real_distance_cm / 100.0
        self.pixels_per_meter = pixel_distance / meters
        self.is_calibrated = True
        return self.pixels_per_meter

    def distance_cm(
        self,
        pt1: Tuple[float, float],
        pt2: Tuple[float, float],
        image_width: int,
        image_height: int,
    ) -> float:
        """محاسبه فاصله فیزیکی دو نقطه به سانتی‌متر"""
        x1, y1 = pt1[0] * image_width, pt1[1] * image_height
        x2, y2 = pt2[0] * image_width, pt2[1] * image_height

        pixel_dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        meters = pixel_dist / self.pixels_per_meter
        return meters * 100.0
