"""
KinetiQ OS - Enhanced Object & Ball Tracker
ردیابی هوشمند و پرسرعت توپ هندبال با فیلتر کرویت، تخمین حرکت (Kinematic Extrapolation) و محاسبه سرعت پرتاب
"""

import math
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import cv2
import numpy as np


class ObjectTracker:
    """ردیاب اپتیکال ارتقایافته برای تشخیص، ردیابی پیوسته و تخمین سرعت توپ ورزشی"""

    BALL_PRESETS = {
        "orange": {"lower": (5, 90, 90), "upper": (25, 255, 255)},
        "yellow_green": {"lower": (26, 80, 80), "upper": (45, 255, 255)},
        "white_light": {"lower": (0, 0, 180), "upper": (180, 50, 255)},
    }

    def __init__(self, target_name: str = "handball", trajectory_len: int = 30):
        self.target_name = target_name
        self.last_position: Optional[Tuple[float, float, float]] = None  # (cx, cy, radius)
        self.last_velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy)
        self.is_tracking: bool = False
        self.occlusion_frames: int = 0
        self.max_occlusion_frames: int = 5
        self.trajectory: deque = deque(maxlen=trajectory_len)  # ذخیره (timestamp_ms, x, y)
        self.last_timestamp_ms: Optional[float] = None

    def track_hsv_sphere(
        self,
        frame_bgr: np.ndarray,
        lower_hsv: Tuple[int, int, int] = (5, 100, 100),
        upper_hsv: Tuple[int, int, int] = (25, 255, 255),
    ) -> Optional[Tuple[float, float, float]]:
        """متد پایه و سازگار برای تشخیص تک‌فریم توپ بر اساس ماسک رنگی و کرویت"""
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

        # اعتبارسنجی کانتورها با فیلتر کرویت
        best_candidate = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # ضریب کرویت (Circularity) برای تفکیک توپ از موانع کشیده یا خطوط زمین
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            if circularity < 0.45:
                continue

            if area > best_area:
                best_area = area
                best_candidate = cnt

        if best_candidate is None:
            self.is_tracking = False
            return None

        (cx, cy), radius = cv2.minEnclosingCircle(best_candidate)
        if radius < 4:
            self.is_tracking = False
            return None

        norm_pos = (float(cx / w), float(cy / h), float(radius / w))
        self.is_tracking = True
        self.last_position = norm_pos
        self.occlusion_frames = 0
        return norm_pos

    def track_ball_advanced(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float,
        preset: str = "orange",
        pixels_per_meter: float = 500.0,
    ) -> Dict[str, Any]:
        """
        ردیابی پیشرفته همراه با اکستراپولاسیون حرکت در صورت مسدود شدن لحظه‌ای دید (Occlusion Handling)
        و محاسبه بلادرنگ سرعت پرتاب (m/s)
        """
        config = self.BALL_PRESETS.get(preset, self.BALL_PRESETS["orange"])
        detected_pos = self.track_hsv_sphere(frame_bgr, config["lower"], config["upper"])

        current_speed_mps = 0.0

        if detected_pos is not None:
            cx, cy, r = detected_pos
            if self.last_position is not None and self.last_timestamp_ms is not None:
                dt_s = (timestamp_ms - self.last_timestamp_ms) / 1000.0
                if dt_s > 0.001:
                    h, w = frame_bgr.shape[:2]
                    dx_px = (cx - self.last_position[0]) * w
                    dy_px = (cy - self.last_position[1]) * h
                    dist_px = math.sqrt(dx_px ** 2 + dy_px ** 2)
                    dist_m = dist_px / max(1.0, pixels_per_meter)
                    current_speed_mps = dist_m / dt_s

                    self.last_velocity = (dx_px / dt_s / w, dy_px / dt_s / h)

            self.last_position = detected_pos
            self.last_timestamp_ms = timestamp_ms
            self.trajectory.append((timestamp_ms, cx, cy))
            return {
                "detected": True,
                "position": detected_pos,
                "speed_mps": round(current_speed_mps, 2),
                "is_interpolated": False,
                "trajectory_len": len(self.trajectory),
            }

        # در صورت عدم تشخیص در فریم فعلی: اکستراپولاسیون حرکت با بردار سرعت قبلی تا چند فریم
        if self.last_position is not None and self.occlusion_frames < self.max_occlusion_frames:
            self.occlusion_frames += 1
            pred_x = self.last_position[0] + self.last_velocity[0] * 0.033
            pred_y = self.last_position[1] + self.last_velocity[1] * 0.033
            pred_r = self.last_position[2]
            pred_pos = (float(np.clip(pred_x, 0.0, 1.0)), float(np.clip(pred_y, 0.0, 1.0)), pred_r)
            self.trajectory.append((timestamp_ms, pred_pos[0], pred_pos[1]))
            return {
                "detected": True,
                "position": pred_pos,
                "speed_mps": round(current_speed_mps, 2),
                "is_interpolated": True,
                "trajectory_len": len(self.trajectory),
            }

        self.is_tracking = False
        return {
            "detected": False,
            "position": None,
            "speed_mps": 0.0,
            "is_interpolated": False,
            "trajectory_len": len(self.trajectory),
        }

    def clear(self):
        """ریست ردیاب و بافر مسیر حرکت"""
        self.last_position = None
        self.last_velocity = (0.0, 0.0)
        self.is_tracking = False
        self.occlusion_frames = 0
        self.trajectory.clear()
        self.last_timestamp_ms = None
