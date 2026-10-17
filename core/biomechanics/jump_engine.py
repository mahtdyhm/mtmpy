"""
KinetiQ OS - Vertical Jump & Power Engine
موتور تحلیلی پرش‌های عمودی (CMJ)، توان بی‌هوازی، زمان پرواز و شاخص RSI
"""

import math
from typing import Dict, Any, List, Optional
from core.schemas import SkeletonFrame, Landmark3D


GRAVITY = 9.80665  # شتاب گرانش زمین بر حسب متر بر مجذور ثانیه


class JumpRepRecord:
    """اطلاعات ثبت‌شده برای یک تکرار پرش"""
    def __init__(
        self,
        rep_number: int,
        takeoff_time_s: float,
        landing_time_s: float,
        flight_time_s: float,
        jump_height_cm: float,
        contact_time_s: float = 0.0,
        rsi: float = 0.0,
        squat_depth_deg: float = 0.0,
        is_valid: bool = True,
        invalid_reason: Optional[str] = None,
    ):
        self.rep_number = rep_number
        self.takeoff_time_s = takeoff_time_s
        self.landing_time_s = landing_time_s
        self.flight_time_s = flight_time_s
        self.jump_height_cm = jump_height_cm
        self.contact_time_s = contact_time_s
        self.rsi = rsi
        self.squat_depth_deg = squat_depth_deg
        self.is_valid = is_valid
        self.invalid_reason = invalid_reason


class JumpAnalyzer:
    """ماشین حالت و تحلیلگر کینماتیک آزمون‌های پرش (تک پرش CMJ و پرش‌های متوالی Bosco)"""

    def __init__(self, mode: str = "cmj"):
        self.mode = mode  # 'cmj' یا 'repeated_bosco'
        self.state = "STAND"  # 'STAND', 'SQUAT', 'FLIGHT', 'LANDED'
        self.reps: List[JumpRepRecord] = []

        # ردیابی خط پایه لگن و پاها
        self.baseline_hip_y: Optional[float] = None
        self.lowest_knee_angle: float = 180.0
        self.takeoff_timestamp: Optional[float] = None
        self.last_landing_timestamp: Optional[float] = None

        self.in_flight: bool = False
        self.consecutive_flight_frames: int = 0
        self.consecutive_ground_frames: int = 0

    def calculate_knee_angle(self, hip: Landmark3D, knee: Landmark3D, ankle: Landmark3D) -> float:
        """محاسبه زاویه مفصل زانو به درجه با استفاده از ضرب داخلی بردارها"""
        v1 = (hip.x - knee.x, hip.y - knee.y)
        v2 = (ankle.x - knee.x, ankle.y - knee.y)

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 * mag2 <= 0:
            return 180.0

        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    def update_frame(self, frame: SkeletonFrame) -> Dict[str, Any]:
        """به‌روزرسانی ماشین حالت با دریافت فریم جدید و استخراج تله‌متری آنی"""
        lms = frame.landmarks
        if "left_ankle" not in lms or "left_hip" not in lms or "left_knee" not in lms:
            return {"state": self.state, "height_cm": 0.0, "reps_count": len(self.reps)}

        t_sec = frame.timestamp_ms / 1000.0
        hip_y = lms["left_hip"].y
        ankle_y = lms["left_ankle"].y
        knee_angle = self.calculate_knee_angle(lms["left_hip"], lms["left_knee"], lms["left_ankle"])

        # مقداردهی اولیه خط پایه ایستادن
        if self.baseline_hip_y is None:
            self.baseline_hip_y = hip_y

        # ردیابی پایین‌ترین خمش زانو قبل از پرش
        if self.state in ["STAND", "SQUAT"]:
            if knee_angle < 150.0:
                self.state = "SQUAT"
                self.lowest_knee_angle = min(self.lowest_knee_angle, knee_angle)
            elif self.state == "SQUAT" and knee_angle > 165.0:
                pass  # آماده برای جهش

        # تشخیص فاز پرواز: مچ پا بالاتر از سطح زمین قرار گرفته و لگن صعود می‌کند
        # در تصویر، مقادیر y با بالا رفتن کمتر می‌شوند
        is_airborne = (ankle_y < 0.90) and (hip_y < self.baseline_hip_y - 0.03)

        if is_airborne:
            self.consecutive_flight_frames += 1
            self.consecutive_ground_frames = 0
            if self.consecutive_flight_frames >= 2 and not self.in_flight:
                self.in_flight = True
                self.state = "FLIGHT"
                self.takeoff_timestamp = t_sec
        else:
            self.consecutive_ground_frames += 1
            self.consecutive_flight_frames = 0
            if self.consecutive_ground_frames >= 2 and self.in_flight:
                # فرود روی زمین (Landing)
                self.in_flight = False
                self.state = "LANDED"
                landing_time = t_sec

                if self.takeoff_timestamp is not None:
                    flight_time = max(0.05, landing_time - self.takeoff_timestamp)

                    # فرمول زمان پرواز: h = 1/8 * g * t^2
                    height_m = (1.0 / 8.0) * GRAVITY * (flight_time ** 2)
                    height_cm = height_m * 100.0

                    # زمان تماس با زمین برای پرش‌های متوالی
                    contact_time = 0.0
                    rsi = 0.0
                    if self.last_landing_timestamp is not None:
                        contact_time = max(0.05, self.takeoff_timestamp - self.last_landing_timestamp)
                        rsi = round(flight_time / contact_time, 2)

                    # اعتبارسنجی پرش (تست جمع نکردن بیش از حد پاها)
                    is_valid = True
                    invalid_reason = None
                    if flight_time > 1.2:  # پرش بالای ۹۰ سانتی‌متر نیازمند بررسی
                        is_valid = False
                        invalid_reason = "پرواز غیرطبیعی یا عدم تشخیص دقیق فریم فرود"

                    rep_rec = JumpRepRecord(
                        rep_number=len(self.reps) + 1,
                        takeoff_time_s=self.takeoff_timestamp,
                        landing_time_s=landing_time,
                        flight_time_s=round(flight_time, 3),
                        jump_height_cm=round(height_cm, 1),
                        contact_time_s=round(contact_time, 3),
                        rsi=rsi,
                        squat_depth_deg=round(self.lowest_knee_angle, 1),
                        is_valid=is_valid,
                        invalid_reason=invalid_reason,
                    )
                    self.reps.append(rep_rec)
                    self.last_landing_timestamp = landing_time
                    self.lowest_knee_angle = 180.0

        last_height = self.reps[-1].jump_height_cm if self.reps else 0.0
        return {
            "state": self.state,
            "knee_angle": round(knee_angle, 1),
            "reps_count": len(self.reps),
            "last_jump_height_cm": last_height,
            "in_flight": self.in_flight,
        }

    def compute_summary(self) -> Dict[str, Any]:
        """محاسبه خروجی جامع پرش‌ها، میانگین، اوج و شاخص خستگی"""
        if not self.reps:
            return {"total_reps": 0, "max_height_cm": 0.0, "mean_height_cm": 0.0}

        valid_reps = [r for r in self.reps if r.is_valid]
        heights = [r.jump_height_cm for r in valid_reps] if valid_reps else [0.0]
        max_h = max(heights)
        mean_h = sum(heights) / len(heights)

        # شاخص افت خستگی (Fatigue Index) در تست‌های متوالی Bosco
        fatigue_index = 0.0
        if len(heights) >= 4:
            first_half = sum(heights[: len(heights) // 2]) / (len(heights) // 2)
            second_half = sum(heights[len(heights) // 2 :]) / (len(heights) - len(heights) // 2)
            if first_half > 0:
                fatigue_index = round(((first_half - second_half) / first_half) * 100.0, 1)

        mean_rsi = round(sum(r.rsi for r in valid_reps if r.rsi > 0) / max(1, len([r for r in valid_reps if r.rsi > 0])), 2)

        return {
            "total_reps": len(self.reps),
            "valid_reps_count": len(valid_reps),
            "max_height_cm": round(max_h, 1),
            "mean_height_cm": round(mean_h, 1),
            "mean_rsi": mean_rsi,
            "fatigue_index_percent": max(0.0, fatigue_index),
            "reps_detail": [
                {
                    "rep": r.rep_number,
                    "height_cm": r.jump_height_cm,
                    "flight_s": r.flight_time_s,
                    "contact_s": r.contact_time_s,
                    "rsi": r.rsi,
                    "squat_deg": r.squat_depth_deg,
                }
                for r in self.reps
            ],
        }
