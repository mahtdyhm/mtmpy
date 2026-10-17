"""
KinetiQ OS - Handball Tactical & Biomechanics Engine
تحلیل تخصصی هندبال: شمارش ۳ گام، خطای رانینگ، خطای دبل، زاویه شوت و پوشش دروازه‌بانی
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from core.schemas import SkeletonFrame, Landmark3D


class HandballViolationRecord:
    """رکورد ثبت یک خطای داوری هندبال همراه با تایم‌استمپ"""
    def __init__(self, violation_type: str, timestamp_s: float, step_count: int, description: str):
        self.violation_type = violation_type  # 'TRAVELING', 'DOUBLE_DRIBBLE'
        self.timestamp_s = timestamp_s
        self.step_count = step_count
        self.description = description


class HandballEngine:
    """موتور قوانین داوری، گام‌شماری و بیومکانیک شوت هندبال"""

    def __init__(self):
        self.has_possession: bool = False
        self.steps_count: int = 0
        self.is_dribbling: bool = False
        self.dribble_stopped: bool = False
        self.violations: List[HandballViolationRecord] = []

        # ردیابی وضعیت پاها برای تشخیص گام
        self.left_foot_down: bool = True
        self.right_foot_down: bool = True
        self.last_step_foot: Optional[str] = None

        # ردیابی شوت
        self.max_arm_cocking_angle: float = 0.0
        self.max_hand_velocity: float = 0.0
        self.last_wrist_pos: Optional[Tuple[float, float, float]] = None

    def update_ball_possession(self, ball_pos: Optional[Tuple[float, float, float]], lms: Dict[str, Landmark3D]) -> bool:
        """بررسی فاصله توپ با دست‌ها جهت تعیین مالکیت توپ"""
        if ball_pos is None or "left_wrist" not in lms or "right_wrist" not in lms:
            return self.has_possession

        bx, by, _ = ball_pos
        lw = lms["left_wrist"]
        rw = lms["right_wrist"]

        dist_left = math.sqrt((bx - lw.x) ** 2 + (by - lw.y) ** 2)
        dist_right = math.sqrt((bx - rw.x) ** 2 + (by - rw.y) ** 2)

        # اگر توپ در فاصله کمتر از ۱۰٪ ابعاد کادر نسبت به یکی از دست‌ها باشد
        in_hands = (dist_left < 0.12) or (dist_right < 0.12)
        if in_hands and not self.has_possession:
            self.has_possession = True
        return self.has_possession

    def process_step_rhythm(self, frame: SkeletonFrame, is_dribbling: bool = False) -> Dict[str, Any]:
        """
        بررسی تماس پاها با زمین و شمارش ۳ گام هندبال و بررسی خطای رانینگ و دبل
        """
        lms = frame.landmarks
        t_sec = frame.timestamp_ms / 1000.0
        if "left_ankle" not in lms or "right_ankle" not in lms:
            return {"steps": self.steps_count, "has_violation": False}

        la_y = lms["left_ankle"].y
        ra_y = lms["right_ankle"].y

        # آستانه فرود پا روی زمین (تماس کف پا با زمین)
        left_on_floor = la_y > 0.88
        right_on_floor = ra_y > 0.88

        # تشخیص گام جدید پای چپ
        if left_on_floor and not self.left_foot_down and self.last_step_foot != "LEFT":
            if self.has_possession and not is_dribbling:
                self.steps_count += 1
                self.last_step_foot = "LEFT"
            self.left_foot_down = True
        elif not left_on_floor:
            self.left_foot_down = False

        # تشخیص گام جدید پای راست
        if right_on_floor and not self.right_foot_down and self.last_step_foot != "RIGHT":
            if self.has_possession and not is_dribbling:
                self.steps_count += 1
                self.last_step_foot = "RIGHT"
            self.right_foot_down = True
        elif not right_on_floor:
            self.right_foot_down = False

        # بررسی خطای رانینگ (Traveling): بیش از ۳ گام بدون دریبل
        if self.steps_count > 3:
            viol = HandballViolationRecord(
                violation_type="TRAVELING",
                timestamp_s=t_sec,
                step_count=self.steps_count,
                description=f"خطای رانینگ: برداشتن {self.steps_count} گام بدون دریبل کردن توپ",
            )
            self.violations.append(viol)

        # بررسی خطای دبل (Double Dribble): دریبل -> مهار با دو دست -> شروع دریبل مجدد
        if self.dribble_stopped and is_dribbling:
            viol = HandballViolationRecord(
                violation_type="DOUBLE_DRIBBLE",
                timestamp_s=t_sec,
                step_count=self.steps_count,
                description="خطای دبل: شروع مجدد دریبل پس از مهار کامل توپ با دو دست",
            )
            self.violations.append(viol)

        # تحلیل زاویه پرتاب دست شوت‌زن
        if "right_shoulder" in lms and "right_elbow" in lms and "right_wrist" in lms:
            rs, re, rw = lms["right_shoulder"], lms["right_elbow"], lms["right_wrist"]
            v1 = (rs.x - re.x, rs.y - re.y)
            v2 = (rw.x - re.x, rw.y - re.y)
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
            if mag1 * mag2 > 0:
                angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))
                self.max_arm_cocking_angle = max(self.max_arm_cocking_angle, angle)

        return {
            "steps_count": self.steps_count,
            "has_possession": self.has_possession,
            "violations_count": len(self.violations),
            "latest_violation": self.violations[-1].violation_type if self.violations else None,
            "arm_angle": round(self.max_arm_cocking_angle, 1),
        }

    def calculate_goalkeeper_coverage(
        self,
        lms: Dict[str, Landmark3D],
        goal_width_m: float = 3.0,
        goal_height_m: float = 2.0,
    ) -> Dict[str, float]:
        """
        محاسبه هندسی درصد پوشش چارچوب دروازه هندبال توسط دست‌ها و پاهای دروازه‌بان
        ابعاد رسمی دروازه هندبال: ۳ متر عرض در ۲ متر ارتفاع (مساحت ۶ متر مربع)
        """
        if "left_wrist" not in lms or "right_wrist" not in lms or "left_ankle" not in lms or "right_ankle" not in lms:
            return {"coverage_percent": 0.0}

        # طول باز شدن دست‌ها و پاها
        lw, rw = lms["left_wrist"], lms["right_wrist"]
        la, ra = lms["left_ankle"], lms["right_ankle"]

        arm_spread = math.sqrt((rw.x - lw.x) ** 2 + (rw.y - lw.y) ** 2)
        leg_spread = math.sqrt((ra.x - la.x) ** 2 + (ra.y - la.y) ** 2)

        # مدل‌سازی تقریب چندضلعی مساحت پوشش دفاعی
        coverage_ratio = min(1.0, (arm_spread * 0.6 + leg_spread * 0.5) * 1.5)
        return {
            "coverage_percent": round(coverage_ratio * 100.0, 1),
            "arm_spread_ratio": round(arm_spread, 2),
            "leg_spread_ratio": round(leg_spread, 2),
        }
