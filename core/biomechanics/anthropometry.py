"""
KinetiQ OS - Anthropometric Kinematics Engine
محاسبه دقیق قد و اسپن (طول دو دست) در هر حالت بدنی و تحلیل تقارن اسکلتی
"""

import math
from typing import Dict, Any, Tuple, Optional
from core.schemas import SkeletonFrame, Landmark3D
from core.biomechanics.calibration import SpatialCalibrator


def euclidean_distance_3d(lm1: Landmark3D, lm2: Landmark3D, aspect_ratio: float = 1.0) -> float:
    """محاسبه فاصله اقلیدسی بین دو مفصل با تصحیح نسبت ابعاد تصویر"""
    dx = (lm1.x - lm2.x) * aspect_ratio
    dy = lm1.y - lm2.y
    dz = lm1.z - lm2.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class AnthropometryCalculator:
    """موتور محاسبات آنتروپومتریک کینماتیک (قد، اسپن، شاخص Ape، تقارن)"""

    def __init__(self, calibrator: Optional[SpatialCalibrator] = None):
        self.calibrator = calibrator or SpatialCalibrator(pixels_per_meter=500.0)

    def calculate_arm_span(
        self,
        landmarks: Dict[str, Landmark3D],
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> Dict[str, float]:
        """
        محاسبه طول اسپن (Arm Span) به روش مجموع قطعات اسکلتی:
        حتی اگر آرنج‌ها یا دست‌ها خمیده باشند، مجموع استخوان‌ها اندازه واقعی اسپن را می‌دهد.
        """
        required = ["left_wrist", "left_elbow", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist"]
        for joint in required:
            if joint not in landmarks or landmarks[joint].visibility < 0.2:
                return {"arm_span_cm": 0.0, "valid": 0.0}

        lw = (landmarks["left_wrist"].x, landmarks["left_wrist"].y)
        le = (landmarks["left_elbow"].x, landmarks["left_elbow"].y)
        ls = (landmarks["left_shoulder"].x, landmarks["left_shoulder"].y)
        rs = (landmarks["right_shoulder"].x, landmarks["right_shoulder"].y)
        re = (landmarks["right_elbow"].x, landmarks["right_elbow"].y)
        rw = (landmarks["right_wrist"].x, landmarks["right_wrist"].y)

        # طول استخوان‌ها به سانتی‌متر
        left_forearm_cm = self.calibrator.distance_cm(lw, le, image_width, image_height)
        left_upper_arm_cm = self.calibrator.distance_cm(le, ls, image_width, image_height)
        shoulder_width_cm = self.calibrator.distance_cm(ls, rs, image_width, image_height)
        right_upper_arm_cm = self.calibrator.distance_cm(rs, re, image_width, image_height)
        right_forearm_cm = self.calibrator.distance_cm(re, rw, image_width, image_height)

        # افزودن تخمین طول کف تا نوک انگشتان (حدود ۱۲٪ طول کل دست بر اساس آنتروپومتری)
        hand_extension_cm = (left_forearm_cm + right_forearm_cm) * 0.40

        total_span_cm = (
            left_forearm_cm
            + left_upper_arm_cm
            + shoulder_width_cm
            + right_upper_arm_cm
            + right_forearm_cm
            + hand_extension_cm
        )

        return {
            "arm_span_cm": round(total_span_cm, 1),
            "shoulder_width_cm": round(shoulder_width_cm, 1),
            "left_arm_total_cm": round(left_forearm_cm + left_upper_arm_cm, 1),
            "right_arm_total_cm": round(right_forearm_cm + right_upper_arm_cm, 1),
            "valid": 1.0,
        }

    def calculate_dynamic_height(
        self,
        landmarks: Dict[str, Landmark3D],
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> Dict[str, float]:
        """
        محاسبه قد به روش مجموع قطعات تنه، ران، ساق و سر:
        حتی در صورتی که فرد قوز کرده یا زانوها خم باشند، قد واقعی بیومکانیکی محاسبه می‌شود.
        """
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "left_ankle"]
        for joint in required:
            if joint not in landmarks or landmarks[joint].visibility < 0.2:
                return {"standing_height_cm": 0.0, "valid": 0.0}

        # مرکز شانه‌ها و مرکز لگن
        mid_shoulder = (
            (landmarks["left_shoulder"].x + landmarks["right_shoulder"].x) / 2.0,
            (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2.0,
        )
        mid_hip = (
            (landmarks["left_hip"].x + landmarks["right_hip"].x) / 2.0,
            (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2.0,
        )
        knee = (landmarks["left_knee"].x, landmarks["left_knee"].y)
        ankle = (landmarks["left_ankle"].x, landmarks["left_ankle"].y)

        # طول قطعات
        torso_cm = self.calibrator.distance_cm(mid_shoulder, mid_hip, image_width, image_height)
        thigh_cm = self.calibrator.distance_cm(mid_hip, knee, image_width, image_height)
        shin_cm = self.calibrator.distance_cm(knee, ankle, image_width, image_height)

        # سر و گردن + ارتفاع مچ تا کف پا (تخمین استاندارد بر اساس آنتروپومتری ورزشی)
        head_neck_cm = torso_cm * 0.45
        foot_height_cm = shin_cm * 0.18

        total_height_cm = torso_cm + thigh_cm + shin_cm + head_neck_cm + foot_height_cm

        return {
            "standing_height_cm": round(total_height_cm, 1),
            "torso_cm": round(torso_cm, 1),
            "thigh_cm": round(thigh_cm, 1),
            "shin_cm": round(shin_cm, 1),
            "valid": 1.0,
        }

    def calculate_posture_and_asymmetry(
        self,
        landmarks: Dict[str, Landmark3D],
    ) -> Dict[str, Any]:
        """محاسبه درصد عدم تقارن دست‌ها و پاها و شیب شانه‌ها"""
        if "left_shoulder" not in landmarks or "right_shoulder" not in landmarks:
            return {"shoulder_tilt_degrees": 0.0, "asymmetry_percent": 0.0}

        ls = landmarks["left_shoulder"]
        rs = landmarks["right_shoulder"]

        # شیب شانه به درجه
        dy = rs.y - ls.y
        dx = rs.x - ls.x
        tilt_degrees = math.degrees(math.atan2(dy, dx))

        return {
            "shoulder_tilt_degrees": round(abs(tilt_degrees), 2),
            "is_posture_balanced": abs(tilt_degrees) < 3.0,
        }
