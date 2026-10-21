"""
KinetiQ OS - Offline YOLO-Pose Estimator
موتور تخمین پوز ۱۰۰٪ آفلاین مبتنی بر YOLO11 / YOLOv8 Pose
بسیار سریع (بیش از ۶۰ تا ۱۲۰ فریم بر ثانیه)، دقیق و بدون نیاز به اینترنت
"""

from typing import Optional, Dict
import numpy as np
from core.ai.base_estimator import BasePoseEstimator
from core.schemas import SkeletonFrame, Landmark3D
from core.logging import logger

# تطبیق ۱۷ مفصل استاندارد COCO
YOLO_KEYPOINTS_MAP = {
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


class YoloPoseEstimator(BasePoseEstimator):
    """پیاده‌سازی موتور پوز آفلاین با مدل YOLO11-Nano Pose"""

    def __init__(self, model_name: str = "yolo11n-pose.pt", conf_threshold: float = 0.5):
        super().__init__(mode_name="offline_yolo")
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None

    def initialize(self) -> None:
        """بارگذاری مدل محلی YOLO-Pose"""
        if self.is_initialized:
            return
        try:
            from ultralytics import YOLO
            from pathlib import Path
            from config.settings import settings

            target_path = Path(self.model_name)
            if not target_path.is_file():
                for base in [getattr(settings, "BUNDLE_DIR", None), settings.BASE_DIR]:
                    if base and (base / self.model_name).is_file():
                        target_path = base / self.model_name
                        break

            self.model = YOLO(str(target_path))
            self.is_initialized = True
            logger.info(f"موتور پوز آفلاین YOLO با مدل '{target_path}' با موفقیت لود شد.")
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل YOLO: {e}")
            raise

    def estimate_frame(self, image_bgr: np.ndarray, frame_index: int = 0, timestamp_ms: float = 0.0) -> Optional[SkeletonFrame]:
        """پردازش فریم و استخراج ۱۷ مفصل استاندارد"""
        if not self.is_initialized:
            self.initialize()

        if image_bgr is None or image_bgr.size == 0:
            return None

        h, w = image_bgr.shape[:2]
        # استنتاج تصویر با YOLO بدون چاپ لاگ‌های اضافی
        results = self.model(image_bgr, verbose=False, conf=self.conf_threshold)
        if not results or len(results) == 0 or results[0].keypoints is None:
            return SkeletonFrame(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                landmarks={},
                detected_person=False,
                ai_mode="offline",
            )

        boxes = results[0].boxes
        keypoints = results[0].keypoints

        if len(keypoints) == 0 or keypoints.data is None or len(keypoints.data) == 0:
            return SkeletonFrame(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                landmarks={},
                detected_person=False,
                ai_mode="offline",
            )

        # انتخاب اولین شخص (یا بزرگترین فرد در کادر)
        person_kpts = keypoints.data[0].cpu().numpy()  # آرایه [17, 3] شامل (x, y, conf)

        landmarks_dict: Dict[str, Landmark3D] = {}
        for idx, (kx, ky, conf) in enumerate(person_kpts):
            name = YOLO_KEYPOINTS_MAP.get(idx, f"keypoint_{idx}")
            landmarks_dict[name] = Landmark3D(
                name=name,
                x=float(kx / w),  # نرمالایز افقی [0, 1]
                y=float(ky / h),  # نرمالایز عمودی [0, 1]
                z=0.0,
                visibility=float(conf),
                presence=1.0 if float(conf) > 0.2 else 0.0,
            )

        return SkeletonFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            landmarks=landmarks_dict,
            detected_person=True,
            ai_mode="offline",
        )

    def close(self) -> None:
        self.model = None
        self.is_initialized = False
