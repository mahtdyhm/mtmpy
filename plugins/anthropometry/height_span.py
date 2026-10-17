"""
KinetiQ OS - Height & Arm Span Plugin
پلاگین آزمون آنتروپومتری و استعدادیابی قد و اسپن
"""

from typing import Dict, Any, List
import numpy as np
from core.registry import BaseSportTest, plugin_registry
from core.schemas import SkeletonFrame, MetricSummary
from core.biomechanics.anthropometry import AnthropometryCalculator
from core.biomechanics.calibration import SpatialCalibrator


@plugin_registry.register
class HeightSpanTest(BaseSportTest):
    """آزمون اندازه‌گیری آنتروپومتریک دقیق قد و طول دو دست (اسپن)"""

    test_id = "height_span"
    title = "سنجش قد و اسپن (آنتروپومتری)"
    category = "anthropometry"
    description = "محاسبه دقیق قد ایستاده، طول دو دست (اسپن) و شاخص میمونی (Ape Index) در هر حالت بدنی"
    version = "1.0.0"

    def __init__(self, athlete_id: int, config: Dict[str, Any] = None):
        super().__init__(athlete_id=athlete_id, config=config)
        self.calibrator = SpatialCalibrator(
            pixels_per_meter=self.config.get("pixels_per_meter", 500.0)
        )
        self.calculator = AnthropometryCalculator(calibrator=self.calibrator)
        self.height_samples: List[float] = []
        self.span_samples: List[float] = []

    def start(self) -> None:
        self.reset()
        self.is_active = True
        self.state = "COLLECTING"

    def process_frame(self, frame: SkeletonFrame) -> Dict[str, Any]:
        if not self.is_active or not frame.detected_person:
            return {"status": "NO_PERSON"}

        self.frames_history.append(frame)

        # محاسبه قد و اسپن در این فریم
        span_res = self.calculator.calculate_arm_span(frame.landmarks)
        height_res = self.calculator.calculate_dynamic_height(frame.landmarks)
        posture_res = self.calculator.calculate_posture_and_asymmetry(frame.landmarks)

        if span_res.get("valid", 0.0) > 0.5:
            self.span_samples.append(span_res["arm_span_cm"])

        if height_res.get("valid", 0.0) > 0.5:
            self.height_samples.append(height_res["standing_height_cm"])

        current_height = np.median(self.height_samples) if self.height_samples else 0.0
        current_span = np.median(self.span_samples) if self.span_samples else 0.0

        ape_index = round(current_span / current_height, 2) if current_height > 0 else 1.0

        return {
            "status": "TRACKING",
            "live_height_cm": round(float(current_height), 1),
            "live_span_cm": round(float(current_span), 1),
            "ape_index": ape_index,
            "shoulder_tilt": posture_res.get("shoulder_tilt_degrees", 0.0),
            "sample_count": len(self.height_samples),
        }

    def finish(self) -> MetricSummary:
        self.is_active = False
        self.state = "FINISHED"

        final_height = float(np.median(self.height_samples)) if self.height_samples else 0.0
        final_span = float(np.median(self.span_samples)) if self.span_samples else 0.0
        ape_index = round(final_span / final_height, 2) if final_height > 0 else 1.0

        # توصیه پست بازی هندبال بر اساس شاخص Ape و اسپن
        feedback = []
        if ape_index > 1.03:
            feedback.append("شاخص اسپن به قد عالی (> 1.03)؛ بسیار مستعد برای پست‌های دفاع وسط، بغل زن و دروازه‌بانی.")
        elif ape_index < 0.98:
            feedback.append("طول دست نسبت به قد کمتر از میانگین است؛ مناسب برای پست‌های سرعتی گوشه (Wing).")
        else:
            feedback.append("نسبت اسپن به قد متناسب و در رنج طبیعی استاندارد.")

        return MetricSummary(
            test_id=self.test_id,
            athlete_id=self.athlete_id,
            primary_metric_name="standing_height_cm",
            primary_metric_value=round(final_height, 1),
            primary_metric_unit="cm",
            detailed_metrics={
                "standing_height_cm": round(final_height, 1),
                "arm_span_cm": round(final_span, 1),
                "ape_index": ape_index,
                "total_frames_analyzed": len(self.frames_history),
            },
            feedback_notes=feedback,
            passed=final_height > 50.0,
        )
