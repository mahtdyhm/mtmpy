"""
KinetiQ OS - Handball 3-Step & Violations Plugin
پلاگین آزمون ریتم ۳ گام و تشخیص خطاهای رانینگ و دبل هندبال
"""

from typing import Dict, Any
from core.registry import BaseSportTest, plugin_registry
from core.schemas import SkeletonFrame, MetricSummary
from core.sports.handball_engine import HandballEngine


@plugin_registry.register
class HandballThreeStepTest(BaseSportTest):
    """آزمون گام‌شماری، تحلیل خطای رانینگ و ارزیابی شوت هندبال"""

    test_id = "handball_three_step"
    title = "آزمون ریتم ۳ گام و خطای رانینگ هندبال"
    category = "handball"
    description = "شمارش هوشمند گام‌های بازیکن با توپ، تشخیص خطای رانینگ (بیش از ۳ گام) و دبل، و ارزیابی زاویه دست شوت‌زن"
    version = "1.0.0"

    def __init__(self, athlete_id: int, config: Dict[str, Any] = None):
        super().__init__(athlete_id=athlete_id, config=config)
        self.engine = HandballEngine()

    def start(self) -> None:
        self.reset()
        self.is_active = True
        self.state = "PLAYING"
        # در شروع آزمون مالکیت توپ را فعال می‌کنیم
        self.engine.has_possession = True

    def process_frame(self, frame: SkeletonFrame) -> Dict[str, Any]:
        if not self.is_active or not frame.detected_person:
            return {"status": "NO_PERSON"}

        self.frames_history.append(frame)
        telemetry = self.engine.process_step_rhythm(frame, is_dribbling=self.config.get("is_dribbling", False))
        telemetry["status"] = "ACTIVE"
        return telemetry

    def finish(self) -> MetricSummary:
        self.is_active = False
        self.state = "FINISHED"

        steps = self.engine.steps_count
        violations = self.engine.violations
        has_running = any(v.violation_type == "TRAVELING" for v in violations)

        feedback = []
        if steps == 3 and not has_running:
            feedback.append("ریتم ۳ گام استاندارد هندبال (۱-۲-۳) کاملاً بی‌نقص و قانونی اجرا شد.")
        elif steps < 3 and not has_running:
            feedback.append(f"اجرای شتاب‌گرفته با {steps} گام؛ بدون خطا.")
        else:
            feedback.append(f"خطای رانینگ محرز! ورزشکار {steps} گام بدون دریبل توپ برداشت.")

        if self.engine.max_arm_cocking_angle > 90.0:
            feedback.append(f"زاویه آماده‌سازی شوت مطلوب ({self.engine.max_arm_cocking_angle:.1f} درجه).")

        return MetricSummary(
            test_id=self.test_id,
            athlete_id=self.athlete_id,
            primary_metric_name="steps_taken",
            primary_metric_value=float(steps),
            primary_metric_unit="steps",
            detailed_metrics={
                "steps_taken": steps,
                "has_traveling_violation": has_running,
                "violations_total": len(violations),
                "max_arm_cocking_angle": round(self.engine.max_arm_cocking_angle, 1),
            },
            feedback_notes=feedback,
            passed=not has_running,
        )
