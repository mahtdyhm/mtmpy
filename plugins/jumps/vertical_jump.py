"""
KinetiQ OS - Vertical Jump Plugin
پلاگین آزمون پرش عمودی (CMJ و Bosco Test) برای ارزیابی توان انفجاری ورزشکاران
"""

from typing import Dict, Any
from core.registry import BaseSportTest, plugin_registry
from core.schemas import SkeletonFrame, MetricSummary
from core.biomechanics.jump_engine import JumpAnalyzer


@plugin_registry.register
class VerticalJumpTest(BaseSportTest):
    """آزمون پرش عمودی تک (CMJ) و تست متوالی توان بی‌هوازی"""

    test_id = "vertical_jump"
    title = "آزمون پرش عمودی و توان انفجاری (CMJ)"
    category = "jumps"
    description = "محاسبه دقیق ارتفاع پرش از طریق زمان پرواز (Flight Time)، شاخص RSI و توان کشسانی عضلات"
    version = "1.0.0"

    def __init__(self, athlete_id: int, config: Dict[str, Any] = None):
        super().__init__(athlete_id=athlete_id, config=config)
        test_mode = self.config.get("jump_mode", "cmj")  # 'cmj' یا 'repeated_bosco'
        self.analyzer = JumpAnalyzer(mode=test_mode)

    def start(self) -> None:
        self.reset()
        self.is_active = True
        self.state = "READY"

    def process_frame(self, frame: SkeletonFrame) -> Dict[str, Any]:
        if not self.is_active or not frame.detected_person:
            return {"status": "NO_PERSON"}

        self.frames_history.append(frame)
        telemetry = self.analyzer.update_frame(frame)
        telemetry["status"] = "TESTING"
        return telemetry

    def finish(self) -> MetricSummary:
        self.is_active = False
        self.state = "FINISHED"

        summary = self.analyzer.compute_summary()
        max_h = summary["max_height_cm"]

        feedback = []
        if max_h >= 55.0:
            feedback.append("توان انفجاری در سطح نخبگان (Elite)؛ مناسب برای شوت‌های پرشی هندبال و جهش‌های دروازه‌بانی.")
        elif max_h >= 40.0:
            feedback.append("توان پرش در سطح نیمه‌حرفه‌ای و مطلوب؛ تمرینات پلیومتریک برای افزایش شاخص RSI توصیه می‌شود.")
        else:
            feedback.append("پتانسیل پرش پایه؛ نیاز به تقویت عضلات چهارسر و دوقلو در برنامه بدنسازی.")

        if summary.get("fatigue_index_percent", 0.0) > 25.0:
            feedback.append(f"افت توان ناشی از خستگی بالا ({summary['fatigue_index_percent']}٪)؛ نیاز به افزایش استقامت بی‌هوازی لاکتیک.")

        return MetricSummary(
            test_id=self.test_id,
            athlete_id=self.athlete_id,
            primary_metric_name="max_jump_height_cm",
            primary_metric_value=max_h,
            primary_metric_unit="cm",
            detailed_metrics=summary,
            feedback_notes=feedback,
            passed=max_h > 15.0,
        )
