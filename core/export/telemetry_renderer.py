"""
KinetiQ OS - Video Telemetry Overlay & Scout Card Export Engine
رندرینگ ویدیوهای هایلایت با داشبورد گرافیکی (HUD) و صدور گزارشات استعدادیابی
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from core.schemas import SkeletonFrame, MetricSummary
from config.settings import settings


class VideoTelemetryRenderer:
    """رندرینگ ویدیوی خروجی با ترسیم اسکلت مفاصل و داشبورد زنده در گوشه تصویر"""

    def __init__(self, output_width: int = 1280, output_height: int = 720):
        self.output_width = output_width
        self.output_height = output_height

    def render_overlay_frame(
        self,
        frame_bgr: np.ndarray,
        skeleton: Optional[SkeletonFrame],
        telemetry: Dict[str, Any],
        test_title: str = "آزمون ورزشی",
    ) -> np.ndarray:
        """ترسیم اسکلت نئونی و پنل HUD روی تک‌فریم"""
        canvas = cv2.resize(frame_bgr, (self.output_width, self.output_height))
        h, w = canvas.shape[:2]

        # ۱. رسم اسکلت مفاصل در صورت وجود
        if skeleton and skeleton.detected_person:
            lms = skeleton.landmarks
            # رسم اتصالات اصلی
            connections = [
                ("left_shoulder", "right_shoulder"),
                ("left_shoulder", "left_elbow"),
                ("left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow"),
                ("right_elbow", "right_wrist"),
                ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"),
                ("left_hip", "right_hip"),
                ("left_hip", "left_knee"),
                ("left_knee", "left_ankle"),
                ("right_hip", "right_knee"),
                ("right_knee", "right_ankle"),
            ]
            for j1, j2 in connections:
                if j1 in lms and j2 in lms and lms[j1].visibility > 0.25 and lms[j2].visibility > 0.25:
                    pt1 = (int(lms[j1].x * w), int(lms[j1].y * h))
                    pt2 = (int(lms[j2].x * w), int(lms[j2].y * h))
                    cv2.line(canvas, pt1, pt2, (0, 255, 140), 3)  # خط سبز نئونی

            for name, lm in lms.items():
                if lm.visibility > 0.25:
                    center = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(canvas, center, 6, (0, 229, 255), -1)  # دایره فیروزه‌ای
                    cv2.circle(canvas, center, 8, (255, 255, 255), 1)

            # ۲. محاسبه و ترسیم زوایای مفاصل زانو و آرنج (Joint Angles)
            def angle_3pts(a_name, b_name, c_name):
                if a_name in lms and b_name in lms and c_name in lms:
                    a = (lms[a_name].x * w, lms[a_name].y * h)
                    b = (lms[b_name].x * w, lms[b_name].y * h)
                    c = (lms[c_name].x * w, lms[c_name].y * h)
                    v1 = (a[0] - b[0], a[1] - b[1])
                    v2 = (c[0] - b[0], c[1] - b[1])
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    m1 = np.hypot(v1[0], v1[1])
                    m2 = np.hypot(v2[0], v2[1])
                    if m1 * m2 > 0:
                        rad = np.arccos(np.clip(dot / (m1 * m2), -1.0, 1.0))
                        deg = np.degrees(rad)
                        return b, deg
                return None, None

            # زاویه زانوی راست و چپ
            for hip, knee, ankle in [("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle")]:
                center, deg = angle_3pts(hip, knee, ankle)
                if center and deg is not None:
                    pos = (int(center[0] + 12), int(center[1] + 5))
                    cv2.putText(canvas, f"{int(deg)} deg", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

            # زاویه آرنج راست و چپ
            for shoulder, elbow, wrist in [("left_shoulder", "left_elbow", "left_wrist"), ("right_shoulder", "right_elbow", "right_wrist")]:
                center, deg = angle_3pts(shoulder, elbow, wrist)
                if center and deg is not None:
                    pos = (int(center[0] + 12), int(center[1] + 5))
                    cv2.putText(canvas, f"{int(deg)} deg", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 2)

            # خط راهنمای اندازه اسپن بین مچ‌ها
            if "left_wrist" in lms and "right_wrist" in lms:
                pw1 = (int(lms["left_wrist"].x * w), int(lms["left_wrist"].y * h))
                pw2 = (int(lms["right_wrist"].x * w), int(lms["right_wrist"].y * h))
                cv2.line(canvas, pw1, pw2, (0, 200, 255), 1)

        # ۲. پنل شیشه‌ای تله‌متری (HUD Dashboard) در گوشه بالا سمت چپ
        overlay = canvas.copy()
        cv2.rectangle(overlay, (20, 20), (380, 220), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (20, 20), (380, 220), (0, 255, 140), 2)

        # عنوان تست
        cv2.putText(canvas, f"KinetiQ OS | {test_title}", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.line(canvas, (35, 68), (365, 68), (60, 70, 80), 1)

        # پارامترهای لحظه‌ای
        y_offset = 100
        for key, val in telemetry.items():
            if y_offset > 200:
                break
            text = f"{key}: {val}"
            cv2.putText(canvas, text, (35, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
            y_offset += 25

        return canvas


class ScoutReportExporter:
    """صدور داده‌های بیومکانیک به صورت گزارش JSON، فایل CSV و خلاصه استعدادیابی"""

    @staticmethod
    def export_csv(metrics_history: List[Dict[str, Any]], output_path: Path) -> Path:
        """خروجی سری‌های زمانی کینماتیک به فرمت استاندارد CSV"""
        if not metrics_history:
            output_path.write_text("timestamp_ms,metric,value\n", encoding="utf-8")
            return output_path

        headers = list(metrics_history[0].keys())
        lines = [",".join(headers)]
        for row in metrics_history:
            lines.append(",".join(str(row.get(h, "")) for h in headers))

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    @staticmethod
    def export_scout_card_json(metric_summary: MetricSummary, athlete_info: Dict[str, Any], output_path: Path) -> Path:
        """خروجی کارت جامع استعدادیابی جهت ذخیره و صدور گزارشات سازمانی"""
        card = {
            "platform": "KinetiQ OS",
            "athlete": athlete_info,
            "test_summary": metric_summary.model_dump(),
        }
        output_path.write_text(json.dumps(card, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return output_path

    @staticmethod
    def export_pdf_report(
        metric_summary: MetricSummary,
        athlete_info: Dict[str, Any],
        output_path: Path,
        session_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """صدور گزارش استعدادیابی رسمی و شکیل PDF به همراه نمودار راداری عملکرد"""
        from core.export.pdf_generator import ScoutPDFReportGenerator
        return ScoutPDFReportGenerator.create_report(
            athlete_info=athlete_info,
            metric_summary=metric_summary,
            output_path=output_path,
            session_info=session_info,
        )
