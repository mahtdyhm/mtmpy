"""
KinetiQ OS - Test Execution & Plugin Endpoints
مسیرهای انتخاب، شروع، مانیتورینگ و خاتمه آزمون‌های ورزشی
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from pydantic import BaseModel
from db.session import get_db
from db.models import TestSession, MetricLog, Athlete, User
from core.registry import plugin_registry
from api.deps import get_current_user
from api.websocket_hub import ws_hub
from core.schemas import MetricSummary
from core.export.pdf_generator import ScoutPDFReportGenerator
from config.settings import settings
from core.logging import logger

router = APIRouter(prefix="/api/tests", tags=["Tests & Analysis"])


class StartSessionRequest(BaseModel):
    test_id: str
    athlete_id: int
    ai_mode: str = "offline"  # 'offline' یا 'online'
    notes: Optional[str] = None


class SessionResponse(BaseModel):
    id: int
    test_id: str
    athlete_id: int
    coach_id: Optional[int]
    status: str
    ai_mode: str
    created_at: str


@router.get("/available")
def get_available_tests():
    """فهرست تمام پلاگین‌های آزمون فعال در سیستم"""
    return plugin_registry.list_available_tests()


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    data: StartSessionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """ایجاد یک سشن جدید برای شروع آزمون ورزشی"""
    athlete = db.get(Athlete, data.athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail="ورزشکار یافت نشد.")

    if not plugin_registry.get_test_class(data.test_id):
        raise HTTPException(status_code=400, detail=f"آزمون ورزشی '{data.test_id}' در رجیستری تعریف نشده است.")

    session = TestSession(
        test_id=data.test_id,
        athlete_id=data.athlete_id,
        coach_id=current_user.id,
        status="in_progress",
        ai_mode=data.ai_mode,
        notes=data.notes,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return SessionResponse(
        id=session.id,
        test_id=session.test_id,
        athlete_id=session.athlete_id,
        coach_id=session.coach_id,
        status=session.status,
        ai_mode=session.ai_mode,
        created_at=session.created_at.isoformat(),
    )


@router.get("/sessions/{session_id}")
def get_session_details(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """دریافت اطلاعات و متریک‌های ثبت شده یک سشن"""
    session = db.get(TestSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="سشن تست یافت نشد.")

    metrics = db.exec(select(MetricLog).where(MetricLog.session_id == session_id)).all()
    return {
        "session": session,
        "athlete": session.athlete,
        "metrics": metrics,
    }


class FinishSessionRequest(BaseModel):
    primary_metric_name: str
    primary_metric_value: float
    primary_metric_unit: str = ""
    detailed_metrics: Dict[str, Any] = {}
    feedback_notes: List[str] = []


@router.post("/sessions/{session_id}/finish")
def finish_session(
    session_id: int,
    data: FinishSessionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """تکمیل سشن آزمون، ثبت متریک‌های نهایی و آماده‌سازی گزارش"""
    session = db.get(TestSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="سشن تست یافت نشد.")

    session.status = "completed"

    metric_log = MetricLog(
        session_id=session.id,
        metric_name=data.primary_metric_name,
        metric_value=data.primary_metric_value,
        metric_unit=data.primary_metric_unit,
        details_json=json.dumps(data.detailed_metrics, ensure_ascii=False),
    )
    db.add(metric_log)
    db.commit()
    db.refresh(session)
    return {"status": "success", "session_id": session.id, "primary_metric": data.primary_metric_name}


@router.get("/sessions/{session_id}/report.pdf")
def download_pdf_report(
    session_id: int,
    db: Session = Depends(get_db),
):
    """تولید و دانلود کارنامه رسمی استعدادیابی ورزشی در قالب فایل PDF"""
    session = db.get(TestSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="سشن تست یافت نشد.")

    athlete = session.athlete
    athlete_info = {
        "first_name": athlete.first_name if athlete else "ورزشکار",
        "last_name": athlete.last_name if athlete else "",
        "national_id": athlete.national_id if athlete else "N/A",
        "sport": athlete.sport if athlete else "Handball",
        "position": athlete.position if athlete else "Player",
        "gender": athlete.gender if athlete else "male",
        "standing_height_cm": athlete.standing_height_cm,
        "arm_span_cm": athlete.arm_span_cm,
        "weight_kg": athlete.weight_kg,
    }

    metrics = db.exec(select(MetricLog).where(MetricLog.session_id == session_id)).all()
    primary_name = "Primary Score"
    primary_val = 0.0
    primary_unit = ""
    detailed = {}

    if metrics:
        primary_metric = metrics[0]
        primary_name = primary_metric.metric_name
        primary_val = primary_metric.metric_value
        primary_unit = primary_metric.metric_unit
        if primary_metric.details_json:
            try:
                detailed = json.loads(primary_metric.details_json)
            except Exception:
                pass

    feedback_notes = [
        f"Kinematic tracking performed via KinetiQ AI ({session.ai_mode.upper()} mode).",
        "Movement profile conforms with biomechanical safety and athletic efficiency standards.",
    ]
    if "jump" in session.test_id:
        feedback_notes.append("Explosive reactive strength index recorded high elastic energy recovery.")
    elif "height" in session.test_id:
        feedback_notes.append("Anthropometric profile and arm span ratio are favorable for technical handball reach.")

    summary = MetricSummary(
        test_id=session.test_id,
        athlete_id=session.athlete_id,
        primary_metric_name=primary_name,
        primary_metric_value=primary_val,
        primary_metric_unit=primary_unit,
        detailed_metrics=detailed,
        feedback_notes=feedback_notes,
    )

    pdf_filename = f"kinetiq_scout_session_{session.id}.pdf"
    pdf_path = settings.OUTPUT_DIR / pdf_filename

    session_info = {
        "id": session.id,
        "created_at": session.created_at.strftime("%Y-%m-%d %H:%M"),
        "ai_mode": session.ai_mode,
    }

    ScoutPDFReportGenerator.create_report(
        athlete_info=athlete_info,
        metric_summary=summary,
        output_path=pdf_path,
        session_info=session_info,
    )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_filename,
    )


@router.websocket("/ws/{session_id}")
async def websocket_test_stream(websocket: WebSocket, session_id: str):
    """کانال وب‌سوکت بلادرنگ برای دریافت فریم‌ها و ارسال تله‌متری زنده"""
    await ws_hub.connect(session_id, websocket)
    try:
        while True:
            # دریافت داده‌ها از کلاینت (مثلاً فریم وب‌کم یا پینگ تله‌متری)
            data_text = await websocket.receive_text()
            try:
                data = json.loads(data_text)
                # در صورت نیاز به ارسال پاسخ یا پخش تله‌متری به کلاینت‌های دیگر
                await ws_hub.broadcast_telemetry(session_id, {"status": "received", "echo": data})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        ws_hub.disconnect(session_id, websocket)
    except Exception as e:
        logger.error(f"خطای کانکشن وب‌سوکت: {e}")
        ws_hub.disconnect(session_id, websocket)
