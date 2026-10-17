"""
KinetiQ OS - Test Execution & Plugin Endpoints
مسیرهای انتخاب، شروع، مانیتورینگ و خاتمه آزمون‌های ورزشی
"""

import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select
from pydantic import BaseModel
from db.session import get_db
from db.models import TestSession, MetricLog, Athlete, User
from core.registry import plugin_registry
from api.deps import get_current_user
from api.websocket_hub import ws_hub
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
