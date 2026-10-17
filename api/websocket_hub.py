"""
KinetiQ OS - Real-Time High-Speed WebSocket Hub
هاب ارتباطی بلادرنگ جهت انتقال داده‌های زنده مفاصل و تله‌متری به کلاینت‌ها با تاخیر کمتر از ۱۰ میلی‌ثانیه
"""

import json
from typing import Dict, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
from core.logging import logger


class WebSocketHub:
    """مدیریت متمرکز اتصالات زنده وب‌سوکت برای استریم داده‌ها"""

    def __init__(self):
        # نگهداری اتصالات فعال به تفکیک session_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """پذیرش و ثبت اتصال جدید"""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)
        logger.debug(f"کلاینت جدید به نشست وب‌سوکت '{session_id}' متصل شد.")

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        """قطع و پاک‌سازی اتصال کلاینت"""
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.debug(f"اتصال کلاینت از نشست وب‌سوکت '{session_id}' قطع شد.")

    async def broadcast_telemetry(self, session_id: str, data: Dict[str, Any]) -> None:
        """ارسال آنی پیام تله‌متری (فریم مفاصل، خطاها یا رکوردها) به تمامی کلاینت‌های متصل به نشست"""
        if session_id in self.active_connections:
            message = json.dumps(data)
            dead_connections = set()
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.warning(f"خطا در ارسال داده روی وب‌سوکت: {e}")
                    dead_connections.add(connection)

            # پاک‌سازی اتصالات قطع‌شده
            for dead in dead_connections:
                self.disconnect(session_id, dead)


# نمونه سراسری هاب وب‌سوکت
ws_hub = WebSocketHub()
