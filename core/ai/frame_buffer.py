"""
KinetiQ OS - Frame Ring Buffer
صف چرخشی با ظرفیت ثابت جهت جلوگیری از اشغال حافظه RAM در استریم‌های طولانی
"""

from collections import deque
from typing import Optional, List, Any
import numpy as np


class FrameRingBuffer:
    """بافر حلقوی با اندازه ثابت جهت نگهداری بهینه فریم‌های ویدیویی اخیر"""

    def __init__(self, max_capacity: int = 300):
        self.max_capacity = max_capacity
        self.buffer = deque(maxlen=max_capacity)

    def append(self, frame: np.ndarray, metadata: Optional[dict] = None) -> None:
        """افزودن فریم به انتهای بافر"""
        self.buffer.append({"frame": frame, "metadata": metadata or {}})

    def get_latest(self) -> Optional[dict]:
        """دریافت جدیدترین فریم ثبت‌شده"""
        return self.buffer[-1] if self.buffer else None

    def get_all(self) -> List[dict]:
        """دریافت تمام فریم‌های موجود در بافر به ترتیب زمانی"""
        return list(self.buffer)

    def size(self) -> int:
        """تعداد فریم‌های حاضر در بافر"""
        return len(self.buffer)

    def clear(self) -> None:
        """تخلیه کامل بافر و آزادسازی حافظه"""
        self.buffer.clear()
