"""
KinetiQ OS - Core Plugin Registry Engine
معماری ماژولار مبتنی بر رجیستری برای اضافه یا حذف آسان تست‌های ورزشی
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Optional, Any
from core.schemas import SkeletonFrame, MetricSummary
from core.logging import logger


class BaseSportTest(ABC):
    """
    کلاس پایه انتزاعی برای هر آزمون ورزشی یا ماژول تحلیلی.
    هر تست جدید (مثلاً پرش، قد، ۳ گام هندبال) باید از این کلاس ارث‌بری کند.
    """

    test_id: str = "base_test"
    title: str = "Base Sport Test"
    category: str = "general"  # 'anthropometry', 'jumps', 'handball', 'agility'
    description: str = "توضیحات تست پایه"
    version: str = "1.0.0"

    def __init__(self, athlete_id: int, config: Optional[Dict[str, Any]] = None):
        self.athlete_id = athlete_id
        self.config = config or {}
        self.frames_history: List[SkeletonFrame] = []
        self.is_active: bool = False
        self.state: str = "IDLE"

    @abstractmethod
    def start(self) -> None:
        """آماده‌سازی و شروع مانیتورینگ آزمون"""
        pass

    @abstractmethod
    def process_frame(self, frame: SkeletonFrame) -> Dict[str, Any]:
        """
        پردازش یک فریم ویدیویی ورودی و استخراج محاسبات آنی
        خروجی یک دیکشنری از متغیرهای لحظه‌ای (Live Telemetry) برای نمایش در HUD است.
        """
        pass

    @abstractmethod
    def finish(self) -> MetricSummary:
        """اتمام تست و محاسبه نهایی شاخص‌ها و رکوردها"""
        pass

    def reset(self) -> None:
        """ریست وضعیت جهت شروع مجدد"""
        self.frames_history.clear()
        self.is_active = False
        self.state = "IDLE"


class TestRegistry:
    """
    مرکز رجیستری برای ثبت، کشف و مدیریت تمامی پلاگین‌های آزمون ورزشی
    """
    __test__ = False

    def __init__(self):
        self._registry: Dict[str, Type[BaseSportTest]] = {}

    def register(self, test_cls: Type[BaseSportTest]) -> Type[BaseSportTest]:
        """دکوراتور یا متد ثبت یک پلاگین تست ورزشی جدید"""
        test_id = test_cls.test_id
        if test_id in self._registry:
            logger.warning(f"پلاگین با شناسه '{test_id}' قبلاً ثبت شده بود و بازنویسی شد.")
        self._registry[test_id] = test_cls
        logger.info(f"پلاگین آزمون ورزشی '{test_cls.title}' با شناسه '{test_id}' با موفقیت لود شد.")
        return test_cls

    def unregister(self, test_id: str) -> bool:
        """حذف یک پلاگین از سامانه"""
        if test_id in self._registry:
            del self._registry[test_id]
            logger.info(f"پلاگین با شناسه '{test_id}' با موفقیت حذف شد.")
            return True
        return False

    def get_test_class(self, test_id: str) -> Optional[Type[BaseSportTest]]:
        """بازیابی کلاس یک تست بر اساس شناسه"""
        return self._registry.get(test_id)

    def list_available_tests(self) -> List[Dict[str, Any]]:
        """فهرست کامل تمام آزمون‌های فعال در سیستم همراه با متادیتا"""
        result = []
        for test_id, cls in self._registry.items():
            result.append({
                "test_id": test_id,
                "title": cls.title,
                "category": cls.category,
                "description": cls.description,
                "version": cls.version,
            })
        return result

    def create_instance(self, test_id: str, athlete_id: int, config: Optional[Dict[str, Any]] = None) -> BaseSportTest:
        """نمونه‌سازی از یک تست بر اساس شناسه"""
        cls = self.get_test_class(test_id)
        if not cls:
            raise ValueError(f"هیچ آزمون ورزشی با شناسه '{test_id}' در رجیستری یافت نشد.")
        return cls(athlete_id=athlete_id, config=config)


# نمونه سراسری رجیستری
plugin_registry = TestRegistry()
