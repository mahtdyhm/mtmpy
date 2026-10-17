"""
KinetiQ OS - Core Schemas
طرح‌واره‌های استاندارد مفاصل، تله‌متری و نتایج آزمون‌های ورزشی
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class LandmarkName(str, Enum):
    """نام‌های استاندارد ۳۳ مفصل بدن (مشابه MediaPipe/COCO Extended)"""
    NOSE = "nose"
    LEFT_EYE_INNER = "left_eye_inner"
    LEFT_EYE = "left_eye"
    LEFT_EYE_OUTER = "left_eye_outer"
    RIGHT_EYE_INNER = "right_eye_inner"
    RIGHT_EYE = "right_eye"
    RIGHT_EYE_OUTER = "right_eye_outer"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    MOUTH_LEFT = "mouth_left"
    MOUTH_RIGHT = "mouth_right"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_PINKY = "left_pinky"
    RIGHT_PINKY = "right_pinky"
    LEFT_INDEX = "left_index"
    RIGHT_INDEX = "right_index"
    LEFT_THUMB = "left_thumb"
    RIGHT_THUMB = "right_thumb"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_HEEL = "left_heel"
    RIGHT_HEEL = "right_heel"
    LEFT_FOOT_INDEX = "left_foot_index"
    RIGHT_FOOT_INDEX = "right_foot_index"


class Landmark3D(BaseModel):
    """مختصات سه‌بعدی یک مفصل همراه با ضریب اطمینان"""
    name: str
    x: float = Field(..., description="مختصات افقی نرمالایز شده [0, 1]")
    y: float = Field(..., description="مختصات عمودی نرمالایز شده [0, 1]")
    z: float = Field(0.0, description="عمق تخمینی یا متریک")
    visibility: float = Field(1.0, description="ضریب اطمینان تشخیص [0, 1]")
    presence: float = Field(1.0, description="احتمال حضور در کادر [0, 1]")


class SkeletonFrame(BaseModel):
    """اسکلت کامل استخراج‌شده در یک فریم مشخص"""
    frame_index: int
    timestamp_ms: float
    landmarks: Dict[str, Landmark3D]
    raw_fps: float = 30.0
    detected_person: bool = True
    ai_mode: str = "offline"  # 'offline' یا 'online'


class MetricSummary(BaseModel):
    """خلاصه شاخص‌های خروجی یک تست ورزشی"""
    test_id: str
    athlete_id: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    primary_metric_name: str
    primary_metric_value: float
    primary_metric_unit: str
    detailed_metrics: Dict[str, Any] = Field(default_factory=dict)
    feedback_notes: List[str] = Field(default_factory=list)
    passed: bool = True
