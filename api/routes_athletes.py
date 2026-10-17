"""
KinetiQ OS - Athlete Management Routes
مسیرهای مدیریت پرونده‌های ورزشی، استعدادیابی و تاریخچه آزمون‌ها
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session, select
from pydantic import BaseModel
from db.session import get_db
from db.models import Athlete, TestSession
from api.deps import get_current_user
from db.models import User

router = APIRouter(prefix="/api/athletes", tags=["Athletes"])


class AthleteCreate(BaseModel):
    first_name: str
    last_name: str
    national_id: Optional[str] = None
    gender: str = "male"
    birth_date: Optional[str] = None
    sport: str = "handball"
    position: Optional[str] = None
    standing_height_cm: Optional[float] = None
    arm_span_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    notes: Optional[str] = None


class AthleteUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    gender: Optional[str] = None
    birth_date: Optional[str] = None
    sport: Optional[str] = None
    position: Optional[str] = None
    standing_height_cm: Optional[float] = None
    arm_span_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    notes: Optional[str] = None


@router.get("", response_model=List[Athlete])
def list_athletes(
    search: Optional[str] = Query(None, description="جستجو در نام، کد ملی یا رشته"),
    sport: Optional[str] = Query(None, description="فیلتر بر اساس رشته ورزشی"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """دریافت فهرست ورزشکاران با امکان جستجو و فیلتر"""
    query = select(Athlete)
    if sport:
        query = query.where(Athlete.sport == sport)
    if search:
        search_filter = f"%{search}%"
        query = query.where(
            (Athlete.first_name.like(search_filter))
            | (Athlete.last_name.like(search_filter))
            | (Athlete.national_id.like(search_filter))
        )
    return db.exec(query.order_by(Athlete.last_name)).all()


@router.post("", response_model=Athlete, status_code=status.HTTP_201_CREATED)
def create_athlete(
    data: AthleteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """ثبت‌نام و ایجاد پرونده استعدادیابی ورزشکار جدید"""
    if data.national_id:
        existing = db.exec(select(Athlete).where(Athlete.national_id == data.national_id)).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ورزشکاری با این کد ملی قبلاً در سامانه ثبت شده است.",
            )

    athlete = Athlete(**data.model_dump())
    db.add(athlete)
    db.commit()
    db.refresh(athlete)
    return athlete


@router.get("/{athlete_id}", response_model=Athlete)
def get_athlete(
    athlete_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """دریافت جزییات پرونده یک ورزشکار"""
    athlete = db.get(Athlete, athlete_id)
    if not athlete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ورزشکار مورد نظر یافت نشد.",
        )
    return athlete


@router.put("/{athlete_id}", response_model=Athlete)
def update_athlete(
    athlete_id: int,
    data: AthleteUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """ویرایش اطلاعات بیومتریک و شناسنامه ورزشکار"""
    athlete = db.get(Athlete, athlete_id)
    if not athlete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ورزشکار مورد نظر یافت نشد.",
        )

    for field, value in data.dict(exclude_unset=True).items():
        setattr(athlete, field, value)

    db.add(athlete)
    db.commit()
    db.refresh(athlete)
    return athlete


@router.delete("/{athlete_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_athlete(
    athlete_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """حذف پرونده ورزشکار"""
    athlete = db.get(Athlete, athlete_id)
    if not athlete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ورزشکار مورد نظر یافت نشد.",
        )
    db.delete(athlete)
    db.commit()
    return None
