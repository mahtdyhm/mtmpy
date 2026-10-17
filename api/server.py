"""
KinetiQ OS - Main FastAPI Application Server
پیکربندی سرور اصلی، میدل‌ویرها، روت‌ها و چرخه حیات سامانه
"""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from config.settings import settings
from db.session import init_db
from core.logging import logger
from core.registry import plugin_registry
from api.routes_auth import router as auth_router
from api.routes_athletes import router as athletes_router
from api.routes_tests import router as tests_router

# ایمپورت و بارگذاری خودکار پلاگین‌های اصلی
from plugins.anthropometry.height_span import HeightSpanTest
from plugins.jumps.vertical_jump import VerticalJumpTest
from plugins.handball.three_step import HandballThreeStepTest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """مدیریت راه‌اندازی و خاموش‌سازی سرور"""
    logger.info("در حال آماده‌سازی سامانه KinetiQ OS...")
    init_db()
    logger.info(f"سامانه KinetiQ با {len(plugin_registry.list_available_tests())} پلاگین آزمون آماده به کار است.")
    yield
    logger.info("سامانه KinetiQ با موفقیت متوقف شد.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="پلتفرم جامع هوشمند بینایی ماشین ورزشی، استعدادیابی و آنالیز هندبال",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"خطای مدیریت‌نشده در مسیر {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "خطای داخلی سرور رخ داد. لطفا لاگ‌ها را بررسی کنید."},
    )


# اتصال روت‌های API
app.include_router(auth_router)
app.include_router(athletes_router)
app.include_router(tests_router)

# سرو فایل‌های استاتیک
static_dir = settings.BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", tags=["UI"])
def serve_ui():
    """ارائه داشبورد وب زنده KinetiQ OS"""
    index_file = settings.BASE_DIR / "static" / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {"message": "KinetiQ OS Backend Running"}


@app.get("/health", tags=["Health"])
def health_check():
    """بررسی سلامت سرور و وضعیت حالت هوش مصنوعی"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "ai_mode": settings.DEFAULT_AI_MODE,
        "active_plugins": len(plugin_registry.list_available_tests()),
    }
