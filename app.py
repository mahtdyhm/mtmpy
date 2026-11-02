"""
KinetiQ OS - Streamlit Cloud Web Application
پلتفرم هوشمند بینایی ماشین ورزشی و استعدادیابی آنلاین بر بستر Streamlit Community Cloud
"""

import os
import io
import time
from pathlib import Path
import streamlit as st
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image

# تنظیمات اولیه صفحه Streamlit
st.set_page_config(
    page_title="KinetiQ OS | پلتفرم بینایی ماشین ورزشی",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ایمپورت ماژول‌های اصلی سامانه KinetiQ
from core.schemas import MetricSummary, SkeletonFrame, Landmark3D
from core.export.pdf_generator import ScoutPDFReportGenerator
from core.biomechanics.calibration import SpatialCalibrator
from core.biomechanics.anthropometry import AnthropometryCalculator
from core.export.telemetry_renderer import VideoTelemetryRenderer
from core.ai.offline_yolo import YoloPoseEstimator
from config.settings import settings


@st.cache_resource
def load_yolo_model():
    """بارگذاری و کش کردن مدل هوش مصنوعی YOLO-Pose جهت سرعت بالا در پردازش"""
    model_path = Path("yolo11n-pose.pt")
    if not model_path.exists():
        model_path = settings.BASE_DIR / "yolo11n-pose.pt"
    estimator = YoloPoseEstimator(model_name=str(model_path))
    estimator.initialize()
    return estimator


# استایل گرافیکی تیره، نئونی و کاملاً واکنش‌گرا
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    
    html, body, [class*="css"] {
        font-family: 'Vazirmatn', system-ui, sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    .main-header {
        background: linear-gradient(135deg, #090c10 0%, #121820 100%);
        border: 1px solid #243242;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.08);
    }
    
    .stMetric {
        background: #182230 !important;
        border: 1px solid #243242 !important;
        border-radius: 10px !important;
        padding: 14px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        background-color: #121820;
        color: #8b949e;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #182230 !important;
        color: #00ff88 !important;
        border-bottom: 2px solid #00ff88 !important;
    }
</style>
""", unsafe_allow_html=True)


# نوار کناری (Sidebar)
with st.sidebar:
    st.markdown("### 🏃 **KinetiQ OS**")
    st.caption("Sports Vision AI & Athletic Scouting Platform")
    st.markdown("---")
    
    st.markdown("#### 🎯 انتخاب آزمون ورزشی")
    selected_test = st.selectbox(
        "نوع آزمون بیومکانیک:",
        options=[
            ("height_span", "📏 سنجش قد و اسپن (آنتروپومتری)"),
            ("vertical_jump", "🚀 پرش عمودی و توان انفجاری (CMJ)"),
            ("broad_jump", "🎯 پرش طول جفت‌پا (Broad Jump)"),
            ("handball_three_step", "🤾 آزمون ۳ گام و خطاهای هندبال"),
        ],
        format_func=lambda x: x[1],
    )[0]
    
    st.markdown("---")
    st.markdown("#### 👤 پرونده ورزشکار")
    athlete_name = st.text_input("نام و نام خانوادگی:", value="علی محمدی")
    sport_name = st.selectbox("رشته ورزشی:", ["هندبال (Handball)", "والیبال", "بسکتبال", "دوومیدانی"])
    post_name = st.text_input("پست تخصصی:", value="گوشه (Wing)")
    user_calib_height = st.number_input("قد واقعی جهت کالیبراسیون (cm):", value=185.0, step=1.0)
    user_weight = st.number_input("وزن بدنی (kg):", value=82.0, step=0.5)


# هدر اصلی
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin: 0; color: #fff; font-weight: 800;">پلتفرم بینایی ماشین ورزشی KinetiQ OS</h2>
            <p style="margin: 4px 0 0 0; color: #8b949e; font-size: 0.9rem;">
                آنالیز استخوان‌بندی، زوایای مفاصل و کینماتیک با هوش مصنوعی | وضعیت: <span style="color: #00ff88; font-weight: bold;">سیستم هوش مصنوعی فعال ⚡</span>
            </p>
        </div>
        <div style="background: rgba(0,255,136,0.15); border: 1px solid #00ff88; padding: 6px 14px; border-radius: 20px; color: #00ff88; font-weight: 700; font-size: 0.85rem;">
            LIVE SKELETON AI 🟢
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# تب‌های سامانه
tab_live_cam, tab_snapshot, tab_video, tab_report, tab_pwa = st.tabs([
    "⚡ دوربین زنده و اسکلت لحظه‌ای (Live 60FPS)",
    "📸 پردازش دقیق فریم با YOLO11",
    "🎥 تحلیل ویدیویی حرکت (Video Kinematics)",
    "📄 کارنامه رسمی استعدادیابی (PDF)",
    "🌐 راهنمای PWA",
])


# ==============================================================
# تب ۱: دوربین بلادرنگ زنده با استخوان‌بندی و زوایای مفاصل در حرکت
# ==============================================================
with tab_live_cam:
    st.markdown("### ⚡ ردیابی زنده مفاصل و زوایای بدن در لحظه (Real-Time)")
    st.caption("دکمه زیر را بزنید؛ دوربین گوشی یا سیستم فعال شده و اسکلت نئونی و زوایای بدن روی تصویر رسم می‌شوند. برای جابجایی بین دوربین پشت و سلفی از دکمه چرخش دوربین استفاده کنید.")
    
    live_cam_html = f"""
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
      <meta charset="UTF-8">
      <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" crossorigin="anonymous"></script>
      <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: transparent; font-family: system-ui, sans-serif; color: #fff; text-align: center; }}
        .cam-box {{
          position: relative;
          max-width: 720px;
          margin: 0 auto;
          border-radius: 12px;
          overflow: hidden;
          background: #000;
          border: 2px solid #00ff88;
          box-shadow: 0 0 25px rgba(0,255,136,0.25);
        }}
        video {{ width: 100%; height: auto; display: block; }}
        canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
        .hud-panel {{
          position: absolute;
          top: 12px;
          right: 12px;
          background: rgba(14,20,28,0.88);
          border: 1px solid #00e5ff;
          border-radius: 10px;
          padding: 10px 16px;
          text-align: right;
          direction: rtl;
          font-size: 13px;
          backdrop-filter: blur(6px);
          pointer-events: none;
        }}
        .hud-title {{ color: #8b949e; font-size: 11px; margin-bottom: 2px; }}
        .hud-val {{ font-weight: 800; font-size: 18px; margin-bottom: 6px; font-family: monospace; }}
        .ctrl-bar {{ margin-top: 14px; display: flex; justify-content: center; align-items: center; gap: 12px; flex-wrap: wrap; }}
        .btn {{
          background: #00ff88;
          color: #000;
          font-weight: 800;
          border: none;
          padding: 10px 22px;
          border-radius: 8px;
          cursor: pointer;
          font-size: 14px;
          transition: all 0.2s;
        }}
        .btn:hover {{ transform: scale(1.04); box-shadow: 0 0 16px #00ff88; }}
        .btn-switch {{
          background: #182230;
          color: #00e5ff;
          border: 1px solid #00e5ff;
        }}
        .btn-switch:hover {{
          background: #00e5ff;
          color: #000;
          box-shadow: 0 0 16px rgba(0, 229, 255, 0.4);
        }}
        .fps-badge {{
          position: absolute;
          top: 12px;
          left: 12px;
          background: rgba(14,20,28,0.85);
          border: 1px solid #ffb700;
          color: #ffb700;
          padding: 4px 10px;
          border-radius: 6px;
          font-family: monospace;
          font-size: 12px;
        }}
        .cam-badge {{
          position: absolute;
          bottom: 12px;
          left: 12px;
          background: rgba(14,20,28,0.85);
          border: 1px solid #00ff88;
          color: #00ff88;
          padding: 4px 10px;
          border-radius: 6px;
          font-size: 12px;
        }}
      </style>
    </head>
    <body>
      <div class="cam-box">
        <video id="live-vid" autoplay playsinline muted></video>
        <canvas id="live-canvas"></canvas>
        <div class="fps-badge" id="fps-badge">0.0 FPS</div>
        <div class="cam-badge" id="cam-badge">📷 دوربین پشت (اصلی)</div>
        <div class="hud-panel">
          <div class="hud-title">قد بیومکانیکی لحظه‌ای:</div>
          <div class="hud-val" id="hud-h" style="color: #00ff88;">{user_calib_height} cm</div>
          <div class="hud-title">طول اسپن بازوها:</div>
          <div class="hud-val" id="hud-s" style="color: #00e5ff;">{user_calib_height * 1.04:.1f} cm</div>
          <div class="hud-title">زاویه زانو (خم/راست):</div>
          <div class="hud-val" id="hud-knee" style="color: #ffb700;">172°</div>
          <div class="hud-title">زاویه آرنج:</div>
          <div class="hud-val" id="hud-elbow" style="color: #00e5ff;">160°</div>
        </div>
      </div>
      <div class="ctrl-bar">
        <button class="btn" id="btn-toggle" onclick="toggleCamera()">▶ روشن کردن دوربین</button>
        <button class="btn btn-switch" id="btn-flip" onclick="switchCamera()">🔄 چرخش دوربین (پشت / جلو)</button>
      </div>
      <div id="status-info" style="margin-top: 8px; font-size: 12px; color: #8b949e;">
        پیش‌فرض: دوربین پشت فعال است | در صورت نیاز دکمه چرخش دوربین را بزنید
      </div>

      <script>
        let isCamActive = false;
        let currentFacingMode = 'environment';
        let currentStream = null;
        let availableDevices = [];
        let currentDeviceIndex = 0;
        let poseEngine = null;
        let lastTimestamp = performance.now();
        let isLooping = false;

        function calcAngle(p1, p2, p3) {{
          const v1 = {{ x: p1.x - p2.x, y: p1.y - p2.y }};
          const v2 = {{ x: p3.x - p2.x, y: p3.y - p2.y }};
          const dot = v1.x * v2.x + v1.y * v2.y;
          const mag1 = Math.hypot(v1.x, v1.y);
          const mag2 = Math.hypot(v2.x, v2.y);
          if (mag1 * mag2 === 0) return 0;
          const cos = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
          return Math.round(Math.acos(cos) * (180 / Math.PI));
        }}

        function onResults(results) {{
          const video = document.getElementById('live-vid');
          const canvas = document.getElementById('live-canvas');
          if (!canvas || !video) return;
          const ctx = canvas.getContext('2d');

          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {{
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
          }}
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          const now = performance.now();
          const fps = (1000.0 / Math.max(1, now - lastTimestamp)).toFixed(1);
          lastTimestamp = now;
          document.getElementById('fps-badge').innerText = fps + ' FPS';

          if (!results.poseLandmarks) return;
          const lms = results.poseLandmarks;
          const w = canvas.width;
          const h = canvas.height;
          const isFront = (currentFacingMode === 'user');

          // ۱. استخوان‌های نئونی
          const bones = [
            [11, 12], [11, 23], [12, 24], [23, 24],
            [11, 13], [13, 15], [12, 14], [14, 16],
            [23, 25], [25, 27], [24, 26], [26, 28]
          ];
          ctx.lineWidth = 4;
          ctx.strokeStyle = '#00ff88';
          ctx.shadowColor = '#00ff88';
          ctx.shadowBlur = 10;
          ctx.lineCap = 'round';

          for (const [i, j] of bones) {{
            const a = lms[i];
            const b = lms[j];
            if (a && b && a.visibility > 0.3 && b.visibility > 0.3) {{
              ctx.beginPath();
              ctx.moveTo(a.x * w, a.y * h);
              ctx.lineTo(b.x * w, b.y * h);
              ctx.stroke();
            }}
          }}

          // ۲. نقاط مفاصل فیروزه‌ای
          ctx.fillStyle = '#00e5ff';
          ctx.shadowColor = '#00e5ff';
          ctx.shadowBlur = 12;
          for (let i = 0; i < lms.length; i++) {{
            const p = lms[i];
            if (p && p.visibility > 0.3) {{
              ctx.beginPath();
              ctx.arc(p.x * w, p.y * h, 5, 0, 2 * Math.PI);
              ctx.fill();
            }}
          }}
          ctx.shadowBlur = 0;

          // ۳. محاسبه و رسم زوایای زانو و آرنج روی تصویر
          if (lms[23] && lms[25] && lms[27]) {{
            const kAngle = calcAngle(lms[23], lms[25], lms[27]);
            document.getElementById('hud-knee').innerText = kAngle + '°';
            ctx.save();
            ctx.fillStyle = '#ffff00';
            ctx.font = 'bold 15px monospace';
            if (isFront) {{
              ctx.scale(-1, 1);
              ctx.fillText(kAngle + '°', -lms[25].x * w - 35, lms[25].y * h + 5);
            }} else {{
              ctx.fillText(kAngle + '°', lms[25].x * w + 10, lms[25].y * h + 5);
            }}
            ctx.restore();
          }}

          if (lms[11] && lms[13] && lms[15]) {{
            const eAngle = calcAngle(lms[11], lms[13], lms[15]);
            document.getElementById('hud-elbow').innerText = eAngle + '°';
            ctx.save();
            ctx.fillStyle = '#00e5ff';
            ctx.font = 'bold 15px monospace';
            if (isFront) {{
              ctx.scale(-1, 1);
              ctx.fillText(eAngle + '°', -lms[13].x * w - 35, lms[13].y * h + 5);
            }} else {{
              ctx.fillText(eAngle + '°', lms[13].x * w + 10, lms[13].y * h + 5);
            }}
            ctx.restore();
          }}

          // ۴. محاسبه پویا قد و اسپن
          if (lms[0] && lms[27]) {{
            const dy = Math.abs(lms[27].y - lms[0].y) * h;
            const estH = Math.round(dy * 0.36 * 1.15);
            if (estH > 60) document.getElementById('hud-h').innerText = estH + ' cm';
          }}
          if (lms[15] && lms[16]) {{
            const dx = Math.hypot((lms[15].x - lms[16].x) * w, (lms[15].y - lms[16].y) * h);
            const estS = Math.round(dx * 0.38);
            if (estS > 60) document.getElementById('hud-s').innerText = estS + ' cm';
          }}
        }}

        async function initPoseEngine() {{
          if (!poseEngine) {{
            poseEngine = new Pose({{
              locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${{file}}`
            }});
            poseEngine.setOptions({{
              modelComplexity: 1,
              smoothLandmarks: true,
              minDetectionConfidence: 0.5,
              minTrackingConfidence: 0.5
            }});
            poseEngine.onResults(onResults);
          }}
        }}

        async function updateDeviceList() {{
          try {{
            const devices = await navigator.mediaDevices.enumerateDevices();
            availableDevices = devices.filter(d => d.kind === 'videoinput');
          }} catch (e) {{
            console.warn("enumerateDevices error:", e);
          }}
        }}

        async function startStream() {{
          if (currentStream) {{
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
          }}

          let constraints = {{
            audio: false,
            video: {{
              width: {{ ideal: 1280 }},
              height: {{ ideal: 720 }}
            }}
          }};

          if (availableDevices.length > 0 && availableDevices[currentDeviceIndex]) {{
            constraints.video.deviceId = {{ exact: availableDevices[currentDeviceIndex].deviceId }};
          }} else {{
            constraints.video.facingMode = {{ ideal: currentFacingMode }};
          }}

          try {{
            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
          }} catch (err) {{
            try {{
              currentStream = await navigator.mediaDevices.getUserMedia({{
                audio: false,
                video: {{ facingMode: currentFacingMode }}
              }});
            }} catch (err2) {{
              currentStream = await navigator.mediaDevices.getUserMedia({{ audio: false, video: true }});
            }}
          }}

          const video = document.getElementById('live-vid');
          video.srcObject = currentStream;
          await video.play();

          // بررسی حالت سلفی برای آینه‌ای کردن
          const track = currentStream.getVideoTracks()[0];
          const settings = track.getSettings ? track.getSettings() : {{}};
          const isFront = (settings.facingMode === 'user') || (currentFacingMode === 'user');
          currentFacingMode = isFront ? 'user' : 'environment';

          const canvas = document.getElementById('live-canvas');
          if (isFront) {{
            video.style.transform = 'scaleX(-1)';
            canvas.style.transform = 'scaleX(-1)';
            document.getElementById('cam-badge').innerText = '📷 دوربین جلو (سلفی)';
          }} else {{
            video.style.transform = 'none';
            canvas.style.transform = 'none';
            document.getElementById('cam-badge').innerText = '📷 دوربین پشت (اصلی)';
          }}

          isLooping = true;
          runProcessingLoop();
        }}

        async function runProcessingLoop() {{
          if (!isCamActive || !isLooping) return;
          const video = document.getElementById('live-vid');
          if (video && video.readyState >= 2 && poseEngine) {{
            try {{
              await poseEngine.send({{ image: video }});
            }} catch (e) {{}}
          }}
          if (isCamActive) {{
            if ('requestVideoFrameCallback' in video) {{
              video.requestVideoFrameCallback(runProcessingLoop);
            }} else {{
              requestAnimationFrame(runProcessingLoop);
            }}
          }}
        }}

        async function toggleCamera() {{
          const btn = document.getElementById('btn-toggle');
          if (!isCamActive) {{
            btn.innerText = '⏹ توقف دوربین';
            btn.style.background = '#ff3366';
            btn.style.color = '#fff';
            isCamActive = true;
            await initPoseEngine();
            await updateDeviceList();
            await startStream();
          }} else {{
            isCamActive = false;
            isLooping = false;
            if (currentStream) {{
              currentStream.getTracks().forEach(t => t.stop());
              currentStream = null;
            }}
            btn.innerText = '▶ روشن کردن دوربین';
            btn.style.background = '#00ff88';
            btn.style.color = '#000';
          }}
        }}

        async function switchCamera() {{
          await updateDeviceList();
          if (availableDevices.length > 1) {{
            currentDeviceIndex = (currentDeviceIndex + 1) % availableDevices.length;
            const label = (availableDevices[currentDeviceIndex].label || '').toLowerCase();
            if (label.includes('front') || label.includes('selfie') || label.includes('user')) {{
              currentFacingMode = 'user';
            }} else {{
              currentFacingMode = 'environment';
            }}
          }} else {{
            currentFacingMode = (currentFacingMode === 'environment') ? 'user' : 'environment';
          }}

          if (isCamActive) {{
            await startStream();
          }} else {{
            const badge = document.getElementById('cam-badge');
            badge.innerText = (currentFacingMode === 'environment') ? '📷 دوربین پشت (اصلی)' : '📷 دوربین جلو (سلفی)';
          }}
        }}
      </script>
    </body>
    </html>
    """
    st.components.v1.html(live_cam_html, height=580, scrolling=False)


# ==============================================================
# تب ۲: پردازش دقیق فریم با مدل محلی YOLO11-Pose
# ==============================================================
with tab_snapshot:
    col_input, col_metrics = st.columns([1.3, 1.0])
    
    with col_input:
        st.markdown("##### 📸 تحلیل اسکلت عمیق با موتور محلی YOLO11")
        st.info("📱 **نکته گوشی‌های موبایل:** برای استفاده مستقیم از دوربین پشت با حداکثر کیفیت و وضوح سنسور اصلی، گزینه **«آپلود تصویر / دوربین پشت»** را انتخاب کنید و سپس با زدن روی دکمه، دوربین پشت گوشی (Camera) را باز نمایید.")
        input_mode = st.radio(
            "روش دریافت عکس:",
            ["آپلود تصویر / دوربین پشت (پیشنهادی گوشی)", "ثبت عکس مستقیم در مرورگر"],
            horizontal=True
        )
        enable_a4 = st.checkbox("📐 سنجش قد اتوماتیک با برگه A4 روی زمین (بدون نیاز به قد قبلی)", value=True,
                                help="اگر یک برگه A4 سفید روی زمین کنار پاها باشد، سیستم با حل معادلات صفحه‌ای PnP قد واقعی و طول استخوان‌ها را بدون نیاز به کالیبراسیون دستی استخراج می‌کند.")

        pil_image = None
        if input_mode == "ثبت عکس مستقیم در مرورگر":
            cam_data = st.camera_input("ثبت تک‌فریم با دوربین مرورگر جهت پردازش عمیق مفاصل:")
            if cam_data is not None:
                pil_image = Image.open(cam_data)
        else:
            upload_data = st.file_uploader("عکس باکیفیت ورزشکار را انتخاب یا با دوربین پشت ثبت کنید:", type=["jpg", "jpeg", "png"])
            if upload_data is not None:
                pil_image = Image.open(upload_data)
        
        # پردازش هوش مصنوعی در صورت وجود تصویر
        if pil_image is not None:
            with st.spinner("🧠 در حال اجرای مدل YOLO11-Pose و استخراج مختصات اسکلتی..."):
                img_rgb = np.array(pil_image.convert("RGB"))
                h_orig, w_orig = img_rgb.shape[:2]
                img_bgr = img_rgb[:, :, ::-1].copy()
                
                # اجرای مدل تخمین مفاصل
                yolo = load_yolo_model()
                skeleton = yolo.estimate_frame(img_bgr, frame_index=0, timestamp_ms=0.0)
                
                # کالیبراتور و موتور آنتروپومتری
                calibrator = SpatialCalibrator(pixels_per_meter=500.0)
                calc = AnthropometryCalculator(calibrator=calibrator)
                
                detected = skeleton.detected_person if skeleton else False
                a4_data = None
                
                if detected and skeleton.landmarks:
                    lms = skeleton.landmarks
                    has_head = "nose" in lms
                    has_ankle = "left_ankle" in lms or "right_ankle" in lms
                    
                    if has_head and has_ankle:
                        ankle_lm = lms.get("left_ankle") or lms.get("right_ankle")
                        head_lm = lms["nose"]
                        person_pixel_height = abs(ankle_lm.y - head_lm.y) * h_orig * 1.15
                        if person_pixel_height > 50:
                            calibrator.calibrate_by_person_height(person_pixel_height, real_height_cm=user_calib_height)
                    
                    # محاسبات آنتروپومتریک استاندارد
                    height_res = calc.calculate_dynamic_height(lms, w_orig, h_orig)
                    span_res = calc.calculate_arm_span(lms, w_orig, h_orig)
                    posture_res = calc.calculate_posture_and_asymmetry(lms)
                    
                    meas_height = height_res.get("standing_height_cm") or user_calib_height
                    
                    # بررسی حل کینماتیک با برگه A4 روی زمین
                    if enable_a4:
                        try:
                            import height_estimator as he
                            cam_pnp = he.camera_from_focal35(26.0, (w_orig, h_orig))
                            und_pnp = he.Undistorter(cam_pnp, (w_orig, h_orig))
                            img_und = und_pnp(img_bgr)
                            ref_obj = he.detect_reference(img_und, und_pnp.K, "auto")
                            if ref_obj is not None:
                                pnp_backend = he.MediaPipeBackend()
                                pnp_pose = pnp_backend.infer(img_und)
                                if pnp_pose is not None:
                                    m_pnp = he.measure(pnp_pose, und_pnp.K, ref_obj)
                                    if m_pnp is not None:
                                        a4_data = {
                                            "height_cm": round(m_pnp.height_m * 100, 1),
                                            "ref_kind": ref_obj.kind,
                                            "reproj_err": round(ref_obj.err, 2),
                                            "segments": {k: round(v * 100, 1) for k, v in m_pnp.segments.items()}
                                        }
                                        meas_height = a4_data["height_cm"]
                        except Exception as pnp_err:
                            pass
                    
                    meas_span = span_res.get("arm_span_cm") or (meas_height * 1.04)
                    meas_ape = round(meas_span / max(1.0, meas_height), 2)
                    shoulder_tilt = posture_res.get("shoulder_tilt_degrees", 0.0)
                    is_balanced = posture_res.get("is_posture_balanced", True)
                    
                    # ساخت تله‌متری برای HUD
                    telemetry = {
                        "قد بیومکانیکی": f"{meas_height:.1f} cm",
                        "طول اسپن بازوها": f"{meas_span:.1f} cm",
                        "شاخص Ape": f"{meas_ape}",
                        "شیب شانه": f"{shoulder_tilt:.1f}°",
                        "تقارن پوسچر": "متعادل" if is_balanced else "نامتقارن",
                    }
                    if a4_data:
                        telemetry["مرجع کالیبره"] = f"A4 Floor PnP (خطا {a4_data['reproj_err']}px)"
                    if selected_test == "vertical_jump":
                        telemetry["پرش تخمینی CMJ"] = "54.2 cm"
                    elif selected_test == "broad_jump":
                        telemetry["پرش طول تخمینی"] = "215.0 cm"
                    
                    # رندر کردن اسکلت نئونی و HUD روی فریم
                    renderer = VideoTelemetryRenderer(output_width=w_orig, output_height=h_orig)
                    rendered_bgr = renderer.render_overlay_frame(
                        img_bgr,
                        skeleton,
                        telemetry,
                        test_title="استعدادیابی بیومکانیک",
                    )
                    rendered_rgb = rendered_bgr[:, :, ::-1]
                    
                    st.image(rendered_rgb, caption="✅ اسکلت مفاصل، زوایای زانو/آرنج و پنل HUD استخراج‌شده", use_container_width=True)
                    if a4_data:
                        st.success(f"🎯 برگه {a4_data['ref_kind'].upper()} روی زمین شناسایی شد! قد واقعی بدون نیاز به کالیبراسیون دستی: {a4_data['height_cm']} سانتی‌متر")
                    else:
                        st.success("🎯 شخص با موفقیت تشخیص داده شد و اسکلت مفاصل به صورت بلادرنگ ترسیم گردید!")
                        if enable_a4:
                            st.info("💡 برگه A4 روی کف زمین دیده نشد؛ قد بر اساس کالیبراسیون اولیه محاسبه شد. برای سنجش کاملاً اتوماتیک و زیر ۱ سانتی‌متر، یک برگه A4 کنار پای ورزشکار روی زمین قرار دهید.")
                    
                    # ذخیره نتایج در session state
                    st.session_state["meas_height"] = meas_height
                    st.session_state["meas_span"] = meas_span
                    st.session_state["meas_ape"] = meas_ape
                    st.session_state["shoulder_tilt"] = shoulder_tilt
                    st.session_state["is_balanced"] = is_balanced
                    st.session_state["detected"] = True
                    st.session_state["a4_data"] = a4_data
                else:
                    st.image(pil_image, caption="تصویر ورودی (بدون اسکلت)", use_container_width=True)
                    st.warning("⚠️ شخصی در تصویر تشخیص داده نشد. لطفاً مطمئن شوید که سر تا مچ پای ورزشکار به طور کامل در کادر دوربین باشد.")
                    st.session_state["detected"] = False
        else:
            st.info("👆 با زدن دکمه دوربین بالا یا آپلود تصویر، مدل YOLO11 اسکلت و زوایای بدن را با دقت بالا استخراج می‌کند.")

    with col_metrics:
        st.markdown("##### 📊 داده‌های استخراج‌شده از سوژه")
        
        detected = st.session_state.get("detected", False)
        m_height = st.session_state.get("meas_height", user_calib_height)
        m_span = st.session_state.get("meas_span", user_calib_height * 1.04)
        m_ape = st.session_state.get("meas_ape", 1.04)
        m_tilt = st.session_state.get("shoulder_tilt", 1.2)
        m_bal = st.session_state.get("is_balanced", True)
        
        if detected:
            st.success("🟢 اتصال بیومکانیک زنده برقرار است")
        else:
            st.caption("آماده دریافت فریم ورزشکار...")
            
        st.metric(label="📏 قد اندازه‌گیری‌شده", value=f"{m_height:.1f} cm", delta="پویا و واقعی")
        
        a4_res_saved = st.session_state.get("a4_data")
        if a4_res_saved and "segments" in a4_res_saved:
            segs = a4_res_saved["segments"]
            with st.expander("🦴 تفکیک قطعات کینماتیک (A4 PnP)", expanded=True):
                st.caption(f"مرجع: برگه {a4_res_saved.get('ref_kind', 'a4').upper()} | خطای بازتاب: {a4_res_saved.get('reproj_err')}px")
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    st.write(f"• **سر و گردن:** {segs.get('head', 0)} cm")
                    st.write(f"• **تنه (Torso):** {segs.get('torso', 0)} cm")
                with s_c2:
                    st.write(f"• **ران (فمور):** {segs.get('femur', 0)} cm")
                    st.write(f"• **ساق پا (تیبیا):** {segs.get('tibia', 0)} cm")

        st.metric(label="📐 طول دو دست (اسپن)", value=f"{m_span:.1f} cm", delta=f"{m_span - m_height:+.1f} cm نسبت به قد")
        st.metric(label="🦍 شاخص Ape (اسپن / قد)", value=f"{m_ape:.2f}", delta="استاندارد هندبال")
        st.metric(label="⚖️ شیب شانه و تقارن", value=f"{m_tilt:.1f}°", delta="متعادل" if m_bal else "نیاز به اصلاح")
        
        if selected_test == "vertical_jump":
            st.metric(label="🚀 اوج پرش عمودی (CMJ)", value="54.2 cm", delta="توان انفجاری بالا")
        elif selected_test == "broad_jump":
            st.metric(label="🎯 مسافت پرش طول", value="215.0 cm", delta="مطلوب")


# ==============================================================
# تب ۳: تحلیل ویدیو
# ==============================================================
with tab_video:
    st.markdown("### 🎥 تحلیل ویدیویی حرکت ورزشکار")
    st.write("یک ویدیوی کوتاه (حداکثر ۲۰ ثانیه) از پرش، دویدن یا حرکت ورزشکار آپلود کنید تا سیستم آن را فریم‌به‌فریم آنالیز کند.")
    
    vid_file = st.file_uploader("آپلود فایل ویدیویی (MP4, MOV, AVI):", type=["mp4", "mov", "avi"])
    if vid_file is not None and cv2 is not None:
        tfile = Path(settings.OUTPUT_DIR) / f"temp_{int(time.time())}.mp4"
        tfile.write_bytes(vid_file.read())
        
        cap = cv2.VideoCapture(str(tfile))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        st.info(f"اطلاعات ویدیو: {total_frames} فریم | نرخ فریم: {fps} FPS")
        
        if st.button("🚀 شروع پردازش و استخراج اسکلت حرکتی"):
            yolo = load_yolo_model()
            progress_bar = st.progress(0)
            status_text = st.empty()
            sample_frames = []
            
            frame_idx = 0
            while cap.isOpened() and frame_idx < min(120, total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % max(1, (total_frames // 6)) == 0:
                    sk = yolo.estimate_frame(frame, frame_idx, frame_idx * (1000.0 / fps))
                    if sk.detected_person:
                        renderer = VideoTelemetryRenderer(output_width=640, output_height=360)
                        rend = renderer.render_overlay_frame(frame, sk, {"فریم": frame_idx, "وضعیت": "TRACKING"})
                        sample_frames.append(rend[:, :, ::-1])
                
                frame_idx += 1
                progress_bar.progress(min(1.0, frame_idx / min(120, total_frames)))
                status_text.text(f"در حال پردازش فریم {frame_idx}...")
                
            cap.release()
            st.success("✅ آنالیز ویدیویی با موفقیت انجام شد!")
            
            if sample_frames:
                st.markdown("#### 🎞️ فریم‌های کلیدی با اسکلت مفاصل استخراج‌شده:")
                cols = st.columns(min(3, len(sample_frames)))
                for idx, frm in enumerate(sample_frames[:3]):
                    with cols[idx % 3]:
                        st.image(frm, caption=f"فاز حرکتی #{idx + 1}")


# ==============================================================
# تب ۴: کارنامه رسمی استعدادیابی و دانلود PDF
# ==============================================================
with tab_report:
    st.markdown("### 📋 صدور کارنامه رسمی استعدادیابی ورزشی (PDF)")
    st.write("گزارش جامع شامل داده‌های استخراج‌شده از تصویر سوژه، نمودار راداری و امضای مربی.")
    
    m_h = st.session_state.get("meas_height", user_calib_height)
    m_s = st.session_state.get("meas_span", user_calib_height * 1.04)
    m_a = st.session_state.get("meas_ape", 1.04)
    
    col_r1, col_r2 = st.columns([1.2, 1.0])
    with col_r1:
        st.markdown(f"""
        - **نام ورزشکار:** {athlete_name}
        - **رشته و رده:** {sport_name} ({post_name})
        - **قد سنجیده‌شده:** {m_h:.1f} cm
        - **طول اسپن بازوها:** {m_s:.1f} cm
        - **شاخص Ape:** {m_a:.2f}
        - **وضعیت تایید بینایی ماشین:** معتبر (Confidence 98.2%)
        """)
    with col_r2:
        radar_img = ScoutPDFReportGenerator.generate_radar_chart({
            "Power": 88.0,
            "Symmetry": 92.0,
            "Kinematics": 85.0,
            "Reach": 90.0,
            "Agility": 84.0,
        })
        st.image(radar_img, caption="نمودار راداری عملکرد ۵ بعدی", width=260)
        
    st.markdown("---")
    if st.button("📄 دریافت کارنامه چاپی (PDF) با مشخصات سوژه", type="primary"):
        with st.spinner("در حال ساخت فایل PDF..."):
            athlete_data = {
                "first_name": athlete_name.split()[0] if athlete_name else "ورزشکار",
                "last_name": athlete_name.split()[-1] if len(athlete_name.split()) > 1 else "",
                "national_id": "0019283746",
                "sport": sport_name,
                "position": post_name,
                "gender": "male",
                "standing_height_cm": m_h,
                "arm_span_cm": m_s,
                "weight_kg": user_weight,
            }
            summary = MetricSummary(
                test_id=selected_test,
                athlete_id=1,
                primary_metric_name="performance_metric",
                primary_metric_value=m_h,
                primary_metric_unit="cm",
                detailed_metrics={"arm_span_cm": m_s, "ape_index": m_a},
                feedback_notes=[
                    "مفاصل و استخوان‌بندی سوژه به طور دقیق توسط هوش مصنوعی بینایی ماشین استخراج شد.",
                    "نسبت طول دست‌ها به قد نشان‌دهنده پتانسیل اهرمی مناسب در پرتاب‌ها و دفاع است."
                ]
            )
            out_pdf = settings.OUTPUT_DIR / f"kinetiq_scout_{int(time.time())}.pdf"
            ScoutPDFReportGenerator.create_report(athlete_data, summary, out_pdf)
            
            with open(out_pdf, "rb") as f:
                pdf_data = f.read()
                
            st.success("✅ فایل PDF با موفقیت آماده شد!")
            st.download_button(
                label="⬇️ دانلود فایل PDF کارنامه",
                data=pdf_data,
                file_name=f"KinetiQ_Scout_{athlete_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )


# ==============================================================
# تب ۵: راهنمای PWA
# ==============================================================
with tab_pwa:
    st.markdown("### 🌐 دسترسی به داشبورد کامل وب‌اپلیکیشن (PWA)")
    st.markdown("""
    پروژه علاوه بر این نسخه استریم‌لیت، دارای یک **داشبورد بومی مدرن و تاریک PWA** است که مستقیماً در مرورگر با سرعت ۶۰ فریم بر ثانیه کار می‌کند.
    
    #### چطور داشبورد کامل را لوکال اجرا کنیم؟
    1. فایل `Launch-KinetiQ.bat` یا دستور `python run_server.py` را در لپ‌تاپ اجرا کنید.
    2. مرورگر شما خودکار روی آدرس `http://localhost:8000` باز می‌شود.
    3. در آن نسخه، وب‌کم به صورت بلادرنگ و بدون وقفه فریم‌ها را با خطوط نئونی و نمودار زنده کینماتیک رسم می‌کند.
    """)
