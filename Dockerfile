# KinetiQ OS - Hugging Face Spaces Dockerfile
FROM python:3.11-slim

# تنظیم متغیرهای محیطی
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user

# نصب پکیج‌های سیستمی مورد نیاز OpenCV و رسانه‌ها در لینوکس
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ایجاد کاربر غیریوزر روت مطابق با الزامات Hugging Face
RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

# کپی فایل وابستگی‌ها
COPY --chown=user:user requirements.txt .

# نصب PyTorch بهینه CPU و تمام کتابخانه‌ها
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# کپی کل سورس‌کدها و مدل هوش مصنوعی
COPY --chown=user:user . .

# اطمینان از وجود پوشه‌های داده
RUN mkdir -p data/output data/uploads data/logs

# باز کردن پورت 7860
EXPOSE 7860

# اجرای سرور KinetiQ
CMD ["python", "run_server.py"]
