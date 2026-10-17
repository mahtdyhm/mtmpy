# KinetiQ OS ⚡
> **Sports Vision AI & Offline-First Athletic Scouting Platform**

KinetiQ OS is an offline-first, zero-config sports vision AI and kinematic analysis engine designed for athletic talent identification, biomechanical screening, and tactical sports analytics (including Handball, Vertical Jump, and Anthropometry).

---

## 🌟 Key Features

- **Hybrid AI Pose Estimation:**
  - Fast offline edge inference via **YOLO11 Pose** & **MediaPipe**.
  - Optional cloud fallback with **VLM (Vision-Language Models)**.
  - Multi-person tracking and object tracking (ball possession).
- **Advanced Biomechanics Engine:**
  - Real-time kinematic jitter reduction using the **1€ (One-Euro) Filter**.
  - **Spatial Calibrator** for pixel-to-metric conversions without dedicated hardware markers.
  - Full **Anthropometric** measurement (Standing height, Arm span, Ape index).
- **Modular Sports Plugin Architecture:**
  - 📏 **Anthropometry Test:** Standing height, reach, wingspan, and Ape index analysis.
  - 🚀 **Vertical Jump (CMJ / Bosco Test):** Newtonian free-fall flight time calculation, Reactive Strength Index (RSI), and fatigue monitoring.
  - 🤾 **Handball Test:** Ball possession detection, step-rhythm cadence, and 3-step violation / traveling detection.
- **Enterprise-Ready Backend:**
  - Built with **FastAPI** & **SQLModel** (SQLite offline database).
  - RBAC security & JWT authentication (SuperAdmin, Coach, Athlete).
  - High-frequency live telemetry via **WebSockets**.
- **Interactive UI & Scouting Export:**
  - Modern dark-themed dashboard.
  - One-click talent report generation (CSV / JSON / Printable Scouting Cards).

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- Conda or virtualenv

### 2. Run the Platform
You can start the server directly using:
`ash
# Windows quick launcher:
start_kinetiq.bat

# Or manually:
python run_server.py
`

The web dashboard will be available at:
👉 http://localhost:8000

Default credentials:
- **Username:** dmin
- **Password:** dmin123

---

## 🧪 Testing & Validation

The entire system is thoroughly tested across 7 architectural phases:
`ash
pytest
`
*31 passed unit and end-to-end integration tests.*

---

## 📄 License
This project is licensed under the MIT License.
