# Packer Efficiency Monitoring System
### Full-stack industrial AI system for real-time packer performance tracking — deployed 24/7 on NVIDIA Jetson

![Status](https://img.shields.io/badge/status-production-brightgreen)
![Platform](https://img.shields.io/badge/platform-NVIDIA%20Jetson-76b900)
![Model](https://img.shields.io/badge/model-YOLOv8-00BFFF)
![Backend](https://img.shields.io/badge/backend-Flask%20REST%20API-blue)
![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61DAFB)

---

## Overview

A production-grade full-stack AI system that monitors packer efficiency on cement manufacturing lines in real time. The system uses YOLOv8 computer vision to detect and track packer activity via live RTSP camera feeds, computes per-shift efficiency metrics, and surfaces them through a React dashboard used daily by plant operations teams.

Deployed 24/7 across 4+ cement plants on NVIDIA Jetson edge devices with **100% counting accuracy**.

---

## Demo
<img width="1290" height="651" alt="image" src="https://github.com/user-attachments/assets/f448407d-6381-40ad-82ac-c1a2fdc87acd" />
<img width="1295" height="652" alt="image" src="https://github.com/user-attachments/assets/74fd44be-bacf-407a-b899-33b7bdcc1268" />
<img width="1292" height="650" alt="image" src="https://github.com/user-attachments/assets/f43c40be-c7ef-49df-b7ee-59b5370ec9c8" />
<img width="1293" height="660" alt="image" src="https://github.com/user-attachments/assets/87210f0f-4f11-4beb-9a22-dba1621c80d1" />

**Planned screenshots:**
- Live monitoring view with detection overlay
- Dashboard shift-wise efficiency report
- Packer Master configuration panel

---

## Key Metrics

| Metric | Value |
|---|---|
| Counting accuracy | 100% |
| Deployment | 24/7, 4+ cement plants |
| Hardware | NVIDIA Jetson (edge) |
| Reporting | Shift-wise automated reports |
| Alerts | Automated email via `report_mailer.py` |

---

## System Architecture

```
RTSP Camera Feed (per packer line)
        │
        ▼
┌─────────────────────────────────────┐
│  Flask Backend (app.py)             │
│                                     │
│  ┌──────────────────────────────┐   │
│  │  YOLOv8 Detection            │   │  ← best.pt / best.engine (TensorRT)
│  │  (video_processing_routes)   │   │
│  └──────────────┬───────────────┘   │
│                 │ detections        │
│  ┌──────────────▼───────────────┐   │
│  │  Packer Monitor Model        │   │  ← models/packer_monitor.py
│  │  Efficiency computation      │   │
│  └──────────────┬───────────────┘   │
│                 │ metrics           │
│  ┌──────────────▼───────────────┐   │
│  │  Database (database.py)      │   │  ← stores shift data, counts
│  └──────────────┬───────────────┘   │
│                 │                   │
│  REST API Routes:                   │
│  /auth          (auth_routes)       │
│  /camera        (camera_routes)     │
│  /dashboard     (dashboard_routes)  │
│  /monitoring    (monitoring_routes) │
│  /packer        (packer_routes)     │
│  /reports       (reports_routes)    │
│  /video         (video_processing)  │
└──────────────────┬──────────────────┘
                   │ JSON API
                   ▼
┌─────────────────────────────────────┐
│  React Frontend (Vite)              │
│                                     │
│  Login.jsx          — Auth          │
│  Dashboard.jsx      — Overview      │
│  Monitoring.jsx     — Live feed     │
│  PackerMaster.jsx   — Config        │
│  Reports.jsx        — Shift reports │
│  DataChart.jsx      — Analytics     │
│  Support.jsx        — Help          │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Automated Reporting                │
│  report_mailer.py  — email reports  │
│  shift_scheduler.py — shift timing  │
└─────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Inference | TensorRT engine (`best.engine`) |
| Backend | Python, Flask REST API |
| Database | SQLite / custom `database.py` |
| Frontend | React + Vite |
| Styling | CSS (`App.css`, `index.css`) |
| Email reporting | `report_mailer.py` |
| Shift scheduling | `shift_scheduler.py` |
| Edge hardware | NVIDIA Jetson |

---

## Project Structure

```
industrial-efficiency-monitoring-system/
│
├── Backend/
│   ├── models/
│   │   ├── __init__.py
│   │   └── packer_monitor.py        # Packer detection & efficiency logic
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth_routes.py           # Login / authentication
│   │   ├── camera_routes.py         # Camera feed management
│   │   ├── dashboard_routes.py      # Dashboard data endpoints
│   │   ├── monitoring_routes.py     # Live monitoring endpoints
│   │   ├── packer_routes.py         # Packer-specific operations
│   │   ├── reports_routes.py        # Shift report generation
│   │   └── video_processing_routes.py  # YOLOv8 inference pipeline
│   │
│   ├── utils/                       # Shared utility functions
│   ├── app.py                       # Flask app entry point
│   ├── database.py                  # DB connection & queries
│   ├── report_mailer.py             # Automated email alerts
│   ├── shift_scheduler.py           # Shift timing logic
│   ├── best.pt                      # YOLOv8 trained weights
│   ├── best.engine                  # TensorRT engine
│   ├── requirements.txt
│   └── test.py
│
├── Frontend/
│   ├── src/
│   │   ├── Dashboard.jsx            # Main overview dashboard
│   │   ├── DataChart.jsx            # Analytics & charts
│   │   ├── Login.jsx                # Authentication UI
│   │   ├── Monitoring.jsx           # Live camera monitoring
│   │   ├── PackerMaster.jsx         # Packer configuration
│   │   ├── Reports.jsx              # Shift-wise reports
│   │   ├── Support.jsx              # Help & support
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── main.jsx
│   │   └── index.css
│   │
│   ├── public/
│   ├── dist/                        # Production build output
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── eslint.config.js
│
├── UI-build/                        # Compiled frontend for deployment
└── README.md
```

---

## Setup & Usage

### Backend

```bash
cd Backend
pip install -r requirements.txt
python app.py
```

The Flask API will start on `http://0.0.0.0:5000` (or configured port).

### Frontend (Development)

```bash
cd Frontend
npm install
npm run dev
```

### Frontend (Production Build)

```bash
cd Frontend
npm run build
# Output goes to dist/ — served by Flask or Nginx
```

### TensorRT Engine (Jetson)

If running on Jetson with TensorRT:

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="engine", half=True, device=0)
# Generates best.engine for fast edge inference
```

---

## Key Features

**Live Monitoring** — `Monitoring.jsx` + `monitoring_routes.py`
Real-time YOLOv8 detection on RTSP camera feeds displayed in the browser dashboard.

**Packer Efficiency Computation** — `packer_monitor.py`
Detects packer activity per shift, computes efficiency metrics (bags packed, idle time, throughput), and stores results in the database.

**Shift-wise Reports** — `Reports.jsx` + `reports_routes.py`
Per-shift performance breakdowns accessible from the dashboard. Used daily by plant operations teams to track productivity.

**Automated Email Alerts** — `report_mailer.py`
End-of-shift reports emailed automatically to plant managers — no manual export needed.

**Shift Scheduling** — `shift_scheduler.py`
Handles shift boundary detection and triggers report generation at shift changeover.

**Authentication** — `auth_routes.py` + `Login.jsx`
Role-based login to restrict dashboard access to authorised plant personnel.

**Camera Management** — `camera_routes.py`
Configure and manage RTSP camera sources per packer line.

---

## Deployment Context

Built for industrial cement manufacturing — running on NVIDIA Jetson edge devices directly at plant sites. Edge deployment eliminates network latency and keeps camera feeds on-premise. The system has operated continuously across multiple plant shifts, replacing fully manual packer tracking with automated computer vision.

---

## Related Projects

Other production AI systems from the same industrial platform:

- **Bag Counting System** — YOLOv8 + SORT + TensorRT on Jetson Orin for real-time cement bag counting (99% accuracy, 100ms latency)
- **Predictive Maintenance Platform** — 2-stage XGBoost pipeline predicting equipment failure 30 minutes ahead with 94%+ accuracy
- **Firlobot** — LLM Text-to-SQL chatbot (LLaMA 3.3-70B) enabling natural language data queries for non-technical operators

---

## Author

**Sakshi Tandon** — Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/sakshi-tandon-865371249) · [GitHub](https://github.com/Sakshi13t)
