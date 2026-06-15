"""
Flask API Backend — Jetson AGX Orin Optimised
PackerVision AI — Main Application

Changes vs previous version:
  - ShiftScheduler integrated: auto-starts/stops monitoring at 6AM, 2PM, 10PM.
  - New endpoints: GET /api/shift/status, POST /api/shift/override (manual override).
  - debug=False kept (prevents Werkzeug double-load on Jetson).
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_mail import Mail, Message
import time
import os
from database import init_db, get_db_connection

# ── App initialisation ────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

init_db()


def reset_stale_statuses():
    """Reset all packer statuses to idle on startup (clears any crash leftovers)."""
    try:
        conn = get_db_connection()
        conn.execute("UPDATE packers SET status='idle', session_id=NULL")
        conn.commit()
        conn.close()
        print("[STARTUP] All packer statuses reset to idle.")
    except Exception as exc:
        print(f"[STARTUP] Reset error: {exc}")


reset_stale_statuses()

# ── Email ─────────────────────────────────────────────────────────────────────
# TODO: Move credentials to environment variables before deploying externally.
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "sakshitandon1193@gmail.com"
app.config["MAIL_PASSWORD"] = "xpye gnab pfkm ctna"
app.config["MAIL_DEFAULT_SENDER"] = "sakshi.tandon@amzbizsol.in"

mail = Mail(app)
from report_mailer import DailyDigestScheduler
_digest_scheduler = DailyDigestScheduler(app)

@app.route("/api/send_feedback", methods=["POST"])
def send_feedback():
    data = request.json
    overall     = data.get("overallExperience", 0)
    ease        = data.get("easeOfUse", 0)
    performance = data.get("applicationPerformance", 0)
    comments    = data.get("comments", "")
    try:
        msg = Message(
            subject="New PackerVision AI Feedback",
            recipients=["recipient-sakshitandon1193@gmail.com"],
            body=(
                f"New Feedback Received:\n"
                f"Overall Experience: {overall}/5 Stars\n"
                f"Ease of Use: {ease}/5 Stars\n"
                f"Application Performance: {performance}/5 Stars\n"
                f"User Comments:\n{comments}"
            ),
        )
        mail.send(msg)
        return jsonify({"status": "success", "message": "Feedback sent"}), 200
    except Exception as exc:
        print(f"[MAIL] Error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


# ── App config ────────────────────────────────────────────────────────────────
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB
app.config["UPLOAD_FOLDER"]  = "uploads"
app.config["OUTPUT_FOLDER"]  = "outputs"
app.config["REPORTS_FOLDER"] = "reports"

MODEL_PATH    = "best.pt"
app_start_time = time.time()

os.makedirs("uploads",  exist_ok=True)
os.makedirs("outputs",  exist_ok=True)
os.makedirs("reports",  exist_ok=True)

# ── Blueprint registration ────────────────────────────────────────────────────
from routes.dashboard_routes       import dashboard_bp
from routes.packer_routes          import packer_bp
from routes.monitoring_routes      import monitoring_bp
from routes.video_processing_routes import video_bp
from routes.reports_routes         import reports_bp
from routes.camera_routes          import camera_bp
from routes.auth_routes            import auth_bp

app.register_blueprint(dashboard_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(packer_bp)
app.register_blueprint(monitoring_bp)
app.register_blueprint(video_bp)
app.register_blueprint(reports_bp)
app.register_blueprint(camera_bp)


# ── Shift Scheduler ───────────────────────────────────────────────────────────
from shift_scheduler import ShiftScheduler

_shift_scheduler = ShiftScheduler(app)

# Auto-mode is ON by default.  Set to False here (or via env var) to disable.
AUTO_SHIFT_ENABLED = os.environ.get("AUTO_SHIFT_ENABLED", "true").lower() != "false"

if AUTO_SHIFT_ENABLED:
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        _shift_scheduler.start()
        _digest_scheduler.start()
        print("[APP] Shift auto-monitoring ENABLED")
else:
    print("[APP] Shift auto-monitoring DISABLED (set AUTO_SHIFT_ENABLED=true to enable)")


# ── Shift control / status endpoints ─────────────────────────────────────────

@app.route("/api/shift/status", methods=["GET"])
def shift_status():
    """
    Returns the current shift, active auto-sessions, and minutes until next boundary.
    Frontend can poll this at 60-second intervals for a live shift indicator.
    """
    from shift_scheduler import get_current_shift, SHIFTS
    from routes.monitoring_routes import active_sessions, _sessions_lock
    from datetime import datetime

    now = datetime.now()
    current = get_current_shift()

    with _sessions_lock:
        auto_sessions = [
            {"session_id": sid, "packer_id": sess["packer_id"], "shift": sess.get("shift")}
            for sid, sess in active_sessions.items()
            if sess.get("auto")
        ]

    sched_status = _shift_scheduler.get_status()

    return jsonify({
        "auto_mode_enabled": AUTO_SHIFT_ENABLED,
        "current_shift": current,
        "shift_label": SHIFTS[current]["label"],
        "current_time": now.strftime("%H:%M:%S"),
        "auto_sessions": auto_sessions,
        "auto_sessions_count": len(auto_sessions),
        "minutes_to_next_change": sched_status["next_shift_change"],
        "scheduler": sched_status,
    }), 200


@app.route("/api/shift/override", methods=["POST"])
def shift_override():
    """
    Manual override — force-trigger a shift transition immediately.
    Useful for testing or for edge cases (e.g. unscheduled shift swap).

    Body: { "action": "restart" | "stop_all" }
    """
    data   = request.json or {}
    action = data.get("action", "restart")

    from shift_scheduler import get_current_shift

    if action == "stop_all":
        # Stop all auto-sessions without starting new ones
        with _shift_scheduler._sched_lock:
            sessions = dict(_shift_scheduler._packer_sessions)
            _shift_scheduler._packer_sessions = {}
        for packer_id, session_id in sessions.items():
            _shift_scheduler._stop_session(packer_id, session_id,
                                            _shift_scheduler._current_shift)
        return jsonify({"message": "All auto-sessions stopped"}), 200

    elif action == "restart":
        # Stop current auto-sessions and restart for the current shift
        current = get_current_shift()
        with _shift_scheduler._sched_lock:
            old_sessions = dict(_shift_scheduler._packer_sessions)
            _shift_scheduler._packer_sessions = {}
        for packer_id, session_id in old_sessions.items():
            _shift_scheduler._stop_session(packer_id, session_id,
                                            _shift_scheduler._current_shift)
        started = _shift_scheduler._start_all_packers(current)
        return jsonify({
            "message": f"Restarted {started} packer session(s) for shift {current}",
            "shift": current,
        }), 200

    return jsonify({"error": "Unknown action. Use 'restart' or 'stop_all'"}), 400


# ── Static / evidence serving ─────────────────────────────────────────────────
@app.route("/api/static/evidence/<session_id>/<filename>")
def serve_evidence(session_id, filename):
    evidence_dir = os.path.join("/media/amazin/store/evidences", session_id)
    if not os.path.exists(evidence_dir):
        return jsonify({"error": "Evidence directory not found"}), 404
    try:
        return send_from_directory(evidence_dir, filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health_check():
    from shift_scheduler import get_current_shift
    uptime = time.time() - app_start_time
    return jsonify({
        "status":         "healthy",
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_seconds": round(uptime, 2),
        "uptime_hours":   round(uptime / 3600, 2),
        "model_loaded":   os.path.exists(MODEL_PATH),
        "model_path":     MODEL_PATH,
        "current_shift":  get_current_shift(),
        "auto_shift":     AUTO_SHIFT_ENABLED,
    }), 200


# ── Root / docs ───────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service":  "PackerPro Efficiency Monitor API",
        "version":  "1.1.0",
        "status":   "running",
        "endpoints": {
            "dashboard":       "/api/dashboard",
            "packers":         "/api/packers",
            "monitoring":      "/api/monitor",
            "video_processing":"/api/process",
            "reports":         "/api/reports",
            "shift":           "/api/shift/status",
        },
        "documentation": "/api/docs",
    }), 200


@app.route("/api/docs", methods=["GET"])
def api_docs():
    return jsonify({
        "title":    "PackerPro API Documentation",
        "version":  "1.1.0",
        "base_url": "http://localhost:5001/api",
        "endpoints": {
            "Dashboard": {
                "GET /api/dashboard/stats":               "KPI cards",
                "GET /api/dashboard/recent-reports":      "Recent reports",
                "GET /api/dashboard/overview":            "Comprehensive overview",
                "GET /api/dashboard/performance-comparison": "Bar chart data",
            },
            "Packer Management": {
                "GET /api/packers":      "List packers",
                "POST /api/packers":     "Create packer",
                "GET /api/packers/{id}": "Packer details",
                "PUT /api/packers/{id}": "Update packer",
                "DELETE /api/packers/{id}": "Delete packer",
            },
            "Live Monitoring": {
                "POST /api/monitor/start":              "Start monitoring (manual)",
                "POST /api/monitor/stop/{id}":          "Stop monitoring",
                "GET /api/monitor/metrics/{id}":        "Live metrics",
                "GET /api/monitor/active-sessions":     "Active sessions",
                "GET /api/monitor/video_feed/{id}":     "MJPEG stream (10 FPS, 640×360)",
                "GET /api/monitor/test-camera/{id}":    "Test camera connection",
            },
            "Shift Auto-Monitoring": {
                "GET /api/shift/status":    "Current shift, active auto-sessions, next boundary",
                "POST /api/shift/override": "Force restart or stop-all auto-sessions",
            },
            "Video Processing": {
                "POST /api/process/upload":             "Upload video",
                "POST /api/process/start":              "Start batch processing",
                "GET /api/process/status/{job_id}":     "Job status",
                "GET /api/process/jobs":                "List jobs",
                "GET /api/process/download/{job_id}":   "Download output",
                "POST /api/process/cancel/{job_id}":    "Cancel job",
            },
            "Reports": {
                "GET /api/reports":                     "List reports (paginated, ?shift=A|B|C)",
                "GET /api/reports/{id}":                "Report details",
                "DELETE /api/reports/{id}":             "Delete report",
                "GET /api/reports/export-csv":          "Export CSV (?shift=A|B|C &from= &to=)",
                "GET /api/reports/shift-summary":       "Per-shift aggregated stats",
                "GET /api/reports/stats":               "Aggregate stats (?shift=A|B|C)",
            },
        },
    }), 200


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "status": 404}), 404

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "status": 400}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal Server Error", "status": 500}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File Too Large (max 500 MB)", "status": 413}), 413


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PackerPro Efficiency Monitor — Jetson AGX Orin")
    print("=" * 60)
    print(f"  Host        : http://0.0.0.0:5001")
    print(f"  Model       : {MODEL_PATH}  (exists={os.path.exists(MODEL_PATH)})")
    print(f"  Auto-Shift  : {'ON' if AUTO_SHIFT_ENABLED else 'OFF (set AUTO_SHIFT_ENABLED=true)'}")
    print(f"  Debug       : OFF")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        threaded=True,
    )

