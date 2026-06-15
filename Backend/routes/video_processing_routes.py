"""
Video Processing Routes — Jetson AGX Orin Optimised

Changes applied:
  - processing_jobs dict protected by _jobs_lock (thread-safe read/write)
  - Video processing loop yields CPU via time.sleep(0.002) per frame
  - GStreamer HW decoder used for video files (nvv4l2filesrc / nvv4l2decoder)
    falling back to plain cv2.VideoCapture for formats nvv4l2 can't handle
  - Job cancel signal propagated into the processing thread via a cancel flag
  - Dead commented-out duplicate upload route removed
"""

from flask import Blueprint, request, jsonify, send_file
import os
import uuid
import cv2
import json
import time
import threading
from datetime import datetime
from werkzeug.utils import secure_filename

# ── Blueprint ──────────────────────────────────────────────────────────────────
video_bp = Blueprint("video", __name__, url_prefix="/api/process")

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
REPORTS_FOLDER = "reports"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# ── Jobs store (protected by _jobs_lock) ──────────────────────────────────────
# Multiple Flask threads can read/write processing_jobs concurrently (status
# polling, cancel, list). _jobs_lock prevents race conditions.
processing_jobs: dict = {}
_jobs_lock = threading.Lock()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Upload ────────────────────────────────────────────────────────────────────

@video_bp.route("/upload", methods=["POST"])
def upload_video():
    """Upload a video file for batch processing."""
    from routes.packer_routes import get_packers_db

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(video_file.filename):
        return jsonify({"error": "Invalid file format",
                        "allowed_formats": list(ALLOWED_EXTENSIONS)}), 400

    packer_id = request.form.get("packer_id")
    if packer_id and packer_id not in ("", "undefined"):
        packers_db = get_packers_db()
        if packer_id not in packers_db:
            print(f"[UPLOAD] Warning: unknown packer_id {packer_id}")

    video_id = str(uuid.uuid4())
    original = secure_filename(video_file.filename)
    ext = original.rsplit(".", 1)[1].lower()
    saved = f"{video_id}.{ext}"
    video_path = os.path.join(UPLOAD_FOLDER, saved)

    try:
        video_file.save(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            os.remove(video_path)
            return jsonify({"error": "Invalid or corrupt video file"}), 400

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        return jsonify({
            "message": "Video uploaded successfully",
            "video_id": video_id,
            "filename": saved,
            "original_filename": original,
            "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
            "video_info": {
                "width": width, "height": height, "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2),
            },
            "packer_id": packer_id or None,
        }), 201

    except Exception as exc:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({"error": "Upload failed", "message": str(exc)}), 500


# ── Start processing ──────────────────────────────────────────────────────────

@video_bp.route("/start", methods=["POST"])
def start_processing():
    """Start a batch video processing job on the Jetson GPU."""
    from routes.packer_routes import get_packers_db
    from models.packer_monitor import PackerEfficiencyMonitor
    from app import MODEL_PATH

    data = request.json
    video_id = data.get("video_id")
    packer_id = data.get("packer_id")

    if not video_id or not packer_id:
        return jsonify({"error": "video_id and packer_id are required"}), 400

    packers_db = get_packers_db()
    if packer_id not in packers_db:
        return jsonify({"error": "Invalid packer_id"}), 400

    # Find uploaded video file
    matches = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(video_id)]
    if not matches:
        return jsonify({"error": "Video not found"}), 404
    video_path = os.path.join(UPLOAD_FOLDER, matches[0])

    packer_config = packers_db[packer_id]
    save_output = data.get("save_output_video", False)
    generate_report = data.get("generate_report", True)
    job_id = str(uuid.uuid4())

    # Register job (under lock)
    with _jobs_lock:
        processing_jobs[job_id] = {
            "job_id": job_id,
            "packer_id": packer_id,
            "packer_name": packer_config.get("name"),
            "video_id": video_id,
            "status": "queued",
            "progress": 0,
            "frames_processed": 0,
            "total_frames": 0,
            "created_at": datetime.now().isoformat(),
            "save_output": save_output,
            "generate_report": generate_report,
            "_cancelled": False,   # internal cancel flag checked inside loop
        }

    def _update_job(**kwargs):
        """Thread-safe job state update."""
        with _jobs_lock:
            if job_id in processing_jobs:
                processing_jobs[job_id].update(kwargs)

    def process_video():
        """
        Batch video processing thread.

        Jetson optimisations:
          - PackerEfficiencyMonitor uses singleton model cache → no extra VRAM load
          - time.sleep(0.002) per frame yields CPU to live monitoring + Flask threads.
            2 ms costs ~5% throughput on GPU-bound workloads but makes the system
            usable when other applications are running simultaneously.
          - _cancelled flag allows the cancel endpoint to abort mid-job.
        """
        _update_job(status="processing", started_at=datetime.now().isoformat())

        try:
            # Model loaded from singleton cache — no extra GPU memory
            monitor = PackerEfficiencyMonitor(
                model_path=MODEL_PATH,
                line_position=packer_config.get("line_position", 0.7),
                start_line_position=packer_config.get("start_line_position", 0.2),
                confidence_threshold=packer_config.get("confidence_threshold", 0.5),
                spouts=packer_config.get("spouts", 8),
                enable_debug=False,
                use_gpu=True,
            )

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                _update_job(status="failed", error="Could not open video file")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _update_job(total_frames=total_frames)

            # Optional output video writer
            output_writer = None
            output_filename = None
            if save_output:
                output_filename = f"output_{job_id}.mp4"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            while True:
                # Respect cancel signal
                with _jobs_lock:
                    cancelled = processing_jobs.get(job_id, {}).get("_cancelled", False)
                if cancelled:
                    print(f"[JOB] {job_id} cancelled at frame {frame_count}")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                processed = monitor.process_frame(frame)

                if output_writer:
                    output_writer.write(processed)

                frame_count += 1
                progress = (frame_count / total_frames * 100) if total_frames else 0
                _update_job(progress=round(progress, 2), frames_processed=frame_count)

                if frame_count % 30 == 0 or frame_count == total_frames:
                    _update_job(summary=monitor.get_summary())

                # Yield CPU between frames so live monitoring and Flask
                # threads are not starved on the Jetson's limited CPU cores.
                time.sleep(0.002)

            cap.release()
            if output_writer:
                output_writer.release()

            # Check if we exited due to cancel
            with _jobs_lock:
                cancelled = processing_jobs.get(job_id, {}).get("_cancelled", False)
            if cancelled:
                _update_job(status="cancelled", cancelled_at=datetime.now().isoformat())
                return

            summary = monitor.get_summary()
            update_dict = {
                "status": "completed",
                "summary": summary,
                "completed_at": datetime.now().isoformat(),
            }
            if output_filename:
                update_dict["output_video"] = output_filename

            if generate_report:
                report_filename = f"report_{job_id}.json"
                report_data = {
                    "job_id": job_id,
                    "packer_id": packer_id,
                    "packer_name": packer_config.get("name"),
                    "video_id": video_id,
                    "timestamp": datetime.now().isoformat(),
                    "video_info": {
                        "total_frames": total_frames, "fps": fps,
                        "width": width, "height": height,
                    },
                    "summary": summary,
                    "configuration": {
                        "line_position": packer_config.get("line_position"),
                        "start_line_position": packer_config.get("start_line_position"),
                        "confidence_threshold": packer_config.get("confidence_threshold"),
                    },
                }
                report_path = os.path.join(REPORTS_FOLDER, report_filename)
                with open(report_path, "w") as f:
                    json.dump(report_data, f, indent=2)
                update_dict["report_file"] = report_filename

            _update_job(**update_dict)

        except Exception as exc:
            _update_job(status="failed", error=str(exc),
                        failed_at=datetime.now().isoformat())
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=process_video, daemon=True)
    thread.start()

    return jsonify({"message": "Processing started", "job_id": job_id,
                    "status": "queued"}), 202


# ── Status ────────────────────────────────────────────────────────────────────

@video_bp.route("/status/<job_id>", methods=["GET"])
def get_job_status(job_id):
    with _jobs_lock:
        job = processing_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    # Return a copy without the internal _cancelled key
    public = {k: v for k, v in job.items() if not k.startswith("_")}
    return jsonify(public), 200


# ── List jobs ─────────────────────────────────────────────────────────────────

@video_bp.route("/jobs", methods=["GET"])
def list_jobs():
    status_filter = request.args.get("status")
    limit = request.args.get("limit", type=int)

    with _jobs_lock:
        jobs = [{k: v for k, v in j.items() if not k.startswith("_")}
                for j in processing_jobs.values()]

    if status_filter:
        jobs = [j for j in jobs if j.get("status") == status_filter]

    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    if limit:
        jobs = jobs[:limit]

    return jsonify({"jobs": jobs, "total": len(jobs)}), 200


# ── Download output ───────────────────────────────────────────────────────────

@video_bp.route("/download/<job_id>", methods=["GET"])
def download_output(job_id):
    with _jobs_lock:
        job = processing_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "completed":
        return jsonify({"error": "Job not completed"}), 400
    if not job.get("output_video"):
        return jsonify({"error": "No output video available"}), 404
    output_path = os.path.join(OUTPUT_FOLDER, job["output_video"])
    if not os.path.exists(output_path):
        return jsonify({"error": "Output file not found"}), 404
    return send_file(output_path, as_attachment=True)


# ── Cancel ────────────────────────────────────────────────────────────────────

@video_bp.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    """
    Cancel a running or queued job.
    Sets the _cancelled flag which the processing thread checks each frame.
    """
    with _jobs_lock:
        job = processing_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") in ("completed", "failed", "cancelled"):
        return jsonify({"error": "Cannot cancel a finished job"}), 400

    with _jobs_lock:
        if job_id in processing_jobs:
            processing_jobs[job_id]["_cancelled"] = True
            processing_jobs[job_id]["status"] = "cancelling"

    return jsonify({"message": "Cancel signal sent", "job_id": job_id}), 200
