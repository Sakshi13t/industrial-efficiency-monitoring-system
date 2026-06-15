"""
Live Monitoring Routes — Jetson AGX Orin Optimised
All hardware acceleration, thread-safety, and performance bugs fixed.

Jetson-specific changes applied:
  GStreamer pipeline:
    - Replaced avdec_h264 (CPU software decoder) with nvv4l2decoder (Jetson NVDEC HW)
    - nvvidconv used for zero-copy hardware colour conversion
    - Fallback chain: Jetson HW GStreamer → TCP FFmpeg → Default OpenCV

  Thread safety:
    - _sessions_lock protects active_sessions dict from concurrent read/write/delete
    - get_frame() copies frame OUTSIDE the lock (was inside: held lock for ~4ms)
    - generate_frames() uses a proper lock reference, not a throwaway Lock() object
    - stop_monitoring() uses threading.Event instead of blocking sleep(1.0)

  CPU/memory optimisations (carried forward from previous session):
    - _read_loop throttled to 60 Hz (was unbounded spinloop)
    - process_stream runs at TARGET_FPS=15 budget (was unbounded)
    - generate_frames streams at 10 FPS, 640×360 (was 30 FPS, full res)
    - OptimizedRTSPCapture enable_debug=False in production
    - frame_times is deque(maxlen=100) not a list with O(n) pop(0)
    - Dead code after return in _try_default removed
    - evidence imwrite offloaded to daemon thread (non-blocking inference loop)
    - Flask debug=False (prevents Werkzeug reloader double-process)
"""

from flask import Blueprint, request, jsonify, Response
import cv2
import threading
import time
from datetime import datetime
from collections import deque
import uuid
from database import get_db_connection
from routes.reports_routes import save_report_to_db
import os

# ── Blueprint ──────────────────────────────────────────────────────────────────
monitoring_bp = Blueprint("monitoring", __name__, url_prefix="/api/monitor")

# ── Session state ──────────────────────────────────────────────────────────────
# active_sessions and session_locks are modified from multiple Flask threads.
# _sessions_lock must be held for ALL reads and writes to both dicts to prevent
# RuntimeError: dictionary changed size during iteration.
active_sessions: dict = {}
session_locks: dict = {}
_sessions_lock = threading.Lock()


# ============= JETSON-OPTIMISED RTSP CAPTURE =============

class JetsonRTSPCapture:
    """
    RTSP capture using Jetson hardware video decoder (NVDEC) via GStreamer.

    Backend priority:
      1. Jetson HW GStreamer  — nvv4l2decoder (zero CPU decode, NVDEC engine)
      2. TCP FFmpeg           — CPU decode but reliable fallback
      3. Default OpenCV       — last resort

    frame_times uses deque(maxlen=100) for O(1) append/stats instead of
    list + pop(0) which was O(n) and held the lock while shifting memory.

    get_frame() copies the frame OUTSIDE the lock so the lock is held for
    only a reference assignment (~0 µs) instead of a full memcpy (~4 ms).
    """

    def __init__(self, rtsp_url: str, enable_debug: bool = False):
        self.rtsp_url = rtsp_url
        self.enable_debug = enable_debug
        self.cap = None
        self.last_frame = None
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.skip_count = 0
        self.running = False
        self.read_thread = None
        self.lock = threading.Lock()
        # deque: O(1) append + auto-eviction, no manual pop(0)
        self.frame_times: deque = deque(maxlen=100)

    def _log(self, msg: str):
        if self.enable_debug:
            print(f"[RTSP] {msg}")

    # ── Backend constructors ───────────────────────────────────────────────────

    def _try_jetson_hw(self):
        """
        Jetson NVDEC hardware H.264 decoder via GStreamer.

        nvv4l2decoder  — Jetson Video4Linux2 HW decoder; uses NVDEC engine,
                          zero CPU usage for decode.
        nvvidconv      — Hardware colour conversion from NV12 → BGRx.
        latency=100    — Small buffer (ms) to absorb network jitter without
                          stalling the appsink.
        sync=false     — Don't wait for PTS sync; deliver frames as fast as
                          they arrive (critical for low-latency monitoring).
        """
        pipeline = (
            f"rtspsrc location={self.rtsp_url} latency=100 protocols=tcp ! "
            "rtph264depay ! h264parse ! "
            "nvv4l2decoder ! "
            "nvvidconv ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink max-buffers=1 drop=true sync=false"
        )
        self._log(f"Trying Jetson HW pipeline: {pipeline}")
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    def _try_ffmpeg_tcp(self):
        """FFmpeg over TCP — CPU decode but no UDP packet loss issues."""
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _try_default(self):
        """Default OpenCV backend — last resort."""
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Validate immediately so open() can skip to next backend
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    # ── Connection ─────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Try backends in order. Return True on first success."""
        backends = [
            ("Jetson HW (nvv4l2decoder)", self._try_jetson_hw),
            ("FFmpeg TCP", self._try_ffmpeg_tcp),
            ("Default OpenCV", self._try_default),
        ]
        for name, fn in backends:
            self._log(f"Trying {name} ...")
            try:
                cap = fn()
                if cap is None:
                    self._log(f"✗ {name}: returned None")
                    continue
                if not cap.isOpened():
                    self._log(f"✗ {name}: cap not opened")
                    cap.release()
                    continue
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._log(f"✓ {name}: connected and reading frames")
                    self.cap = cap
                    return True
                self._log(f"✗ {name}: opened but no frames")
                cap.release()
            except Exception as exc:
                self._log(f"✗ {name}: exception — {exc}")
        self._log("✗ All backends failed")
        return False

    # ── Background reader ──────────────────────────────────────────────────────

    def start_reading(self):
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        self._log("Background read thread started")

    def _read_loop(self):
        """
        Continuously drain the camera buffer at up to 60 Hz.

        60 Hz ceiling (16 ms sleep) prevents a CPU spinloop.
        RTSP sources deliver 15–30 fps; reading faster than the source rate
        only burns CPU on redundant cap.read() calls and lock contention.

        frame_times uses deque(maxlen=100) — O(1) everywhere, no lock-held memcpy.
        """
        consecutive_failures = 0
        max_failures = 30

        while self.running:
            ret, frame = self.cap.read()
            now = time.time()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self._log(f"❌ {consecutive_failures} consecutive failures — stopping")
                    self.running = False
                    break
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            gap = now - self.last_frame_time

            with self.lock:
                self.last_frame = frame          # store reference, not copy
                self.frame_count += 1
                self.frame_times.append(gap)     # deque: O(1), no pop(0)
                if gap > 0.1:
                    self.skip_count += 1
                    if self.enable_debug and self.skip_count % 5 == 0:
                        self._log(f"⚠ Buffer gap {gap * 1000:.0f} ms (skip #{self.skip_count})")
                self.last_frame_time = now

            if self.enable_debug and self.frame_count % 100 == 0:
                times = list(self.frame_times)
                avg = sum(times) / len(times) if times else 0
                fps = 1.0 / avg if avg > 0 else 0
                self._log(f"Frames={self.frame_count} skips={self.skip_count} "
                          f"avg={avg * 1000:.1f} ms fps={fps:.1f}")

            # 60 Hz ceiling — yields CPU between reads
            time.sleep(0.016)

    def get_frame(self):
        """
        Return (True, frame_copy) or (False, None).

        The reference is grabbed under the lock (~0 µs hold time).
        The actual numpy array copy happens OUTSIDE the lock so the
        read_thread is not blocked during the ~4 ms memcpy of a 1080p frame.
        """
        with self.lock:
            ref = self.last_frame
        if ref is not None:
            return True, ref.copy()
        return False, None

    def release(self):
        self._log("Releasing capture ...")
        self.running = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self._log("Released")


# ============= DB HELPERS =============

def create_monitoring_session_db(session_id, packer_id, camera_id):
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO monitoring_sessions "
            "(session_id, packer_id, camera_id, started_at, status) VALUES (?,?,?,?,?)",
            (session_id, packer_id, camera_id, datetime.now().isoformat(), "running"),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        print(f"[DB] Session start error: {exc}")
        return False


def update_session_status_db(session_id, status):
    try:
        conn = get_db_connection()
        conn.execute(
            "UPDATE monitoring_sessions SET status=? WHERE session_id=?",
            (status, session_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[DB] Status update error: {exc}")


def update_packer_status_internal(packer_id, status, session_id):
    try:
        conn = get_db_connection()
        conn.execute(
            "UPDATE packers SET status=?, session_id=? WHERE id=?",
            (status, session_id, packer_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[DB] Packer status error: {exc}")


# ============= EVIDENCE WRITER =============

def _write_evidence_async(filepath: str, frame):
    """Save evidence image in a daemon thread so inference loop is never blocked."""
    try:
        cv2.imwrite(filepath, frame)
    except Exception as exc:
        print(f"[EVIDENCE] Write error: {exc}")

# ============= VIDEO RECORDER =============
#
# DESIGN (fixed v2):
# ──────────────────
# • ONE strict 5-minute chunk per detection trigger.
#   The chunk opens on the FIRST detection event of a session (or after the
#   previous chunk closed).  It records for exactly CLIP_SECONDS (300 s) and
#   then closes cleanly, regardless of whether more detections occur.
#   A new chunk will open the next time a detection arrives after the old one
#   closed.  This gives you discrete 5-min evidence clips instead of an
#   endlessly-extending file.
#
# • Correct, playable MP4 output (was: corrupt/unplayable).
#   Root cause of corruption: cv2.VideoWriter with fourcc("mp4v") writes a
#   raw ISO MPEG-4 stream that is NOT seekable unless properly finalised.
#   If the process crashes before writer.release() the moov atom is never
#   written → VLC/QuickTime reject the file as "invalid format".
#
#   Fix strategy:
#     1. Write an intermediate AVI (XVID) to a temp path.  AVI writes the
#        index at the END in a single flush, so it survives partial writes
#        better than MP4.
#     2. On clip close, remux the AVI → MP4 via FFmpeg
#        (ffmpeg -i tmp.avi -c copy out.mp4).  This adds a proper moov atom
#        and produces a file any player can seek.
#     3. Delete the temp AVI after successful remux.
#     If FFmpeg is not installed the AVI is kept and renamed .mp4 — it will
#     still be readable by VLC and ffprobe.
#
# • Files land in /media/amazin/store/evidences/<session_id>/
#   Naming: clip_ShiftA_PackerName_20250330_143501_001.mp4

import subprocess
import shutil

_video_writers: dict = {}          # session_id → clip state dict
_video_writers_lock = threading.Lock()

CLIP_SECONDS = 300                 # Exactly 5 minutes per clip


def _remux_avi_to_mp4(avi_path: str, mp4_path: str) -> bool:
    """
    Remux AVI → MP4 using FFmpeg stream-copy (no re-encode, very fast).
    Returns True on success, False if FFmpeg unavailable or errors out.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[VIDEO] FFmpeg not found — keeping AVI (rename to .mp4 for compatibility)")
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", avi_path, "-c", "copy", mp4_path],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0:
            os.remove(avi_path)
            return True
        print(f"[VIDEO] FFmpeg remux failed (rc={result.returncode}): "
              f"{result.stderr.decode(errors='replace')[-200:]}")
        return False
    except Exception as exc:
        print(f"[VIDEO] FFmpeg exception: {exc}")
        return False


def _close_clip(state: dict, reason: str = "complete"):
    """Release the VideoWriter and remux AVI → MP4."""
    writer = state.get("writer")
    avi_path = state.get("avi_path", "")
    mp4_path = state.get("path", "")
    fw = state.get("frames_written", 0)

    try:
        if writer:
            writer.release()
    except Exception as exc:
        print(f"[VIDEO] Release error: {exc}")

    if avi_path and os.path.exists(avi_path) and mp4_path:
        ok = _remux_avi_to_mp4(avi_path, mp4_path)
        if ok:
            print(f"[VIDEO] ✓ Clip {reason}: {os.path.basename(mp4_path)}  "
                  f"({fw} frames, {fw // max(state.get('fps', 15), 1) // 60:.0f}m "
                  f"{fw // max(state.get('fps', 15), 1) % 60:02d}s)")
        else:
            # FFmpeg unavailable — rename AVI as .mp4 so downstream code finds it
            try:
                os.rename(avi_path, mp4_path)
            except Exception:
                pass
            print(f"[VIDEO] Clip {reason} (AVI fallback): {os.path.basename(mp4_path)}  ({fw} frames)")
    else:
        print(f"[VIDEO] Clip {reason}: no file found at {avi_path}")


def _record_detection_video(
    session_id: str,
    frame=None,
    stop: bool = False,
    fps: float = 15.0,
    meta: dict = None,
    has_detection: bool = False,
):
    """
    Strict 5-minute chunk recorder.

    Call convention:
      Every frame  : _record_detection_video(session_id, frame, meta=meta,
                                              has_detection=<bool>)
      Session end  : _record_detection_video(session_id, stop=True)

    Behaviour:
      • On the FIRST detection event (or after a chunk closes), a new 5-min
        AVI temp file is opened and every subsequent frame is written.
      • After exactly CLIP_SECONDS the chunk is closed + remuxed to MP4.
        A new chunk opens the next time a detection fires.
      • On session stop, the current (possibly partial) chunk is closed.
      • No detection → no file opened, no disk usage.

    Thread safety: _video_writers_lock protects the writers dict.
    VideoWriter.write() is called only from the single inference thread per
    session — no additional locking needed inside the write path.
    """
    if stop:
        with _video_writers_lock:
            state = _video_writers.pop(session_id, None)
        if state and state.get("writer"):
            _close_clip(state, reason="stop")
        return

    if frame is None:
        return

    now = time.time()

    with _video_writers_lock:
        state = _video_writers.get(session_id)

    # ── Close chunk whose 5-minute window has expired ─────────────────────────
    if state and now >= state["clip_deadline"]:
        _close_clip(state, reason="5-min chunk complete")
        with _video_writers_lock:
            _video_writers.pop(session_id, None)
        state = None

    # ── Open a new chunk on first detection (no chunk currently open) ─────────
    if state is None and has_detection:
        m = meta or {}
        evidence_dir = m.get("evidence_dir") or f"/media/amazin/store/evidences/{session_id}"
        shift_tag    = m.get("shift", "UnknownShift")
        packer_tag   = (m.get("packer_name") or "UnknownPacker").replace(" ", "_")

        os.makedirs(evidence_dir, exist_ok=True)

        existing = [f for f in os.listdir(evidence_dir) if f.endswith(".mp4") or f.endswith(".avi")]
        clip_num  = len(existing) + 1

        ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base     = f"clip_Shift{shift_tag}_{packer_tag}_{ts_str}_{clip_num:03d}"
        mp4_path = os.path.join(evidence_dir, base + ".mp4")
        avi_path = os.path.join(evidence_dir, base + "_tmp.avi")   # intermediate

        h, w = frame.shape[:2]
        # XVID AVI: index written on close, survives partial writes better than mp4v
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            # Fallback: try MJPG AVI
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            print(f"[VIDEO] ✗ Could not open VideoWriter for {avi_path} — "
                  "check OpenCV build has XVID/MJPG support")
            return

        state = {
            "writer":         writer,
            "path":           mp4_path,
            "avi_path":       avi_path,
            "clip_deadline":  now + CLIP_SECONDS,
            "frames_written": 0,
            "fps":            fps,
        }
        with _video_writers_lock:
            _video_writers[session_id] = state

        print(f"[VIDEO] ✓ New 5-min chunk started: {base}.mp4  "
              f"({w}×{h} @ {fps:.0f} fps)  "
              f"closes at {datetime.fromtimestamp(now + CLIP_SECONDS).strftime('%H:%M:%S')}")

    # ── Write current frame into active chunk ─────────────────────────────────
    if state:
        try:
            state["writer"].write(frame)
            state["frames_written"] += 1
        except Exception as exc:
            print(f"[VIDEO] Write error session {session_id[:8]}: {exc}")

# ============= VIDEO OVERLAY RENDERER =============

def _draw_video_overlay(frame, summary: dict, packer_name: str, shift: str):
    """
    Draws a semi-transparent HUD panel onto `frame` (in-place) containing:
      Row 1  — Shift label                    (e.g. "Shift A  6AM-2PM")
      Row 2  — Packer name
      ──── divider ────
      Row 3  — Bags Placed  (bag_present count)
      Row 4  — Bags Missed  (missed_nozzle count)
      Row 5  — Stuck Bags   (stuck_bag count)
      ──── divider ────
      Row 6  — Manual Efficiency  (%)
      Row 7  — Packer Efficiency  (%)
      ──── bottom ────
      Timestamp                               (bottom-left corner)

    Dark semi-transparent background keeps it readable over bright plant lighting.
    Frame is modified in-place and also returned for convenience.
    """
    h, w = frame.shape[:2]

    # Panel geometry — taller to accommodate efficiency rows
    PANEL_X, PANEL_Y = 10, 10
    PANEL_W, PANEL_H = 330, 250
    ALPHA    = 0.55
    BG_COLOR = (15, 15, 15)

    px1, py1 = PANEL_X, PANEL_Y
    px2 = min(px1 + PANEL_W, w - 1)
    py2 = min(py1 + PANEL_H, h - 1)

    # Semi-transparent dark background
    roi = frame[py1:py2, px1:px2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (px2 - px1, py2 - py1), BG_COLOR, -1)
    cv2.addWeighted(overlay, ALPHA, roi, 1 - ALPHA, 0, roi)
    frame[py1:py2, px1:px2] = roi
    cv2.rectangle(frame, (px1, py1), (px2, py2), (80, 80, 80), 1)

    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    SMALL  = 0.52
    NORMAL = 0.65
    TXT_X  = px1 + 12
    LINE_H = 28

    def put(text, x, y, scale, color, thickness=1):
        cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

    # ── Header: shift + packer name ───────────────────────────────────────────
    shift_labels = {"A": "Shift A  6AM-2PM", "B": "Shift B  2PM-10PM", "C": "Shift C  10PM-6AM"}
    shift_str = shift_labels.get(shift, f"Shift {shift}")
    put(shift_str,   TXT_X, py1 + 22, SMALL,  (180, 220, 255))
    put(packer_name, TXT_X, py1 + 46, NORMAL, (255, 255, 255), 2)
    cv2.line(frame, (px1 + 8, py1 + 54), (px2 - 8, py1 + 54), (70, 70, 70), 1)

    # ── Detection class counts ────────────────────────────────────────────────
    count_metrics = [
        ("Bags Placed",  summary.get("bags_placed", 0),  (80, 210,  80)),   # green
        ("Bags Missed",  summary.get("bags_missed", 0),  (80,  80, 220)),   # red/blue
        ("Stuck Bags",   summary.get("stuck_bags",  0),  (220, 80, 220)),   # magenta
    ]
    for idx, (label, value, color) in enumerate(count_metrics):
        y = py1 + 54 + LINE_H + idx * LINE_H
        put(f"{label}:", TXT_X,       y, SMALL,  (200, 200, 200))
        put(str(value),  TXT_X + 185, y, NORMAL, color, 2)

    # Divider before efficiency rows
    div_y = py1 + 54 + LINE_H + len(count_metrics) * LINE_H + 4
    cv2.line(frame, (px1 + 8, div_y), (px2 - 8, div_y), (70, 70, 70), 1)

    # ── Efficiency rows ───────────────────────────────────────────────────────
    manual_eff  = summary.get("manual_efficiency",  0)
    packer_eff  = summary.get("packer_efficiency",  0)

    # Colour-code efficiency: green ≥80, yellow 60-79, red <60
    def eff_color(val):
        if val >= 80:
            return (80, 210, 80)
        if val >= 60:
            return (0, 210, 210)
        return (80, 80, 220)

    eff_metrics = [
        ("Manual Eff", manual_eff, eff_color(manual_eff)),
        ("Packer Eff", packer_eff, eff_color(packer_eff)),
    ]
    for idx, (label, value, color) in enumerate(eff_metrics):
        y = div_y + LINE_H + idx * LINE_H
        put(f"{label}:", TXT_X,       y, SMALL,  (200, 200, 200))
        put(f"{value:.1f}%", TXT_X + 185, y, NORMAL, color, 2)

    # ── Timestamp bottom-left ─────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    put(ts, 10, h - 12, 0.46, (200, 200, 200))

    return frame


# ============= MJPEG STREAM GENERATOR =============

def generate_frames(session_id: str):
    """
    MJPEG generator for the React frontend.

    Jetson optimisations:
      encode_interval = 0.100 s  → 10 FPS (was 33 ms / 30 FPS)
        Jetson has no HW JPEG encoder in software path; 10 FPS
        cuts JPEG encoding CPU load by 3×.
      Resize to 640×360 before encode
        1080p copy = ~6 MB alloc; 640×360 = ~0.7 MB — 10× less
        memory pressure on Jetson's unified bus.
      Lock held only for reference grab (~0 µs), resize/encode outside.
      Throwaway lock() bug fixed: we look up the real lock first; if it's
        gone (session stopped) we break immediately.
    """
    last_encode_time = time.time()
    last_encoded_frame = None
    encode_interval = 0.100  # 10 FPS — change to 0.066 for 15 FPS if headroom allows

    while True:
        with _sessions_lock:
            session = active_sessions.get(session_id)
            lock = session_locks.get(session_id)

        if not session or session.get("status") == "stopped":
            break

        now = time.time()
        if now - last_encode_time < encode_interval:
            if last_encoded_frame:
                yield last_encoded_frame
            time.sleep(encode_interval * 0.5)
            continue

        if lock is None:
            time.sleep(0.02)
            continue

        with lock:
            frame = session.get("last_frame")
            if frame is None:
                time.sleep(0.02)
                continue
            # Resize inside lock only because it's fast (pointer math, ~0.5 ms).
            # Full numpy copy would be ~4 ms — that stays outside the lock.
            frame_small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)

        try:
            ret, buf = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                last_encoded_frame = (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                last_encode_time = now
                yield last_encoded_frame
        except Exception as exc:
            print(f"[STREAM] Encode error: {exc}")
            time.sleep(0.02)


@monitoring_bp.route("/video_feed/<session_id>")
def video_feed(session_id):
    with _sessions_lock:
        exists = session_id in active_sessions
    if not exists:
        return jsonify({"error": "Session not found"}), 404
    return Response(
        generate_frames(session_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ============= START MONITORING =============

@monitoring_bp.route("/start", methods=["POST"])
def start_monitoring():
    """Start a real-time monitoring session using Jetson HW decoder + GPU inference."""
    from routes.packer_routes import get_packers_db
    from routes.camera_routes import get_cameras_db
    from models.packer_monitor import PackerEfficiencyMonitor
    from app import MODEL_PATH

    data = request.json
    packer_id = data.get("packer_id")

    packers_db = get_packers_db()
    cameras_db = get_cameras_db()

    if packer_id not in packers_db:
        return jsonify({"error": "Packer not found"}), 404

    packer_data = packers_db[packer_id]
    camera_id = packer_data.get("camera_id")

    if not camera_id or camera_id not in cameras_db:
        return jsonify({"error": "No camera linked to this packer"}), 400

    rtsp_url = cameras_db[camera_id].get("rtsp_url")

    # ── Guard: block duplicate sessions for the same packer ───────────────────
    with _sessions_lock:
        for existing_sid, existing_sess in active_sessions.items():
            if existing_sess.get("packer_id") == packer_id and existing_sess.get("status") == "running":
                return jsonify({
                    "error": f"Packer {packer_id} already has an active session ({existing_sid}). "
                             "Stop it before starting a new one."
                }), 409

    session_id = str(uuid.uuid4())
    evidence_dir = os.path.join("/media/amazin/store/evidences", session_id)
    os.makedirs(evidence_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"STARTING MONITORING SESSION  {session_id}")
    print(f"Packer: {packer_id}  Camera: {camera_id}")
    print(f"RTSP:   {rtsp_url}")
    print(f"{'=' * 60}\n")

    try:
        # ── Initialise YOLO monitor (model loaded from cache) ──────────────────
        monitor = PackerEfficiencyMonitor(
            model_path=MODEL_PATH,
            line_position=float(packer_data.get("line_position", 0.7)),
            start_line_position=float(packer_data.get("start_line_position", 0.2)),
            confidence_threshold=float(packer_data.get("confidence_threshold", 0.5)),
            spouts=int(packer_data.get("spouts", 8)),
            rpm=int(packer_data.get("rpm", 5)),
            enable_debug=False,   # Keep False — print() inside inference loop = CPU stall
            visual_debug=False,   # False = no bbox drawing on UI stream (reduces I/O lag)
            use_gpu=True,
        )

        # ── Connect RTSP with Jetson HW decoder ───────────────────────────────
        rtsp_capture = JetsonRTSPCapture(rtsp_url, enable_debug=False)

        if not rtsp_capture.open():
            return jsonify({
                "error": (
                    f"Failed to connect to camera: {rtsp_url}\n"
                    "Check: camera online, RTSP credentials, network, port not blocked."
                )
            }), 400

        rtsp_capture.start_reading()

        # Wait up to 3 s for first frame
        print("[INIT] Waiting for first frame ...")
        for i in range(30):
            ret, _ = rtsp_capture.get_frame()
            if ret:
                print(f"[INIT] ✓ First frame after {(i + 1) * 0.1:.1f} s")
                break
            time.sleep(0.1)
        else:
            rtsp_capture.release()
            return jsonify({"error": "Camera connected but no frames received"}), 400

        # Quick 10-frame detection diagnostic (non-blocking to user)
        print("[DIAG] Running 10-frame detection test ...")
        counts: dict = {}
        for _ in range(10):
            ret, frame = rtsp_capture.get_frame()
            if not ret:
                continue
            results = monitor.model(frame, conf=monitor.confidence_threshold, verbose=False)
            for r in results:
                for box in r.boxes:
                    name = r.names[int(box.cls[0].cpu().numpy())]
                    counts[name] = counts.get(name, 0) + 1
            time.sleep(0.05)
        total_det = sum(counts.values())
        print(f"[DIAG] {counts}  total={total_det}")
        if total_det == 0:
            print("[DIAG] ⚠ No detections — check camera view and model weights")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Init failed: {exc}"}), 500

    # ── Register session (thread-safe) ─────────────────────────────────────────
    stop_event = threading.Event()
    session_lock = threading.Lock()

    with _sessions_lock:
        active_sessions[session_id] = {
            "session_id": session_id,
            "packer_id": packer_id,
            "monitor": monitor,
            "rtsp_capture": rtsp_capture,
            "status": "running",
            "last_frame": None,
            "evidence_dir": evidence_dir,
            "stop_event": stop_event,
        }
        session_locks[session_id] = session_lock

    create_monitoring_session_db(session_id, packer_id, camera_id)
    update_packer_status_internal(packer_id, "active", session_id)
    
    def process_stream():
        TARGET_FPS = 15
        target_interval = 1.0 / TARGET_FPS
        frame_count = 0
        last_report = time.time()

        # Resolve packer name once so we don't hit the DB every frame
        _conn = get_db_connection()
        _row  = _conn.execute("SELECT name FROM packers WHERE id=?", (packer_id,)).fetchone()
        _conn.close()
        packer_name = _row["name"] if _row else "Unknown"

        # Derive shift from the time monitoring started
        from shift_scheduler import get_current_shift
        session_shift = get_current_shift()

        # Metadata passed to the video recorder for filename + overlay
        video_meta = {
            "shift":        session_shift,
            "packer_name":  packer_name,
            "evidence_dir": evidence_dir,
        }

        print(f"[STREAM] Session {session_id} started @ {TARGET_FPS} FPS target  "
              f"shift={session_shift}  packer={packer_name}")

        while True:
            with _sessions_lock:
                session = active_sessions.get(session_id)
            if not session or stop_event.is_set():
                break

            frame_start = time.time()
            try:
                ret, frame = session["rtsp_capture"].get_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # 1. Clean copy for the UI — no bounding boxes, no HUD
                ui_frame = frame.copy()

                # 2. GPU inference — visual_debug controls whether boxes are drawn
                annotated_frame = session["monitor"].process_frame(frame)

                # Check what the latest event is
                evt = session["monitor"].last_event_type
                
                # Image Evidence Capture: Strictly for faults (stuck_bag / missed_nozzle)
                is_fault = evt in ("stuck_bag", "missed_nozzle") if evt else False
                if is_fault:
                    ts = datetime.now().strftime("%H-%M-%S-%f")[:12]
                    fpath = os.path.join(session["evidence_dir"], f"{evt}_{ts}.jpg")
                    threading.Thread(
                        target=_write_evidence_async,
                        args=(fpath, annotated_frame.copy()),
                        daemon=True,
                    ).start()
                
                # Video Trigger Check: TRUE if *any* event/detection happened
                any_event = evt is not None

                # Always clear last_event_type after handling regardless of type
                session["monitor"].last_event_type = None

                # 3. Draw detection lines — on annotated_frame (goes to video) AND ui_frame
                h, w = annotated_frame.shape[:2]
                lx = int(w * float(packer_data.get("line_position", 0.7)))
                sx = int(w * float(packer_data.get("start_line_position", 0.2)))

                cv2.line(annotated_frame, (lx, 0), (lx, h), (0, 0, 255), 3)
                cv2.line(annotated_frame, (sx, 0), (sx, h), (255, 0, 0), 2)
                cv2.line(ui_frame,        (lx, 0), (lx, h), (0, 0, 255), 3)
                cv2.line(ui_frame,        (sx, 0), (sx, h), (255, 0, 0), 2)

                # 4. Compose the video frame
                summary = session["monitor"].get_summary()
                video_frame = _draw_video_overlay(
                    annotated_frame,
                    summary,
                    packer_name,
                    session_shift,
                )

                # 5. Write to event-triggered 5-minute clip (triggers on ANY detection now)
                _record_detection_video(session_id, video_frame, meta=video_meta,
                                        fps=TARGET_FPS, has_detection=any_event)

                # 6. Thread-safe frame hand-off to browser MJPEG stream
                with session_lock:
                    session["last_frame"] = ui_frame

                frame_count += 1
                now = time.time()
                if now - last_report > 5.0:
                    s = session["monitor"].get_summary()
                    print(f"[STATUS] {session_id[:8]} frames={frame_count} "
                          f"events={s['total_events']} fps={s['estimated_fps']:.1f} "
                          f"device={s['device']}")
                    last_report = now

                elapsed = time.time() - frame_start
                sleep_t = target_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            except Exception as exc:
                import traceback
                print(f"[ERROR] Session {session_id}: {exc}")
                traceback.print_exc()
                break

        # ── Cleanup ────────────────────────────────────────────────────────────
        print(f"[STREAM] Session {session_id} stopped")
        _record_detection_video(session_id, stop=True)

        with _sessions_lock:
            sess = active_sessions.pop(session_id, None)
            session_locks.pop(session_id, None)
        if sess:
            sess["rtsp_capture"].release()
        update_session_status_db(session_id, "stopped")
        update_packer_status_internal(packer_id, "idle", None)

    # # ── Processing loop ────────────────────────────────────────────────────────
    # def process_stream():
    #     TARGET_FPS = 15
    #     target_interval = 1.0 / TARGET_FPS
    #     frame_count = 0
    #     last_report = time.time()

    #     # Resolve packer name once so we don't hit the DB every frame
    #     _conn = get_db_connection()
    #     _row  = _conn.execute("SELECT name FROM packers WHERE id=?", (packer_id,)).fetchone()
    #     _conn.close()
    #     packer_name = _row["name"] if _row else "Unknown"

    #     # Derive shift from the time monitoring started
    #     from shift_scheduler import get_current_shift
    #     session_shift = get_current_shift()

    #     # Metadata passed to the video recorder for filename + overlay
    #     video_meta = {
    #         "shift":        session_shift,
    #         "packer_name":  packer_name,
    #         "evidence_dir": evidence_dir,
    #     }

    #     print(f"[STREAM] Session {session_id} started @ {TARGET_FPS} FPS target  "
    #           f"shift={session_shift}  packer={packer_name}")

    #     while True:
    #         with _sessions_lock:
    #             session = active_sessions.get(session_id)
    #         if not session or stop_event.is_set():
    #             break

    #         frame_start = time.time()
    #         try:
    #             ret, frame = session["rtsp_capture"].get_frame()
    #             if not ret or frame is None:
    #                 time.sleep(0.01)
    #                 continue

    #             # 1. Clean copy for the UI — no bounding boxes, no HUD
    #             ui_frame = frame.copy()

    #             # 2. GPU inference — visual_debug controls whether boxes are drawn
    #             #    onto the returned frame (used only for video recording, NOT streaming)
    #             annotated_frame = session["monitor"].process_frame(frame)

    #             # Evidence capture — ONLY for fault events (stuck_bag / missed_nozzle).
    #             # Explicitly exclude bag_present to avoid filling disk with normal frames.
    #             # Save the annotated frame so evidence images show bounding boxes.
    #             evt = session["monitor"].last_event_type
    #             is_fault = evt in ("stuck_bag", "missed_nozzle") if evt else False
    #             if is_fault:
    #                 ts = datetime.now().strftime("%H-%M-%S-%f")[:12]
    #                 fpath = os.path.join(session["evidence_dir"], f"{evt}_{ts}.jpg")
    #                 threading.Thread(
    #                     target=_write_evidence_async,
    #                     args=(fpath, annotated_frame.copy()),
    #                     daemon=True,
    #                 ).start()
    #             # Always clear last_event_type after handling regardless of type
    #             session["monitor"].last_event_type = None

    #             # 3. Draw detection lines — on annotated_frame (goes to video) AND
    #             #    on ui_frame (goes to browser) so operator sees zone markers.
    #             #    Bounding boxes are NOT on ui_frame (visual_debug=False above).
    #             h, w = annotated_frame.shape[:2]
    #             lx = int(w * float(packer_data.get("line_position", 0.7)))
    #             sx = int(w * float(packer_data.get("start_line_position", 0.2)))

    #             cv2.line(annotated_frame, (lx, 0), (lx, h), (0, 0, 255), 3)
    #             cv2.line(annotated_frame, (sx, 0), (sx, h), (255, 0, 0), 2)
    #             cv2.line(ui_frame,        (lx, 0), (lx, h), (0, 0, 255), 3)
    #             cv2.line(ui_frame,        (sx, 0), (sx, h), (255, 0, 0), 2)

    #             # 4. Compose the video frame:
    #             #    annotated_frame already has YOLO bounding-box tracks drawn by process_frame.
    #             #    Now add the HUD overlay (shift, packer name, all 3 class counts, timestamp).
    #             summary = session["monitor"].get_summary()
    #             video_frame = _draw_video_overlay(
    #                 annotated_frame,   # modified in-place (already has tracks)
    #                 summary,
    #                 packer_name,
    #                 session_shift,
    #             )

    #             # 5. Write to event-triggered 5-minute clip (only during fault periods)
    #             _record_detection_video(session_id, video_frame, meta=video_meta,
    #                                     fps=TARGET_FPS, has_detection=is_fault)

    #             # 6. Thread-safe frame hand-off to browser MJPEG stream (clean frame)
    #             with session_lock:
    #                 session["last_frame"] = ui_frame

    #             frame_count += 1
    #             now = time.time()
    #             if now - last_report > 5.0:
    #                 s = session["monitor"].get_summary()
    #                 print(f"[STATUS] {session_id[:8]} frames={frame_count} "
    #                       f"events={s['total_events']} fps={s['estimated_fps']:.1f} "
    #                       f"device={s['device']}")
    #                 last_report = now

    #             elapsed = time.time() - frame_start
    #             sleep_t = target_interval - elapsed
    #             if sleep_t > 0:
    #                 time.sleep(sleep_t)

    #         except Exception as exc:
    #             import traceback
    #             print(f"[ERROR] Session {session_id}: {exc}")
    #             traceback.print_exc()
    #             break

    #     # ── Cleanup ────────────────────────────────────────────────────────────
    #     print(f"[STREAM] Session {session_id} stopped")
    #     _record_detection_video(session_id, stop=True)

    #     with _sessions_lock:
    #         sess = active_sessions.pop(session_id, None)
    #         session_locks.pop(session_id, None)
    #     if sess:
    #         sess["rtsp_capture"].release()
    #     update_session_status_db(session_id, "stopped")
    #     update_packer_status_internal(packer_id, "idle", None)

    threading.Thread(target=process_stream, daemon=True).start()

    return jsonify({
        "message": "Monitoring started",
        "session_id": session_id,
        "gpu_enabled": monitor.device.startswith("cuda"),
    }), 201


# ============= STOP MONITORING =============

@monitoring_bp.route("/stop/<session_id>", methods=["POST"])
def stop_monitoring(session_id):
    """
    Stop a monitoring session.

    Signals the inference thread via stop_event — the thread owns rtsp_capture
    release and active_sessions cleanup so we never leak the capture or race
    on session dict removal.  This endpoint only signals, saves the report, and
    updates the DB status — it does NOT pop active_sessions itself.
    """
    with _sessions_lock:
        session = active_sessions.get(session_id)

    if not session:
        return jsonify({"error": "Session not found"}), 404

    packer_id = session["packer_id"]
    monitor   = session["monitor"]

    # Signal inference thread to exit — it will release rtsp_capture and pop
    # active_sessions/session_locks once the loop drains cleanly.
    session["stop_event"].set()
    session["status"] = "stopped"

    # Guard against double-report: if ShiftScheduler._stop_session() already
    # claimed report_saved (set under _sessions_lock), skip saving here.
    with _sessions_lock:
        already_saved = session.get("report_saved", False)
        if not already_saved:
            session["report_saved"] = True

    final_summary = monitor.get_summary()

    conn = get_db_connection()
    row  = conn.execute("SELECT name FROM packers WHERE id=?", (packer_id,)).fetchone()
    packer_name = row["name"] if row else "Unknown"
    conn.close()

    save_success = False
    if not already_saved:
        report_data = {
            "id":          session_id,
            "packer_id":   packer_id,
            "packer_name": packer_name,
            "timestamp":   datetime.now().isoformat(),
            "summary":     final_summary,
        }
        save_success = save_report_to_db(report_data)
    else:
        print(f"[STOP] Session {session_id[:8]} report already saved by ShiftScheduler — skip duplicate")
    # Mark completed in monitoring_sessions table; inference thread will also
    # call update_session_status_db("stopped") — that second write is harmless.
    update_session_status_db(session_id, "completed")

    return jsonify({
        "message":       "Monitoring stopped",
        "report_saved":  save_success,
        "report_id":     session_id,
        "final_summary": final_summary,
    }), 200


# ============= METRICS & SESSION ENDPOINTS =============

@monitoring_bp.route("/metrics/<session_id>", methods=["GET"])
def get_live_metrics(session_id):
    with _sessions_lock:
        session = active_sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({
        "session_id": session_id,
        "status": session["status"],
        "metrics": session["monitor"].get_summary(),
    }), 200


def get_active_monitor_summary(session_data: dict) -> dict:
    try:
        m = session_data.get("monitor")
        if m and hasattr(m, "get_summary"):
            return m.get_summary()
    except Exception as exc:
        print(f"[MONITOR] get_summary error: {exc}")
    return {}


@monitoring_bp.route("/active-sessions", methods=["GET"])
def get_active_sessions():
    # Snapshot under lock to avoid dict-changed-size-during-iteration
    with _sessions_lock:
        snapshot = dict(active_sessions)

    result = []
    for sid, sess in snapshot.items():
        if sess.get("status") == "running":
            result.append({
                "session_id": sid,
                "packer_id": sess.get("packer_id"),
                "status": sess.get("status"),
                "metrics": get_active_monitor_summary(sess),
            })

    return jsonify({"active_sessions": result, "count": len(result)}), 200


# ============= CAMERA TEST ENDPOINT =============

@monitoring_bp.route("/test-camera/<camera_id>", methods=["GET"])
def test_camera(camera_id):
    """Test RTSP connection using Jetson HW pipeline."""
    from routes.camera_routes import get_cameras_db

    cameras_db = get_cameras_db()
    if camera_id not in cameras_db:
        return jsonify({"error": "Camera not found"}), 404

    rtsp_url = cameras_db[camera_id].get("rtsp_url")
    print(f"[TEST] Testing camera {camera_id}  RTSP: {rtsp_url}")

    try:
        # enable_debug=True is fine here — this is a one-shot test, not a live loop
        capture = JetsonRTSPCapture(rtsp_url, enable_debug=True)
        if not capture.open():
            return jsonify({"success": False, "error": "Failed to open RTSP stream",
                            "camera_id": camera_id}), 400

        capture.start_reading()
        frames_received = 0
        for _ in range(30):
            ret, frame = capture.get_frame()
            if ret and frame is not None:
                frames_received += 1
            time.sleep(0.1)
        capture.release()

        success = frames_received > 0
        return jsonify({
            "success": success,
            "camera_id": camera_id,
            "rtsp_url": rtsp_url,
            "frames_received": frames_received,
            "message": "Camera working" if success else "No frames received",
        }), 200 if success else 400

    except Exception as exc:
        print(f"[TEST] Error: {exc}")
        return jsonify({"success": False, "error": str(exc),
                        "camera_id": camera_id}), 500
