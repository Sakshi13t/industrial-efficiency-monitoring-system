"""
Shift Scheduler — Auto Monitoring for PackerVision AI
======================================================

Shift definitions:
  A  06:00 → 13:59   (6 AM  – 2 PM)
  B  14:00 → 21:59   (2 PM  – 10 PM)
  C  22:00 → 05:59   (10 PM – 6 AM, wraps midnight)

How it works
------------
A single background daemon thread (ShiftScheduler) wakes every
TICK_SECONDS (default 30 s) and compares the current wall-clock
time against the three shift windows.

On a shift boundary:
  1. The outgoing session (if any) is stopped → report saved to DB
     with shift label, packer_id, packer_name.
  2. All packers that have a linked, reachable camera are started
     in a new session automatically.

The scheduler stores its own per-packer session_id mapping so it
can stop exactly the sessions it started (manual sessions are left
alone — they carry shift=None in the DB).

Thread safety
-------------
_sched_lock protects _current_shift and _packer_sessions.
The scheduler never touches active_sessions directly — it calls
the same internal helpers used by the HTTP endpoints, which carry
their own _sessions_lock. This keeps locking hierarchies clean.

Usage (in app.py)
-----------------
    from shift_scheduler import ShiftScheduler
    scheduler = ShiftScheduler(app)
    scheduler.start()

The scheduler is intentionally idempotent: calling start() twice
is a no-op.
"""

import threading
import time
import uuid
import os
from datetime import datetime
from typing import Optional
from database import get_db_connection
from report_mailer import shift_report_email

# ── Shift definitions ─────────────────────────────────────────────────────────

SHIFTS = {
    "A": {"label": "Shift A",  "start_hour": 6,  "end_hour": 14},   # 06:00–13:59
    "B": {"label": "Shift B",  "start_hour": 14, "end_hour": 22},   # 14:00–21:59
    "C": {"label": "Shift C",  "start_hour": 22, "end_hour": 6},    # 22:00–05:59 (wraps)
}

TICK_SECONDS = 30   # How often the scheduler wakes to check the clock


def get_current_shift() -> str:
    """Return 'A', 'B', or 'C' based on current local time."""
    hour = datetime.now().hour
    if 6 <= hour < 14:
        return "A"
    elif 14 <= hour < 22:
        return "B"
    else:
        return "C"


def get_shift_for_time(dt: datetime) -> str:
    """Return shift label for an arbitrary datetime."""
    hour = dt.hour
    if 6 <= hour < 14:
        return "A"
    elif 14 <= hour < 22:
        return "B"
    else:
        return "C"


# ── Scheduler ─────────────────────────────────────────────────────────────────

class ShiftScheduler:
    """
    Auto-starts and auto-stops monitoring sessions at shift boundaries.

    Attributes
    ----------
    _current_shift  : Optional[str]   — shift letter currently being monitored
    _packer_sessions: dict          — packer_id → session_id for auto sessions
    _thread         : Thread        — background daemon
    _running        : bool          — set False to stop
    _sched_lock     : Lock          — protects _current_shift & _packer_sessions
    _app            : Flask app     — needed to push app context inside thread
    """

    def __init__(self, app=None):
        self._app = app
        self._current_shift: Optional[str] = None
        self._packer_sessions: dict = {}   # packer_id → session_id
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._sched_lock = threading.Lock()
        self._started = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        if self._started:
            return
        self._started = True
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="ShiftScheduler",
        )
        self._thread.start()
        print("[SHIFT] Scheduler started — auto-monitoring enabled")

    def stop(self):
        self._running = False

    def get_status(self) -> dict:
        with self._sched_lock:
            return {
                "current_shift": self._current_shift,
                "shift_label": SHIFTS.get(self._current_shift, {}).get("label") if self._current_shift else None,
                "active_packer_sessions": dict(self._packer_sessions),
                "next_shift_change": self._minutes_to_next_boundary(),
            }

    # ── Core loop ──────────────────────────────────────────────────────────────

    def _run_loop(self):
        """
        Wakes every TICK_SECONDS.
        On first run (no current shift) it immediately starts the current shift.
        On subsequent ticks it checks whether the shift has changed.
        """
        # Push Flask app context so DB helpers work inside this thread
        if self._app:
            ctx = self._app.app_context()
            ctx.push()

        print(f"[SHIFT] Scheduler loop running — tick every {TICK_SECONDS}s")

        while self._running:
            try:
                new_shift = get_current_shift()

                with self._sched_lock:
                    old_shift = self._current_shift

                if new_shift != old_shift:
                    print(f"[SHIFT] ── Boundary hit: {old_shift} → {new_shift} ──")
                    self._handle_shift_change(old_shift, new_shift)

            except Exception as exc:
                import traceback
                print(f"[SHIFT] Loop error: {exc}")
                traceback.print_exc()

            time.sleep(TICK_SECONDS)

    # ── Shift transition ───────────────────────────────────────────────────────

    def _handle_shift_change(self, old_shift: Optional[str], new_shift: str):
        """Stop all auto-sessions from old shift, start new ones."""

        # 1. Stop old sessions
        with self._sched_lock:
            sessions_to_stop = dict(self._packer_sessions)
            self._packer_sessions = {}

        for packer_id, session_id in sessions_to_stop.items():
            self._stop_session(packer_id, session_id, old_shift)

        # 2. Update shift marker BEFORE starting new sessions
        with self._sched_lock:
            self._current_shift = new_shift

        # 3. Start new sessions
        started = self._start_all_packers(new_shift)
        print(f"[SHIFT] Shift {new_shift} — started {started} packer session(s)")

    # ── Start helpers ──────────────────────────────────────────────────────────

    def _start_all_packers(self, shift: str) -> int:
        """
        Start monitoring for every packer that has a linked, online camera.
        Returns number of sessions successfully started.
        """
        from routes.packer_routes import get_packers_db
        from routes.camera_routes import get_cameras_db

        packers_db = get_packers_db()
        cameras_db = get_cameras_db()

        from routes.monitoring_routes import active_sessions, _sessions_lock

        started = 0
        for packer_id, packer_data in packers_db.items():
            camera_id = packer_data.get("camera_id")
            if not camera_id or camera_id not in cameras_db:
                print(f"[SHIFT] Packer {packer_id}: no camera linked — skipping")
                continue

            camera = cameras_db[camera_id]
            if camera.get("status") == "offline":
                print(f"[SHIFT] Packer {packer_id}: camera offline — skipping")
                continue

            # Guard: atomically check AND mark packer as reserved before starting.
            # Checking then starting in two separate steps creates a race where
            # two callers (scheduler tick + override restart) both see no active
            # session and both proceed to start one — causing double sessions.
            with _sessions_lock:
                already_running = any(
                    s.get("packer_id") == packer_id and s.get("status") == "running"
                    for s in active_sessions.values()
                )
                if not already_running:
                    # Reserve the slot immediately under the same lock so no
                    # concurrent caller can sneak in before _start_session registers.
                    _reservation_key = f"__reserving_{packer_id}__"
                    active_sessions[_reservation_key] = {
                        "packer_id": packer_id,
                        "status": "running",
                        "_reservation": True,
                    }

            if already_running:
                print(f"[SHIFT] Packer {packer_id}: session already active — skipping duplicate")
                continue

            session_id = self._start_session(packer_id, packer_data, camera, shift)

            # Always remove reservation placeholder regardless of outcome
            with _sessions_lock:
                active_sessions.pop(_reservation_key, None)

            if session_id:
                with self._sched_lock:
                    self._packer_sessions[packer_id] = session_id
                started += 1

        return started

    def _start_session(self, packer_id: str, packer_data: dict,
                       camera: dict, shift: str) -> Optional[str]:
        """
        Spin up a monitoring session for one packer.
        Returns session_id on success, None on failure.

        Mirrors the logic in monitoring_routes.start_monitoring() but
        called programmatically (no HTTP request object needed).
        """
        try:
            from routes.monitoring_routes import (
                active_sessions, session_locks, _sessions_lock,
                JetsonRTSPCapture,
                create_monitoring_session_db, update_session_status_db,
                update_packer_status_internal, _write_evidence_async, _record_detection_video
            )
            from models.packer_monitor import PackerEfficiencyMonitor
            from app import MODEL_PATH

            rtsp_url = camera.get("rtsp_url")
            session_id = str(uuid.uuid4())
            evidence_dir = os.path.join("/media/amazin/store/evidences", session_id)
            os.makedirs(evidence_dir, exist_ok=True)

            print(f"[SHIFT] Starting session {session_id[:8]}… "
                  f"packer={packer_id}  shift={shift}  rtsp={rtsp_url}")

            # Build YOLO monitor (singleton cache — no extra VRAM)
            # visual_debug=True so process_frame() draws bounding-box tracks onto
            # the returned frame, which are then written into the recorded video.
            monitor = PackerEfficiencyMonitor(
                model_path=MODEL_PATH,
                line_position=float(packer_data.get("line_position", 0.7)),
                start_line_position=float(packer_data.get("start_line_position", 0.2)),
                confidence_threshold=float(packer_data.get("confidence_threshold", 0.5)),
                spouts=int(packer_data.get("spouts", 8)),
                rpm=int(packer_data.get("rpm", 5)),
                enable_debug=False,
                visual_debug=False,   # False = no bbox drawing on UI stream (reduces I/O lag)
                use_gpu=True,
            )

            # Open RTSP stream
            rtsp_capture = JetsonRTSPCapture(rtsp_url, enable_debug=False)
            if not rtsp_capture.open():
                print(f"[SHIFT] Could not open RTSP for packer {packer_id}")
                return None
            rtsp_capture.start_reading()
            time.sleep(1.0)   # Let buffer fill

            stop_event = threading.Event()
            session_lock = threading.Lock()

            session_data = {
                "session_id": session_id,
                "packer_id": packer_id,
                "shift": shift,
                "status": "running",
                "monitor": monitor,
                "rtsp_capture": rtsp_capture,
                "stop_event": stop_event,
                "last_frame": None,
                "evidence_dir": evidence_dir,
                "started_at": datetime.now().isoformat(),
                "auto": True,   # Mark as auto-started so UI can distinguish
            }

            with _sessions_lock:
                active_sessions[session_id] = session_data
                session_locks[session_id] = session_lock

            create_monitoring_session_db(session_id, packer_id, camera.get("id"))
            update_packer_status_internal(packer_id, "active", session_id)

            # Start inference thread
            TARGET_FPS = 15
            target_interval = 1.0 / TARGET_FPS
            
            def process_stream():
                import cv2 as _cv2
                import traceback as _tb

                packer_cfg = packer_data

                # Resolve packer name once
                _conn = get_db_connection()
                _row  = _conn.execute("SELECT name FROM packers WHERE id=?", (packer_id,)).fetchone()
                _conn.close()
                _packer_name = _row["name"] if _row else packer_data.get("name", "Unknown")

                # Import overlay helper from monitoring_routes
                from routes.monitoring_routes import _draw_video_overlay

                video_meta = {
                    "shift":        shift,
                    "packer_name":  _packer_name,
                    "evidence_dir": evidence_dir,
                }

                frame_count = 0
                last_report = time.time()

                while not stop_event.is_set():
                    frame_start = time.time()
                    try:
                        ret, frame = rtsp_capture.get_frame()
                        if not ret or frame is None:
                            time.sleep(0.01)
                            continue

                        # Clean copy for UI stream BEFORE inference mutates the frame
                        ui_frame = frame.copy()

                        # GPU inference
                        processed = monitor.process_frame(frame)

                        # Check what the latest event is
                        evt = monitor.last_event_type
                        
                        # Image Evidence Capture: Strictly for faults
                        is_fault = evt in ("stuck_bag", "missed_nozzle") if evt else False
                        if is_fault:
                            ts = datetime.now().strftime("%H-%M-%S-%f")[:12]
                            fpath = os.path.join(evidence_dir, f"{evt}_{ts}.jpg")
                            threading.Thread(
                                target=_write_evidence_async,
                                args=(fpath, processed.copy()),
                                daemon=True,
                            ).start()
                            
                        # Video Trigger Check: TRUE if *any* event/detection happened
                        any_event = evt is not None

                        # Always clear regardless of event type
                        monitor.last_event_type = None

                        h, w = processed.shape[:2]
                        lx = int(w * float(packer_cfg.get("line_position", 0.7)))
                        sx = int(w * float(packer_cfg.get("start_line_position", 0.2)))
                        _cv2.line(processed, (lx, 0), (lx, h), (0, 0, 255), 3)
                        _cv2.line(processed, (sx, 0), (sx, h), (255, 0, 0), 2)

                        # Also draw lines on ui_frame for operator reference (no HUD/boxes)
                        _cv2.line(ui_frame, (lx, 0), (lx, h), (0, 0, 255), 3)
                        _cv2.line(ui_frame, (sx, 0), (sx, h), (255, 0, 0), 2)

                        # Add HUD overlay only to the video recording frame
                        summary = monitor.get_summary()
                        video_frame = _draw_video_overlay(
                            processed, summary, _packer_name, shift
                        )

                        # Write to event-triggered 5-minute clip (triggers on ANY detection now)
                        _record_detection_video(session_id, video_frame,
                                                meta=video_meta, fps=TARGET_FPS,
                                                has_detection=any_event)

                        # Hand clean frame (lines only, no HUD panel, no bboxes) to browser
                        with session_lock:
                            session_data["last_frame"] = ui_frame

                        frame_count += 1
                        now = time.time()
                        if now - last_report > 5.0:
                            s = monitor.get_summary()
                            print(f"[AUTO] {session_id[:8]} shift={shift} "
                                  f"frames={frame_count} events={s['total_events']} "
                                  f"fps={s.get('estimated_fps', 0):.1f}")
                            last_report = now

                        elapsed = time.time() - frame_start
                        sleep_t = target_interval - elapsed
                        if sleep_t > 0:
                            time.sleep(sleep_t)

                    except Exception as exc:
                        print(f"[AUTO] Session {session_id}: {exc}")
                        _tb.print_exc()
                        break

                # ── Cleanup after stop ─────────────────────────────────────────
                print(f"[AUTO] Session {session_id[:8]} stream stopped")
                _record_detection_video(session_id, stop=True)
                rtsp_capture.release()
                update_session_status_db(session_id, "stopped")
                update_packer_status_internal(packer_id, "idle", None)

                with _sessions_lock:
                    active_sessions.pop(session_id, None)
                    session_locks.pop(session_id, None)

            # def process_stream():
            #     import cv2 as _cv2
            #     import traceback as _tb

            #     packer_cfg = packer_data

            #     # Resolve packer name once
            #     _conn = get_db_connection()
            #     _row  = _conn.execute("SELECT name FROM packers WHERE id=?", (packer_id,)).fetchone()
            #     _conn.close()
            #     _packer_name = _row["name"] if _row else packer_data.get("name", "Unknown")

            #     # Import overlay helper from monitoring_routes
            #     from routes.monitoring_routes import _draw_video_overlay

            #     video_meta = {
            #         "shift":        shift,
            #         "packer_name":  _packer_name,
            #         "evidence_dir": evidence_dir,
            #     }

            #     frame_count = 0
            #     last_report = time.time()

            #     while not stop_event.is_set():
            #         frame_start = time.time()
            #         try:
            #             ret, frame = rtsp_capture.get_frame()
            #             if not ret or frame is None:
            #                 time.sleep(0.01)
            #                 continue

            #             # Clean copy for UI stream BEFORE inference mutates the frame
            #             ui_frame = frame.copy()

            #             # GPU inference (visual_debug=False so no bbox drawn on ui_frame)
            #             processed = monitor.process_frame(frame)

            #             # Evidence capture — ONLY fault events (stuck_bag / missed_nozzle).
            #             # Explicitly exclude bag_present. Save processed (annotated) frame
            #             # so evidence images show bounding boxes for context.
            #             evt = monitor.last_event_type
            #             is_fault = evt in ("stuck_bag", "missed_nozzle") if evt else False
            #             if is_fault:
            #                 ts = datetime.now().strftime("%H-%M-%S-%f")[:12]
            #                 fpath = os.path.join(evidence_dir, f"{evt}_{ts}.jpg")
            #                 threading.Thread(
            #                     target=_write_evidence_async,
            #                     args=(fpath, processed.copy()),
            #                     daemon=True,
            #                 ).start()
            #             # Always clear regardless of event type
            #             monitor.last_event_type = None

            #             h, w = processed.shape[:2]
            #             lx = int(w * float(packer_cfg.get("line_position", 0.7)))
            #             sx = int(w * float(packer_cfg.get("start_line_position", 0.2)))
            #             _cv2.line(processed, (lx, 0), (lx, h), (0, 0, 255), 3)
            #             _cv2.line(processed, (sx, 0), (sx, h), (255, 0, 0), 2)

            #             # Also draw lines on ui_frame for operator reference (no HUD/boxes)
            #             _cv2.line(ui_frame, (lx, 0), (lx, h), (0, 0, 255), 3)
            #             _cv2.line(ui_frame, (sx, 0), (sx, h), (255, 0, 0), 2)

            #             # Add HUD overlay only to the video recording frame
            #             summary = monitor.get_summary()
            #             video_frame = _draw_video_overlay(
            #                 processed, summary, _packer_name, shift
            #             )

            #             # Write to event-triggered 5-minute clip (only during fault periods)
            #             _record_detection_video(session_id, video_frame,
            #                                     meta=video_meta, fps=TARGET_FPS,
            #                                     has_detection=is_fault)

            #             # Hand clean frame (lines only, no HUD panel, no bboxes) to browser
            #             with session_lock:
            #                 session_data["last_frame"] = ui_frame

            #             frame_count += 1
            #             now = time.time()
            #             if now - last_report > 5.0:
            #                 s = monitor.get_summary()
            #                 print(f"[AUTO] {session_id[:8]} shift={shift} "
            #                       f"frames={frame_count} events={s['total_events']} "
            #                       f"fps={s.get('estimated_fps', 0):.1f}")
            #                 last_report = now

            #             elapsed = time.time() - frame_start
            #             sleep_t = target_interval - elapsed
            #             if sleep_t > 0:
            #                 time.sleep(sleep_t)

            #         except Exception as exc:
            #             print(f"[AUTO] Session {session_id}: {exc}")
            #             _tb.print_exc()
            #             break

            #     # ── Cleanup after stop ─────────────────────────────────────────
            #     print(f"[AUTO] Session {session_id[:8]} stream stopped")
            #     _record_detection_video(session_id, stop=True)
            #     rtsp_capture.release()
            #     update_session_status_db(session_id, "stopped")
            #     update_packer_status_internal(packer_id, "idle", None)

            #     with _sessions_lock:
            #         active_sessions.pop(session_id, None)
            #         session_locks.pop(session_id, None)

            threading.Thread(
                target=process_stream,
                daemon=True,
                name=f"auto-stream-{session_id[:8]}",
            ).start()

            print(f"[SHIFT] ✓ Session {session_id[:8]} started for packer {packer_id}")
            return session_id

        except Exception as exc:
            import traceback
            print(f"[SHIFT] Failed to start session for packer {packer_id}: {exc}")
            traceback.print_exc()
            return None

    # ── Stop helpers ───────────────────────────────────────────────────────────

    def _stop_session(self, packer_id: str, session_id: str, shift: Optional[str]):
        """
        Stop one auto-monitoring session and save its report with shift label.

        Uses a 'report_saved' flag (set atomically under _sessions_lock) to
        prevent the double-report bug where both _stop_session() and
        stop_monitoring() HTTP endpoint both call save_report_to_db() for the
        same session_id when a shift boundary coincides with a manual stop.
        """
        try:
            from routes.monitoring_routes import active_sessions, _sessions_lock
            from routes.reports_routes import save_report_to_db

            with _sessions_lock:
                session = active_sessions.get(session_id)
                if not session:
                    print(f"[SHIFT] Session {session_id[:8]} already gone — skip stop")
                    return
                # Claim report-saving rights atomically — if already True, another
                # path (stop_monitoring HTTP endpoint) already saved the report.
                if session.get("report_saved"):
                    print(f"[SHIFT] Session {session_id[:8]} report already saved — skip duplicate")
                    session["stop_event"].set()
                    return
                session["report_saved"] = True

            session["stop_event"].set()
            session["status"] = "stopped"

            monitor = session["monitor"]
            final_summary = monitor.get_summary()

            conn = get_db_connection()
            row = conn.execute(
                "SELECT name FROM packers WHERE id=?", (packer_id,)
            ).fetchone()
            packer_name = row["name"] if row else "Unknown"
            conn.close()

            report_data = {
                "id": session_id,
                "packer_id": packer_id,
                "packer_name": packer_name,
                "shift": shift or get_shift_for_time(datetime.now()),
                "timestamp": datetime.now().isoformat(),
                "summary": final_summary,
            }
            save_report_to_db(report_data)
            threading.Thread(
                target=shift_report_email,
                args=(shift or get_shift_for_time(datetime.now()),
                  packer_id,
                  final_summary,
                  packer_name),
                daemon=True,
                name=f"mail-shift-{session_id[:8]}",
            ).start()
            
            print(f"[SHIFT] ✓ Shift {shift} report saved for packer {packer_id}")

        except Exception as exc:
            import traceback
            print(f"[SHIFT] Stop session error for {packer_id}: {exc}")
            traceback.print_exc()

    #  Utility 

    def _minutes_to_next_boundary(self) -> int:
        """Minutes until the next shift boundary."""
        now = datetime.now()
        boundaries = [6, 14, 22]   # Hours when shifts switch
        current_minutes = now.hour * 60 + now.minute
        boundary_minutes = [h * 60 for h in boundaries]
        for b in sorted(boundary_minutes):
            if current_minutes < b:
                return b - current_minutes
        # Past last boundary — next is 6 AM tomorrow
        return (24 * 60 - current_minutes) + 6 * 60
