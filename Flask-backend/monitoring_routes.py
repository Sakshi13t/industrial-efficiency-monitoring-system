"""
Live Monitoring Routes - SQLite Version (FIXED - Real-Time Detection)
Handles real-time camera monitoring, video streaming, and metrics with DB persistence.
"""

from flask import Blueprint, request, jsonify, Response
import cv2
import threading
import time
from datetime import datetime
import uuid
from database import get_db_connection
from routes.reports_routes import save_report_to_db
import os

# Create Blueprint
monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/api/monitor')

# Active monitoring sessions (Keep in-memory for the video thread/objects)
active_sessions = {}
session_locks = {}

# --- DB HELPER FUNCTIONS ---

def create_monitoring_session_db(session_id, packer_id, camera_id):
    """Log the start of a monitoring session in SQLite"""
    try:
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO monitoring_sessions (session_id, packer_id, camera_id, started_at, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, packer_id, camera_id, datetime.now().isoformat(), 'running'))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"DB Error (Session Start): {e}")
        return False

def update_session_status_db(session_id, status):
    """Update session status in SQLite"""
    try:
        conn = get_db_connection()
        conn.execute('UPDATE monitoring_sessions SET status = ? WHERE session_id = ?', (status, session_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error (Status Update): {e}")

# --- STREAMING LOGIC ---

def generate_frames(session_id):
    """Video streaming generator function for the React Frontend"""
    while True:
        session = active_sessions.get(session_id)
        if not session or session.get('status') == 'stopped':
            break
            
        with session_locks.get(session_id, threading.Lock()):
            frame = session.get('last_frame')
            if frame is None:
                time.sleep(0.01)
                continue
                
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@monitoring_bp.route('/video_feed/<session_id>')
def video_feed(session_id):
    if session_id not in active_sessions:
        return jsonify({"error": "Session not found"}), 404
    return Response(generate_frames(session_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- MONITORING CONTROL ---

@monitoring_bp.route('/start', methods=['POST'])
def start_monitoring():
    """Start real-time monitoring with RTSP optimizations"""
    global active_sessions, session_locks
    
    from routes.packer_routes import get_packers_db
    from routes.camera_routes import get_cameras_db
    from models.packer_monitor import PackerEfficiencyMonitor
    from app import MODEL_PATH
    
    data = request.json
    packer_id = data.get('packer_id')
    
    packers_db = get_packers_db()
    cameras_db = get_cameras_db()
    
    if packer_id not in packers_db:
        return jsonify({"error": "Packer not found"}), 404
    
    packer_data = packers_db[packer_id]
    camera_id = packer_data.get('camera_id')
    
    if not camera_id or camera_id not in cameras_db:
        return jsonify({"error": "No camera linked to this packer"}), 400
    
    actual_source = cameras_db[camera_id].get('rtsp_url')
    session_id = str(uuid.uuid4())
    
    # Create evidence directory for Proof of Work
    evidence_dir = os.path.join('evidence', session_id)
    os.makedirs(evidence_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"STARTING MONITORING SESSION")
    print(f"{'='*60}")
    print(f"Session ID: {session_id}")
    print(f"Packer ID: {packer_id}")
    print(f"Camera ID: {camera_id}")
    print(f"RTSP URL: {actual_source}")
    print(f"Evidence Dir: {evidence_dir}")
    print(f"\n⚠️  LINE CONFIGURATION (Frame width: 1280px)")
    print(f"Start Line Position: {packer_data.get('start_line_position', 0.2)} → {int(1280 * float(packer_data.get('start_line_position', 0.2)))}px")
    print(f"Detection Line Position: {packer_data.get('line_position', 0.7)} → {int(1280 * float(packer_data.get('line_position', 0.7)))}px")
    print(f"\n⚠️  RECOMMENDATION: If objects disappear before crossing detection line,")
    print(f"   reduce 'line_position' from {packer_data.get('line_position', 0.7)} to 0.5 or 0.55")
    print(f"{'='*60}\n")
    
    try:
        # Initialize monitor with OPTIMIZED settings
        monitor = PackerEfficiencyMonitor(
            model_path=MODEL_PATH,
            line_position=float(packer_data.get('line_position', 0.7)),
            start_line_position=float(packer_data.get('start_line_position', 0.2)),
            confidence_threshold=float(packer_data.get('confidence_threshold', 0.5)),
            spouts=int(packer_data.get('spouts', 8)),
            rpm=int(packer_data.get('rpm', 5)),
            enable_debug=True,  # Console debug
            visual_debug=False  # CRITICAL: Disable visual debug for performance
        )
        
        # CRITICAL FIX: Optimized RTSP capture configuration
        cap = cv2.VideoCapture(actual_source)
        
        # CRITICAL: These settings prevent buffering and frame glitches
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer - prevents lag
        
        # Additional RTSP optimizations to prevent frame drops
        cap.set(cv2.CAP_PROP_FPS, 25)  # Reduced from 30 to 25 for stability
        
        # Try to use hardware acceleration if available
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        except:
            pass  # Fallback to default codec
        
        # CRITICAL: Set backend to use FFmpeg for better RTSP handling
        # Reopen with explicit backend
        cap.release()
        cap = cv2.VideoCapture(actual_source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set low-latency flags for RTSP
        # This reduces buffering at the cost of occasional frame drops
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 second read timeout
        
        if not cap.isOpened():
            print(f"[ERROR] Failed to open RTSP stream: {actual_source}")
            return jsonify({"error": f"Failed to connect to: {actual_source}"}), 400
        
        # Read and discard first few frames (camera warmup)
        print("[INIT] Warming up camera stream...")
        for i in range(5):
            ret, _ = cap.read()
            if not ret:
                print(f"[WARNING] Warmup frame {i+1}/5 failed")
        
        # CRITICAL DIAGNOSTIC: Verify we can actually read frames AND detect objects
        print("\n[DIAGNOSTIC] Testing detection capability on camera feed...")
        test_detections = {'bag_present': 0, 'no_bag': 0, 'bag_stuck_filled': 0}
        
        for test_frame_num in range(10):
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                continue
            
            # Run detection
            test_results = monitor.model(test_frame, conf=monitor.confidence_threshold, verbose=False)
            for result in test_results:
                for box in result.boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = result.names[cls]
                    test_detections[class_name] = test_detections.get(class_name, 0) + 1
        
        print(f"[DIAGNOSTIC] Test Results (10 frames):")
        print(f"  bag_present: {test_detections.get('bag_present', 0)}")
        print(f"  no_bag: {test_detections.get('no_bag', 0)}")
        print(f"  bag_stuck_filled: {test_detections.get('bag_stuck_filled', 0)}")
        print(f"  Total: {sum(test_detections.values())}")
        
        if sum(test_detections.values()) == 0:
            print("\n⚠️  WARNING: NO DETECTIONS FOUND IN TEST FRAMES!")
            print("   Possible causes:")
            print("   1. Camera angle differs from training video")
            print("   2. Objects too small/far in camera view")
            print("   3. Lighting conditions very different")
            print("   4. Camera resolution mismatch")
        
        if test_detections.get('no_bag', 0) == 0 and test_detections.get('bag_present', 0) > 0:
            print("\n⚠️  WARNING: Detecting bag_present but NOT no_bag!")
            print("   This suggests:")
            print("   1. no_bag objects appear differently in camera vs training")
            print("   2. no_bag objects may be smaller/faster in camera feed")
            print("   3. Timing/position of no_bag differs from training")
        
        print("")
        
        print(f"[SUCCESS] Stream ready! Frame shape: {test_frame.shape}")
        
    except Exception as e:
        print(f"[ERROR] Initialization failed: {str(e)}")
        return jsonify({"error": f"Init failed: {str(e)}"}), 500

    # Record in SQLite
    create_monitoring_session_db(session_id, packer_id, camera_id)

    # Add to active sessions
    active_sessions[session_id] = {
        "session_id": session_id,
        "packer_id": packer_id,
        "monitor": monitor,
        "capture": cap,
        "status": "running",
        "last_frame": None,
        "evidence_dir": evidence_dir
    }
    session_locks[session_id] = threading.Lock()
    
    # Update packer status
    from routes.packer_routes import update_packer_status_internal
    update_packer_status_internal(packer_id, 'active', session_id)
    
    def process_stream():
        """Background thread for processing RTSP stream with error recovery"""
        session = active_sessions.get(session_id)
        frame_count = 0
        error_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5  # Reconnect after 5 consecutive failures
        last_frame = None  # Keep last good frame for display
        
        print(f"\n[THREAD] Stream processing thread started for session {session_id}")
        
        while session and session.get('status') == 'running':
            ret, frame = cap.read()
            
            if not ret or frame is None:
                consecutive_errors += 1
                error_count += 1
                print(f"[WARNING] Frame read failed ({consecutive_errors}/{max_consecutive_errors})")
                
                # CRITICAL FIX: If too many consecutive errors, try to flush buffer
                if consecutive_errors >= 3:
                    print("[FIX] Flushing RTSP buffer...")
                    # Read and discard several frames to clear buffer
                    for _ in range(5):
                        cap.grab()  # Just grab, don't decode
                    consecutive_errors = 0  # Reset counter after flush
                
                # If still failing after flush, try reconnection
                if consecutive_errors >= max_consecutive_errors:
                    print("[CRITICAL] Too many errors, attempting stream reconnection...")
                    cap.release()
                    time.sleep(1)  # Brief pause before reconnect
                    
                    # Attempt reconnection
                    try:
                        new_cap = cv2.VideoCapture(actual_source, cv2.CAP_FFMPEG)
                        new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        if new_cap.isOpened():
                            globals()['cap'] = new_cap  # Replace global cap
                            print("[SUCCESS] Stream reconnected")
                            consecutive_errors = 0
                        else:
                            print("[ERROR] Reconnection failed")
                            time.sleep(2)
                    except Exception as e:
                        print(f"[ERROR] Reconnection error: {e}")
                        time.sleep(2)
                
                # Display last good frame if available
                if last_frame is not None:
                    with session_locks[session_id]:
                        session['last_frame'] = last_frame.copy()
                
                time.sleep(0.05)  # Wait before retry
                continue
            
            # Reset error counter on successful read
            consecutive_errors = 0
            frame_count += 1
            
            # Keep this as last good frame
            last_frame = frame.copy()
            
            # Verify frame is valid
            if frame.size == 0:
                print("[ERROR] Empty frame received")
                continue
            
            # CRITICAL FIX: Clear buffer by grabbing extra frames occasionally
            # This prevents buffer buildup that causes glitches
            if frame_count % 30 == 0:
                # Grab (but don't decode) next frame to stay current
                cap.grab()
            
            # Log frame info periodically
            if frame_count % 100 == 0:
                fps = frame_count / (time.time() - session['monitor'].start_time)
                print(f"[STREAM] {frame_count} frames, {fps:.1f} FPS, Errors: {error_count}")
            
            # Process with monitor
            try:
                processed_frame = session['monitor'].process_frame(frame)
            except Exception as e:
                print(f"[ERROR] Frame processing failed: {e}")
                continue
            
            # PROOF LOGIC: Capture evidence when event detected
            if hasattr(session['monitor'], 'last_event_type') and session['monitor'].last_event_type:
                event_type = session['monitor'].last_event_type
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{event_type}_{timestamp}.jpg"
                filepath = os.path.join(session['evidence_dir'], filename)
                
                cv2.imwrite(filepath, frame)
                print(f"[EVIDENCE] Saved: {filename}")
                session['monitor'].last_event_type = None
            
            # Draw detection lines on frame
            h, w = frame.shape[:2]
            
            # Detection line (red)
            line_x = int(w * float(packer_data['line_position']))
            cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 3)
            cv2.putText(frame, "DETECTION", (line_x + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Start line (yellow)
            start_line_x = int(w * float(packer_data['start_line_position']))
            cv2.line(frame, (start_line_x, 0), (start_line_x, h), (0, 255, 255), 2)
            cv2.putText(frame, "START", (start_line_x + 10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add frame counter and status
            status_color = (0, 255, 0) if consecutive_errors == 0 else (0, 165, 255)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            if error_count > 0:
                cv2.putText(frame, f"Errors: {error_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Update session frame (thread-safe)
            with session_locks[session_id]:
                session['last_frame'] = frame
            
            # CRITICAL: Adaptive frame rate control
            # Faster when no errors, slower when recovering
            sleep_time = 0.04 if consecutive_errors == 0 else 0.1
            time.sleep(sleep_time)
        
        # Cleanup
        print(f"\n[THREAD] Stopping stream processing for session {session_id}")
        cap.release()
        update_session_status_db(session_id, 'stopped')
        update_packer_status_internal(packer_id, 'idle', None)
        print(f"[THREAD] Session {session_id} cleanup complete\n")

    # Start background processing thread
    threading.Thread(target=process_stream, daemon=True).start()

    return jsonify({
        "message": "Monitoring started successfully",
        "session_id": session_id,
        "debug_enabled": True
    }), 201

@monitoring_bp.route('/stop/<session_id>', methods=['POST'])
def stop_monitoring(session_id):
    """Stop monitoring and generate report"""
    if session_id not in active_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    print(f"\n{'='*60}")
    print(f"STOPPING MONITORING SESSION")
    print(f"{'='*60}")
    print(f"Session ID: {session_id}")
    
    session = active_sessions[session_id]
    session['status'] = 'stopped'
    packer_id = session['packer_id']
    monitor = session['monitor']
    
    # Get final summary
    final_summary = monitor.get_summary()
    
    print(f"Final Summary:")
    print(f"  Total Events: {final_summary['total_events']}")
    print(f"  Bags Placed: {final_summary['bags_placed']}")
    print(f"  Bags Missed: {final_summary['bags_missed']}")
    print(f"  Stuck Bags: {final_summary['stuck_bags']}")
    print(f"  Packer Efficiency: {final_summary['packer_efficiency']}%")
    print(f"{'='*60}\n")
    
    # Get packer info
    conn = get_db_connection()
    packer = conn.execute('SELECT name FROM packers WHERE id = ?', (packer_id,)).fetchone()
    packer_name = packer['name'] if packer else 'Unknown'
    conn.close()
    
    # Create report
    report_data = {
        'id': session_id,
        'packer_id': packer_id,
        'packer_name': packer_name,
        'timestamp': datetime.now().isoformat(),
        'summary': final_summary
    }
    
    # Save to database
    save_success = save_report_to_db(report_data)
    
    # Update session status in DB
    update_session_status_db(session_id, 'completed')
    
    return jsonify({
        "message": "Monitoring stopped successfully",
        "report_saved": save_success,
        "report_id": report_data['id'],
        "summary": final_summary
    }), 200


@monitoring_bp.route('/metrics/<session_id>', methods=['GET'])
def get_live_metrics(session_id):
    """Get live metrics for active session"""
    if session_id not in active_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = active_sessions[session_id]
    return jsonify({
        "session_id": session_id,
        "status": session['status'],
        "metrics": session['monitor'].get_summary()
    }), 200


# --- HELPER FUNCTION ---
def get_active_monitor_summary(session_data):
    """
    Helper function to safely get monitor summary from active session
    Returns empty dict if monitor is not available
    """
    try:
        if session_data.get('monitor') and hasattr(session_data['monitor'], 'get_summary'):
            return session_data['monitor'].get_summary()
    except Exception as e:
        print(f"Error getting monitor summary: {e}")
    return {}


@monitoring_bp.route('/active-sessions', methods=['GET'])
def get_active_sessions_endpoint():
    """Get all active monitoring sessions with their metrics"""
    sessions_list = []
    for session_id, session in active_sessions.items():
        if session.get('status') == 'running':
            sessions_list.append({
                'session_id': session_id,
                'packer_id': session.get('packer_id'),
                'status': session.get('status'),
                'metrics': get_active_monitor_summary(session)
            })
    
    return jsonify({
        'active_sessions': sessions_list, 
        'count': len(sessions_list)
    }), 200


# --- DIAGNOSTIC ENDPOINT ---
@monitoring_bp.route('/test-camera/<camera_id>', methods=['GET'])
def test_camera(camera_id):
    """Test if camera stream is working (Diagnostic Tool)"""
    from routes.camera_routes import get_cameras_db
    
    cameras_db = get_cameras_db()
    if camera_id not in cameras_db:
        return jsonify({"error": "Camera not found"}), 404
    
    rtsp_url = cameras_db[camera_id]['rtsp_url']
    
    print(f"\n[TEST] Testing camera: {camera_id}")
    print(f"[TEST] RTSP URL: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"[TEST] FAILED - Cannot open stream")
        return jsonify({
            "status": "failed", 
            "error": "Cannot open stream",
            "rtsp_url": rtsp_url
        }), 400
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print(f"[TEST] FAILED - Cannot read frame")
        return jsonify({
            "status": "failed", 
            "error": "Cannot read frame",
            "rtsp_url": rtsp_url
        }), 400
    
    print(f"[TEST] SUCCESS - Frame shape: {frame.shape}")
    
    return jsonify({
        "status": "success",
        "frame_shape": list(frame.shape),
        "rtsp_url": rtsp_url,
        "message": "Camera stream is working correctly"
    }), 200
    

# """
# Live Monitoring Routes - SQLite Version (FIXED)
# Handles real-time camera monitoring, video streaming, and metrics with DB persistence.
# """

# from flask import Blueprint, request, jsonify, Response
# import cv2
# import threading
# import time
# from datetime import datetime
# import uuid
# from database import get_db_connection
# from routes.reports_routes import save_report_to_db
# import os

# # Create Blueprint
# monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/api/monitor')

# # Active monitoring sessions (Keep in-memory for the video thread/objects)
# active_sessions = {}
# session_locks = {}

# # --- DB HELPER FUNCTIONS ---

# def create_monitoring_session_db(session_id, packer_id, camera_id):
#     """Log the start of a monitoring session in SQLite"""
#     try:
#         conn = get_db_connection()
#         conn.execute('''
#             INSERT INTO monitoring_sessions (session_id, packer_id, camera_id, started_at, status)
#             VALUES (?, ?, ?, ?, ?)
#         ''', (session_id, packer_id, camera_id, datetime.now().isoformat(), 'running'))
#         conn.commit()
#         conn.close()
#         return True
#     except Exception as e:
#         print(f"DB Error (Session Start): {e}")
#         return False

# def update_session_status_db(session_id, status):
#     """Update session status in SQLite"""
#     try:
#         conn = get_db_connection()
#         conn.execute('UPDATE monitoring_sessions SET status = ? WHERE session_id = ?', (status, session_id))
#         conn.commit()
#         conn.close()
#     except Exception as e:
#         print(f"DB Error (Status Update): {e}")

# # --- STREAMING LOGIC ---

# def generate_frames(session_id):
#     """Video streaming generator function for the React Frontend"""
#     while True:
#         session = active_sessions.get(session_id)
#         if not session or session.get('status') == 'stopped':
#             break
            
#         with session_locks.get(session_id, threading.Lock()):
#             frame = session.get('last_frame')
#             if frame is None:
#                 time.sleep(0.01)
#                 continue
                
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 continue
#             frame_bytes = buffer.tobytes()
            
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @monitoring_bp.route('/video_feed/<session_id>')
# def video_feed(session_id):
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
#     return Response(generate_frames(session_id),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# # --- MONITORING CONTROL ---

# @monitoring_bp.route('/start', methods=['POST'])
# def start_monitoring():
#     """
#     FIXED: Start monitoring with improved live detection settings
#     - Reduced class_stability_frames to 2
#     - Enabled debug mode by default for troubleshooting
#     """
#     global active_sessions, session_locks
    
#     from routes.packer_routes import get_packers_db
#     from routes.camera_routes import get_cameras_db
#     from models.packer_monitor import PackerEfficiencyMonitor
#     from app import MODEL_PATH
    
#     data = request.json
#     packer_id = data.get('packer_id')
    
#     packers_db = get_packers_db()
#     cameras_db = get_cameras_db()
    
#     if packer_id not in packers_db:
#         return jsonify({"error": "Packer not found"}), 404
    
#     packer_data = packers_db[packer_id]
#     camera_id = packer_data.get('camera_id')
    
#     if not camera_id or camera_id not in cameras_db:
#         return jsonify({"error": "No camera linked to this packer"}), 400
    
#     actual_source = cameras_db[camera_id].get('rtsp_url')
#     session_id = str(uuid.uuid4())

#     # Create evidence directory for Proof of Work
#     evidence_dir = os.path.join('evidence', session_id)
#     os.makedirs(evidence_dir, exist_ok=True)
    
#     try:
#         # FIX: Reduced class_stability_frames from 5 to 2 for live video
#         # FIX: Enabled debug mode to see what's happening
#         monitor = PackerEfficiencyMonitor(
#             model_path=MODEL_PATH,
#             line_position=float(packer_data.get('line_position', 0.7)),
#             start_line_position=float(packer_data.get('start_line_position', 0.4)),
#             confidence_threshold=float(packer_data.get('confidence_threshold', 0.5)),
#             spouts=int(packer_data.get('spouts', 8)),
#             class_stability_frames=2,  # FIXED: Reduced from 5 to 2
#             enable_debug=True  # FIXED: Enable debug logging
#         )
#         cap = cv2.VideoCapture(actual_source)
#         if not cap.isOpened():
#              return jsonify({"error": f"Failed to connect to: {actual_source}"}), 400
#     except Exception as e:
#         return jsonify({"error": f"Init failed: {str(e)}"}), 500

#     create_monitoring_session_db(session_id, packer_id, camera_id)

#     active_sessions[session_id] = {
#         "session_id": session_id,
#         "packer_id": packer_id,
#         "monitor": monitor,
#         "capture": cap,
#         "status": "running",
#         "last_frame": None,
#         "evidence_dir": evidence_dir
#     }
#     session_locks[session_id] = threading.Lock()
    
#     from routes.packer_routes import update_packer_status_internal
#     update_packer_status_internal(packer_id, 'active', session_id)
    
#     def process_stream():
#         session = active_sessions.get(session_id)
#         while session and session.get('status') == 'running':
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"[ERROR] Failed to read frame from camera")
#                 break

#             # Process frame with monitor
#             session['monitor'].process_frame(frame)
            
#             # PROOF LOGIC: If monitor flagged an event, capture a snapshot
#             if hasattr(session['monitor'], 'last_event_type') and session['monitor'].last_event_type:
#                 event_type = session['monitor'].last_event_type
#                 timestamp = datetime.now().strftime("%H-%M-%S")
#                 filename = f"{event_type}_{timestamp}.jpg"
#                 cv2.imwrite(os.path.join(evidence_dir, filename), frame)
#                 session['monitor'].last_event_type = None

#             # Draw detection line on frame
#             h, w = frame.shape[:2]
#             line_x = int(w * float(packer_data['line_position']))
#             start_line_x = int(w * float(packer_data.get('start_line_position', 0.2)))
            
#             # Draw detection line (red)
#             cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 3)
            
#             # Draw start line (blue)
#             cv2.line(frame, (start_line_x, 0), (start_line_x, h), (255, 0, 0), 2)
            
#             # Add labels
#             cv2.putText(frame, "Detection Line", (line_x - 100, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             cv2.putText(frame, "Start Line", (start_line_x - 80, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
#             with session_locks[session_id]:
#                 session['last_frame'] = frame
#             time.sleep(0.01)

#         cap.release()
#         update_session_status_db(session_id, 'stopped')
#         update_packer_status_internal(packer_id, 'idle', None)
#         print(f"[INFO] Stream processing stopped for session {session_id}")

#     threading.Thread(target=process_stream, daemon=True).start()
#     return jsonify({"message": "Monitoring started", "session_id": session_id}), 201


# @monitoring_bp.route('/stop/<session_id>', methods=['POST'])
# def stop_monitoring(session_id):
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
    
#     session = active_sessions[session_id]
#     session['status'] = 'stopped'
#     packer_id = session['packer_id']
#     monitor = session['monitor']
    
#     # Get final summary
#     final_summary = monitor.get_summary()
    
#     print(f"[INFO] Monitoring stopped for session {session_id}")
#     print(f"[SUMMARY] {final_summary}")
    
#     # Get packer info
#     conn = get_db_connection()
#     packer = conn.execute('SELECT name FROM packers WHERE id = ?', (packer_id,)).fetchone()
#     packer_name = packer['name'] if packer else 'Unknown'
#     conn.close()
    
#     # Create report
#     report_data = {
#         'id': session_id,
#         'packer_id': packer_id,
#         'packer_name': packer_name,
#         'timestamp': datetime.now().isoformat(),
#         'summary': final_summary
#     }
    
#     # Save to database
#     save_success = save_report_to_db(report_data)
    
#     # Update session status in DB
#     update_session_status_db(session_id, 'completed')
    
#     return jsonify({
#         "message": "Monitoring stopped",
#         "report_saved": save_success,
#         "report_id": report_data['id'],
#         "summary": final_summary
#     }), 200


# @monitoring_bp.route('/metrics/<session_id>', methods=['GET'])
# def get_live_metrics(session_id):
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
#     session = active_sessions[session_id]
#     return jsonify({
#         "session_id": session_id,
#         "status": session['status'],
#         "metrics": session['monitor'].get_summary()
#     }), 200


# def get_active_monitor_summary(session_data):
#     """
#     Helper function to safely get monitor summary from active session
#     Returns empty dict if monitor is not available
#     """
#     try:
#         if session_data.get('monitor') and hasattr(session_data['monitor'], 'get_summary'):
#             return session_data['monitor'].get_summary()
#     except Exception as e:
#         print(f"Error getting monitor summary: {e}")
#     return {}


# @monitoring_bp.route('/active-sessions', methods=['GET'])
# def get_active_sessions_endpoint():
#     """Get all active monitoring sessions with their metrics"""
#     sessions_list = []
#     for session_id, session in active_sessions.items():
#         if session.get('status') == 'running':
#             sessions_list.append({
#                 'session_id': session_id,
#                 'packer_id': session.get('packer_id'),
#                 'status': session.get('status'),
#                 'metrics': get_active_monitor_summary(session)
#             })
#     return jsonify({
#         'active_sessions': sessions_list, 
#         'count': len(sessions_list)
#     }), 200


# @monitoring_bp.route('/debug/<session_id>', methods=['GET'])
# def debug_session(session_id):
#     """
#     NEW ENDPOINT: Debug endpoint to see track classification details
#     Use this to diagnose why certain classes aren't being detected
#     """
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
    
#     session = active_sessions[session_id]
#     monitor = session['monitor']
    
#     # Get detailed tracking info
#     track_details = {}
#     for track_id in monitor.track_class.keys():
#         track_details[str(track_id)] = {
#             'current_class': monitor.track_class.get(track_id, 'unknown'),
#             'class_history': list(monitor.track_class_history[track_id]),
#             'confidence_history': [round(c, 2) for c in monitor.track_confidence_history[track_id]],
#             'stable_class': monitor.get_stable_class(track_id),
#             'crossed_start_line': track_id in monitor.crossed_start_line,
#             'crossed_detection_line': track_id in monitor.crossed_objects,
#             'is_stuck_bag': track_id in monitor.stuck_bag_ids
#         }
    
#     return jsonify({
#         "session_id": session_id,
#         "track_details": track_details,
#         "total_tracks": len(track_details),
#         "metrics": monitor.get_summary(),
#         "crossed_objects": list(monitor.crossed_objects),
#         "crossed_start_line": list(monitor.crossed_start_line),
#         "stuck_bag_ids": list(monitor.stuck_bag_ids)
#     }), 200


# """
# Live Monitoring Routes - SQLite Version
# Handles real-time camera monitoring, video streaming, and metrics with DB persistence.
# """

# from flask import Blueprint, request, jsonify, Response
# import cv2
# import threading
# import time
# from datetime import datetime
# import uuid
# from database import get_db_connection
# from routes.reports_routes import save_report_to_db
# import os

# # Create Blueprint
# monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/api/monitor')

# # Active monitoring sessions (Keep in-memory for the video thread/objects)
# active_sessions = {}
# session_locks = {}

# # --- DB HELPER FUNCTIONS ---

# def create_monitoring_session_db(session_id, packer_id, camera_id):
#     """Log the start of a monitoring session in SQLite"""
#     try:
#         conn = get_db_connection()
#         conn.execute('''
#             INSERT INTO monitoring_sessions (session_id, packer_id, camera_id, started_at, status)
#             VALUES (?, ?, ?, ?, ?)
#         ''', (session_id, packer_id, camera_id, datetime.now().isoformat(), 'running'))
#         conn.commit()
#         conn.close()
#         return True
#     except Exception as e:
#         print(f"DB Error (Session Start): {e}")
#         return False

# def update_session_status_db(session_id, status):
#     """Update session status in SQLite"""
#     try:
#         conn = get_db_connection()
#         conn.execute('UPDATE monitoring_sessions SET status = ? WHERE session_id = ?', (status, session_id))
#         conn.commit()
#         conn.close()
#     except Exception as e:
#         print(f"DB Error (Status Update): {e}")

# # --- STREAMING LOGIC ---

# def generate_frames(session_id):
#     """Video streaming generator function for the React Frontend"""
#     while True:
#         session = active_sessions.get(session_id)
#         if not session or session.get('status') == 'stopped':
#             break
            
#         with session_locks.get(session_id, threading.Lock()):
#             frame = session.get('last_frame')
#             if frame is None:
#                 time.sleep(0.01)
#                 continue
                
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 continue
#             frame_bytes = buffer.tobytes()
            
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @monitoring_bp.route('/video_feed/<session_id>')
# def video_feed(session_id):
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
#     return Response(generate_frames(session_id),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# # --- MONITORING CONTROL ---

# # routes/monitoring_routes.py

# @monitoring_bp.route('/start', methods=['POST'])
# def start_monitoring():
#     # Use the global variables from the top of the file
#     global active_sessions, session_locks
    
#     from routes.packer_routes import get_packers_db
#     from routes.camera_routes import get_cameras_db
#     from models.packer_monitor import PackerEfficiencyMonitor
#     from app import MODEL_PATH
    
#     data = request.json
#     packer_id = data.get('packer_id')
    
#     packers_db = get_packers_db()
#     cameras_db = get_cameras_db() # Get the dictionary from the helper
    
#     if packer_id not in packers_db:
#         return jsonify({"error": "Packer not found"}), 404
    
#     packer_data = packers_db[packer_id]
#     camera_id = packer_data.get('camera_id')
    
#     if not camera_id or camera_id not in cameras_db:
#         return jsonify({"error": "No camera linked to this packer"}), 400
    
#     # Access the source URL from the global dictionary
#     actual_source = cameras_db[camera_id].get('rtsp_url')
#     session_id = str(uuid.uuid4()) # Generate ID early

#     # Create evidence directory for Proof of Work
#     evidence_dir = os.path.join('evidence', session_id)
#     os.makedirs(evidence_dir, exist_ok=True)
    
#     try:
#         monitor = PackerEfficiencyMonitor(
#             model_path=MODEL_PATH,
#             line_position=float(packer_data.get('line_position', 0.7)),
#             start_line_position=float(packer_data.get('start_line_position', 0.2)),
#             confidence_threshold=float(packer_data.get('confidence_threshold', 0.5)),
#             spouts=int(packer_data.get('spouts', 8))
#         )
#         cap = cv2.VideoCapture(actual_source)
#         if not cap.isOpened():
#              return jsonify({"error": f"Failed to connect to: {actual_source}"}), 400
#     except Exception as e:
#         return jsonify({"error": f"Init failed: {str(e)}"}), 500

#     create_monitoring_session_db(session_id, packer_id, camera_id)

#     active_sessions[session_id] = {
#         "session_id": session_id,
#         "packer_id": packer_id,
#         "monitor": monitor,
#         "capture": cap,
#         "status": "running",
#         "last_frame": None,
#         "evidence_dir": evidence_dir # Store for the thread to use
#     }
#     session_locks[session_id] = threading.Lock()
    
#     from routes.packer_routes import update_packer_status_internal
#     update_packer_status_internal(packer_id, 'active', session_id)
    
#     def process_stream():
#         session = active_sessions.get(session_id)
#         while session and session.get('status') == 'running':
#             ret, frame = cap.read()
#             if not ret: break

#             session['monitor'].process_frame(frame)
            
#             # PROOF LOGIC: If monitor flagged an event, capture a snapshot
#             if hasattr(session['monitor'], 'last_event_type') and session['monitor'].last_event_type:
#                 event_type = session['monitor'].last_event_type
#                 timestamp = datetime.now().strftime("%H-%M-%S")
#                 filename = f"{event_type}_{timestamp}.jpg"
#                 # Save the frame to the unique evidence folder
#                 cv2.imwrite(os.path.join(evidence_dir, filename), frame)
#                 session['monitor'].last_event_type = None # Reset trigger

#             h, w = frame.shape[:2]
#             cv2.line(frame, (int(w * float(packer_data['line_position'])), 0), 
#                      (int(w * float(packer_data['line_position'])), h), (0, 0, 255), 3)
            
#             with session_locks[session_id]:
#                 session['last_frame'] = frame
#             time.sleep(0.01)

#         cap.release()
#         update_session_status_db(session_id, 'stopped')
#         update_packer_status_internal(packer_id, 'idle', None)

#     threading.Thread(target=process_stream, daemon=True).start()
#     return jsonify({"message": "Monitoring started", "session_id": session_id}), 201

# # @monitoring_bp.route('/start', methods=['POST'])
# # def start_monitoring():
# #     from routes.packer_routes import get_packers_db
# #     from routes.camera_routes import get_cameras_db
# #     from models.packer_monitor import PackerEfficiencyMonitor
# #     from app import MODEL_PATH
# #     import os
    
# #     actual_source = cameras_db[camera_id].get('rtsp_url')
# #     session_id = str(uuid.uuid4())
    
# #     evidence_dir = os.path.join('evidence', session_id)
# #     os.makedirs(evidence_dir, exist_ok=True)
    
# #     data = request.json
# #     packer_id = data.get('packer_id')
    
# #     packers_db = get_packers_db()
# #     cameras_db = get_cameras_db()
    
# #     if packer_id not in packers_db:
# #         return jsonify({"error": "Packer not found"}), 404
    
# #     packer_data = packers_db[packer_id]
# #     camera_id = packer_data.get('camera_id')
    
# #     if not camera_id or camera_id not in cameras_db:
# #         return jsonify({"error": "No camera linked to this packer"}), 400
    

    
# #     try:
# #         monitor = PackerEfficiencyMonitor(
# #             model_path=MODEL_PATH,
# #             line_position=float(packer_data.get('line_position', 0.7)),
# #             start_line_position=float(packer_data.get('start_line_position', 0.2)),
# #             confidence_threshold=float(packer_data.get('confidence_threshold', 0.5)),
# #             spouts=int(packer_data.get('spouts', 8))
# #         )
# #         cap = cv2.VideoCapture(actual_source)
# #         if not cap.isOpened():
# #              return jsonify({"error": f"Failed to connect to: {actual_source}"}), 400
# #     except Exception as e:
# #         return jsonify({"error": f"Init failed: {str(e)}"}), 500

# #     # Record in SQLite
# #     create_monitoring_session_db(session_id, packer_id, camera_id)

# #     # Add to active sessions
# #     active_sessions[session_id] = {
# #         "session_id": session_id,
# #         "packer_id": packer_id,
# #         "monitor": monitor,
# #         "capture": cap,
# #         "status": "running",
# #         "last_frame": None
# #     }
# #     session_locks[session_id] = threading.Lock()
    
# #     # Update status in packer_routes (helper)
# #     from routes.packer_routes import update_packer_status_internal
# #     update_packer_status_internal(packer_id, 'active', session_id)
    
# #     def process_stream():
# #         session = active_sessions.get(session_id)
# #         while session and session.get('status') == 'running':
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             session['monitor'].process_frame(frame)
            
# #             # EVIDENCE CAPTURE LOGIC
# #             if session['monitor'].last_event_type:
# #                 event_type = session['monitor'].last_event_type
# #                 timestamp = datetime.now().strftime("%H-%M-%S")
# #                 filename = f"{event_type}_{timestamp}.jpg"
# #                 filepath = os.path.join(evidence_dir, filename)
                
# #                 # Save the frame as proof
# #                 cv2.imwrite(filepath, frame)
                
# #                 # Reset the trigger
# #                 session['monitor'].last_event_type = None
                
# #             h, w = frame.shape[:2]
# #             cv2.line(frame, (int(w * packer_data['line_position']), 0), (int(w * packer_data['line_position']), h), (0, 0, 255), 3)
            
# #             with session_locks[session_id]:
# #                 session['last_frame'] = frame
# #             time.sleep(0.01)

# #         cap.release()
# #         update_session_status_db(session_id, 'stopped')
# #         update_packer_status_internal(packer_id, 'idle', None)

# #     threading.Thread(target=process_stream, daemon=True).start()

# #     return jsonify({"message": "Monitoring started", "session_id": session_id}), 201

# @monitoring_bp.route('/stop/<session_id>', methods=['POST'])
# def stop_monitoring(session_id):
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
    
#     session = active_sessions[session_id]
#     session['status'] = 'stopped'
#     packer_id = session['packer_id']
#     monitor = session['monitor']
    
#     # Get final summary
#     final_summary = monitor.get_summary()
    
#     # Get packer info
#     conn = get_db_connection()
#     packer = conn.execute('SELECT name FROM packers WHERE id = ?', (packer_id,)).fetchone()
#     packer_name = packer['name'] if packer else 'Unknown'
#     conn.close()
    
#     # Create report
#     report_data = {
#         # 'id': str(uuid.uuid4()),
#         'id':session_id,
#         'packer_id': packer_id,
#         'packer_name': packer_name,
#         'timestamp': datetime.now().isoformat(),
#         'summary': final_summary
#     }
    
#     # Save to database
#     save_success = save_report_to_db(report_data)
    
#     # Update session status in DB
#     update_session_status_db(session_id, 'completed')
    
#     return jsonify({
#         "message": "Monitoring stopped",
#         "report_saved": save_success,
#         "report_id": report_data['id']
#     }), 200

# @monitoring_bp.route('/metrics/<session_id>', methods=['GET'])
# def get_live_metrics(session_id):
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
#     session = active_sessions[session_id]
#     return jsonify({
#         "session_id": session_id,
#         "status": session['status'],
#         "metrics": session['monitor'].get_summary()
#     }), 200
    
# # Active monitoring sessions (Keep in-memory for the video thread/objects)
# active_sessions = {}
# session_locks = {}

# # --- ADD THIS NEW HELPER FUNCTION ---
# def get_active_monitor_summary(session_data):
#     """
#     Helper function to safely get monitor summary from active session
#     Returns empty dict if monitor is not available
#     """
#     try:
#         if session_data.get('monitor') and hasattr(session_data['monitor'], 'get_summary'):
#             return session_data['monitor'].get_summary()
#     except Exception as e:
#         print(f"Error getting monitor summary: {e}")
#     return {}

# # --- ADD THIS NEW ENDPOINT ---
# @monitoring_bp.route('/active-sessions', methods=['GET'])
# def get_active_sessions_endpoint():
#     """Get all active monitoring sessions with their metrics"""
#     sessions_list = []
#     for session_id, session in active_sessions.items():
#         if session.get('status') == 'running':
#             sessions_list.append({
#                 'session_id': session_id,
#                 'packer_id': session.get('packer_id'),
#                 'status': session.get('status'),
#                 'metrics': get_active_monitor_summary(session)
#             })
#     return jsonify({
#         'active_sessions': sessions_list, 
#         'count': len(sessions_list)
#     }), 200