# """
# Camera Management Routes
# Handles registration of physical hardware (RTSP, IP Cameras)
# """

# from flask import Blueprint, request, jsonify
# from datetime import datetime
# import uuid

# # Create Blueprint
# camera_bp = Blueprint('camera', __name__, url_prefix='/api/cameras')

# # In-memory storage for testing (integrate with DB for production)
# cameras_db = {}

# @camera_bp.route('', methods=['POST'])
# def add_camera():
#     """
#     Register a new physical camera source
#     Request Body: { "name": "Cam-01", "rtsp_url": "rtsp://..." }
#     """
#     data = request.json
    
#     if not data.get('name') or not data.get('rtsp_url'):
#         return jsonify({"error": "Camera name and RTSP URL are required"}), 400

#     camera_id = str(uuid.uuid4())
#     camera_config = {
#         "id": camera_id,
#         "name": data.get('name'),
#         "rtsp_url": data.get('rtsp_url'),
#         "status": "online", # Initial check status
#         "created_at": datetime.now().isoformat()
#     }
    
#     cameras_db[camera_id] = camera_config
    
#     return jsonify({
#         "message": "Camera registered successfully",
#         "camera_id": camera_id,
#         "camera": camera_config
#     }), 201

# @camera_bp.route('', methods=['GET'])
# def list_cameras():
#     """Returns all registered cameras for the Packer Master dropdown"""
#     return jsonify({
#         "cameras": list(cameras_db.values()),
#         "total": len(cameras_db)
#     }), 200

# @camera_bp.route('/<camera_id>', methods=['DELETE'])
# def delete_camera(camera_id):
#     """Removes a camera registration"""
#     if camera_id not in cameras_db:
#         return jsonify({"error": "Camera not found"}), 404
    
#     del cameras_db[camera_id]
#     return jsonify({"message": "Camera removed successfully"}), 200

# def get_cameras_db():
#     """Helper to access cameras from other modules"""
#     return cameras_db

# """
# Camera Management Routes
# Handles registration of physical hardware (RTSP, IP Cameras)
# """

# from flask import Blueprint, request, jsonify
# from datetime import datetime
# import uuid

# # Create Blueprint
# camera_bp = Blueprint('camera', __name__, url_prefix='/api/cameras')

# # In-memory storage for testing (integrate with DB for production)
# cameras_db = {}

# @camera_bp.route('', methods=['POST'])
# def add_camera():
#     """
#     Register a new physical camera source
#     Request Body: { "name": "Cam-01", "rtsp_url": "rtsp://..." }
#     """
#     data = request.json
    
#     if not data.get('name') or not data.get('rtsp_url'):
#         return jsonify({"error": "Camera name and RTSP URL are required"}), 400

#     camera_id = str(uuid.uuid4())
#     camera_config = {
#         "id": camera_id,
#         "name": data.get('name'),
#         "rtsp_url": data.get('rtsp_url'),
#         "status": "online",
#         "assigned_to": None,  # Track which packer this is assigned to
#         "created_at": datetime.now().isoformat()
#     }
    
#     cameras_db[camera_id] = camera_config
    
#     return jsonify({
#         "message": "Camera registered successfully",
#         "camera_id": camera_id,
#         "camera": camera_config
#     }), 201

# @camera_bp.route('', methods=['GET'])
# def list_cameras():
#     """Returns all registered cameras with assignment status"""
#     from routes.packer_routes import get_packers_db
    
#     # Get all packers to check assignments
#     packers_db = get_packers_db()
    
#     # Build a map of camera_id -> packer_name
#     camera_assignments = {}
#     for packer_id, packer_data in packers_db.items():
#         cam_id = packer_data.get('camera_id')
#         if cam_id:
#             camera_assignments[cam_id] = packer_data.get('name', 'Unknown Packer')
    
#     # Update cameras with assignment info
#     cameras_list = []
#     for cam_id, cam_data in cameras_db.items():
#         camera_info = cam_data.copy()
#         if cam_id in camera_assignments:
#             camera_info['assigned_to'] = camera_assignments[cam_id]
#             camera_info['is_assigned'] = True
#         else:
#             camera_info['assigned_to'] = None
#             camera_info['is_assigned'] = False
#         cameras_list.append(camera_info)
    
#     return jsonify({
#         "cameras": cameras_list,
#         "total": len(cameras_list)
#     }), 200

# @camera_bp.route('/<camera_id>', methods=['DELETE'])
# def delete_camera(camera_id):
#     """Removes a camera registration"""
#     if camera_id not in cameras_db:
#         return jsonify({"error": "Camera not found"}), 404
    
#     # Check if camera is assigned to any packer
#     from routes.packer_routes import get_packers_db
#     packers_db = get_packers_db()
    
#     for packer_id, packer_data in packers_db.items():
#         if packer_data.get('camera_id') == camera_id:
#             return jsonify({
#                 "error": "Camera is assigned to a packer",
#                 "message": f"Please unassign from '{packer_data.get('name')}' before deleting"
#             }), 400
    
#     del cameras_db[camera_id]
#     return jsonify({"message": "Camera removed successfully"}), 200

# @camera_bp.route('/<camera_id>/status', methods=['GET'])
# def get_camera_status(camera_id):
#     """Get camera assignment status"""
#     if camera_id not in cameras_db:
#         return jsonify({"error": "Camera not found"}), 404
    
#     from routes.packer_routes import get_packers_db
#     packers_db = get_packers_db()
    
#     assigned_packer = None
#     for packer_id, packer_data in packers_db.items():
#         if packer_data.get('camera_id') == camera_id:
#             assigned_packer = {
#                 "packer_id": packer_id,
#                 "packer_name": packer_data.get('name')
#             }
#             break
    
#     camera_data = cameras_db[camera_id].copy()
#     camera_data['assigned_packer'] = assigned_packer
#     camera_data['is_assigned'] = assigned_packer is not None
    
#     return jsonify(camera_data), 200

# def get_cameras_db():
#     """Helper to access cameras from other modules"""
#     return cameras_db



# """
# Live Monitoring Routes
# Handles real-time camera monitoring, video streaming, and metrics
# """

# from flask import Blueprint, request, jsonify, Response
# import cv2
# import threading
# import time
# from datetime import datetime
# import uuid

# # Create Blueprint
# monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/api/monitor')

# # Active monitoring sessions
# active_sessions = {}
# session_locks = {}

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
                
#             # Convert to JPEG for streaming
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 continue
#             frame_bytes = buffer.tobytes()
            
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @monitoring_bp.route('/video_feed/<session_id>')
# def video_feed(session_id):
#     """Route for React <img> tag"""
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
#     return Response(generate_frames(session_id),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @monitoring_bp.route('/start', methods=['POST'])
# def start_monitoring():
#     """
#     Start monitoring - accepts either packer_id OR camera_id
#     If camera_id is provided, it finds the associated packer
#     """
#     from routes.packer_routes import get_packers_db
#     from routes.camera_routes import get_cameras_db
#     from models.packer_monitor import PackerEfficiencyMonitor
#     from app import MODEL_PATH
    
#     data = request.json
#     packer_id = data.get('packer_id')
#     camera_id = data.get('camera_id')
    
#     # Must provide either packer_id or camera_id
#     if not packer_id and not camera_id:
#         return jsonify({
#             "error": "Missing parameter",
#             "message": "Please provide either 'packer_id' or 'camera_id'"
#         }), 400
    
#     packers_db = get_packers_db()
#     cameras_db = get_cameras_db()
    
#     # If camera_id is provided, find the packer it's assigned to
#     if camera_id and not packer_id:
#         # Look for packer with this camera_id
#         found_packer = None
#         for p_id, p_data in packers_db.items():
#             if p_data.get('camera_id') == camera_id:
#                 found_packer = p_id
#                 break
        
#         if not found_packer:
#             return jsonify({
#                 "error": "Camera not assigned",
#                 "message": "This camera is not assigned to any packer. Please assign it to a packer in the Packer Master page.",
#                 "camera_id": camera_id
#             }), 400
        
#         packer_id = found_packer
    
#     # Check if packer exists
#     if packer_id not in packers_db:
#         return jsonify({
#             "error": "Packer not found",
#             "message": f"No packer found with ID: {packer_id}",
#             "available_packers": list(packers_db.keys())
#         }), 404
    
#     packer_data = packers_db[packer_id]
    
#     # Check if packer is already being monitored
#     if packer_data.get('status') == 'active':
#         return jsonify({
#             "error": "Packer already active",
#             "message": f"Packer '{packer_data.get('name')}' is already being monitored",
#             "session_id": packer_data.get('session_id')
#         }), 400
    
#     # Get camera_id from packer configuration
#     assigned_camera_id = packer_data.get('camera_id')
    
#     # Validate camera assignment
#     if not assigned_camera_id:
#         return jsonify({
#             "error": "No camera assigned",
#             "message": f"Packer '{packer_data.get('name')}' does not have a camera assigned. Please assign a camera in Packer Master.",
#             "packer_id": packer_id,
#             "packer_name": packer_data.get('name')
#         }), 400
    
#     # Validate camera exists
#     if assigned_camera_id not in cameras_db:
#         return jsonify({
#             "error": "Invalid camera assignment",
#             "message": f"Camera ID '{assigned_camera_id}' assigned to packer does not exist. Please reassign a valid camera.",
#             "packer_id": packer_id,
#             "camera_id": assigned_camera_id
#         }), 400
    
#     # Get the RTSP URL from camera
#     camera_data = cameras_db[assigned_camera_id]
#     actual_source = camera_data.get('rtsp_url')
    
#     if not actual_source:
#         return jsonify({
#             "error": "Invalid camera configuration",
#             "message": f"Camera '{camera_data.get('name')}' does not have an RTSP URL configured",
#             "camera_id": assigned_camera_id
#         }), 400
    
#     # Create monitoring session
#     session_id = str(uuid.uuid4())
    
#     try:
#         # Initialize the PackerEfficiencyMonitor
#         monitor = PackerEfficiencyMonitor(
#             model_path=MODEL_PATH,
#             line_position=float(packer_data.get('line_position', 0.7)),
#             start_line_position=float(packer_data.get('start_line_position', 0.2)),
#             confidence_threshold=float(packer_data.get('confidence_threshold', 0.5)),
#             spouts=int(packer_data.get('spouts', 8))
#         )
        
#         # Try to connect to camera
#         cap = cv2.VideoCapture(actual_source)
#         if not cap.isOpened():
#             return jsonify({
#                 "error": "Camera connection failed",
#                 "message": f"Failed to connect to camera at: {actual_source}",
#                 "camera_name": camera_data.get('name'),
#                 "rtsp_url": actual_source
#             }), 400
            
#     except Exception as e:
#         return jsonify({
#             "error": "Monitor initialization failed",
#             "message": str(e),
#             "packer_id": packer_id
#         }), 500
    
#     # Create session and link to packer
#     active_sessions[session_id] = {
#         "session_id": session_id,
#         "packer_id": packer_id,
#         "camera_id": assigned_camera_id,
#         "monitor": monitor,
#         "capture": cap,
#         "status": "running",
#         "last_frame": None,
#         "started_at": datetime.now().isoformat()
#     }
#     session_locks[session_id] = threading.Lock()
    
#     # Update packer status
#     packer_data['status'] = 'active'
#     packer_data['session_id'] = session_id
#     packer_data['monitor'] = monitor
    
#     # Start processing thread
#     def process_stream():
#         session = active_sessions.get(session_id)
#         line_pos = packer_data.get('line_position', 0.7)
#         start_line_pos = packer_data.get('start_line_position', 0.2)
        
#         while session and session.get('status') == 'running':
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"Failed to read frame for session {session_id}")
#                 session['status'] = 'stopped'
#                 break
            
#             # Process frame for bag detection
#             session['monitor'].process_frame(frame)
            
#             # Draw detection lines
#             h, w = frame.shape[:2]
#             cv2.line(frame, (int(w * line_pos), 0), (int(w * line_pos), h), (0, 0, 255), 3)
#             cv2.line(frame, (int(w * start_line_pos), 0), (int(w * start_line_pos), h), (0, 255, 255), 2)
            
#             # Update frame for streaming
#             with session_locks[session_id]:
#                 session['last_frame'] = frame
            
#             time.sleep(0.01)
        
#         # Cleanup
#         cap.release()
#         packer_data['status'] = 'idle'
#         print(f"Monitoring stopped for session {session_id}")
    
#     threading.Thread(target=process_stream, daemon=True).start()
    
#     return jsonify({
#         "message": "Monitoring started successfully",
#         "session_id": session_id,
#         "packer_id": packer_id,
#         "packer_name": packer_data.get('name'),
#         "camera_id": assigned_camera_id,
#         "camera_name": camera_data.get('name'),
#         "source_used": actual_source
#     }), 201

# @monitoring_bp.route('/start-by-camera', methods=['POST'])
# def start_monitoring_by_camera():
#     """
#     LEGACY ENDPOINT: Start monitoring by camera_id
#     This redirects to the main start endpoint
#     """
#     data = request.json
#     camera_id = data.get('camera_id')
    
#     if not camera_id:
#         return jsonify({
#             "error": "Missing camera_id",
#             "message": "Please provide a camera_id"
#         }), 400
    
#     # Redirect to main start endpoint with camera_id
#     return start_monitoring()

# @monitoring_bp.route('/stop/<session_id>', methods=['POST'])
# def stop_monitoring(session_id):
#     """Stop monitoring session"""
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
    
#     session = active_sessions[session_id]
#     session['status'] = 'stopped'
    
#     # Update packer status
#     from routes.packer_routes import get_packers_db
#     packers_db = get_packers_db()
#     packer_id = session['packer_id']
    
#     if packer_id in packers_db:
#         packers_db[packer_id]['status'] = 'idle'
#         packers_db[packer_id]['session_id'] = None
#         packers_db[packer_id]['monitor'] = None
    
#     return jsonify({
#         "message": "Monitoring stopped successfully",
#         "session_id": session_id,
#         "packer_id": packer_id
#     }), 200

# @monitoring_bp.route('/metrics/<session_id>', methods=['GET'])
# def get_live_metrics(session_id):
#     """Get live metrics for active session"""
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
    
#     session = active_sessions[session_id]
#     return jsonify({
#         "session_id": session_id,
#         "packer_id": session['packer_id'],
#         "camera_id": session.get('camera_id'),
#         "status": session['status'],
#         "metrics": session['monitor'].get_summary(),
#         "started_at": session.get('started_at')
#     }), 200

# @monitoring_bp.route('/sessions', methods=['GET'])
# def list_active_sessions():
#     """List all active monitoring sessions"""
#     sessions = []
#     for session_id, session in active_sessions.items():
#         sessions.append({
#             "session_id": session_id,
#             "packer_id": session['packer_id'],
#             "camera_id": session.get('camera_id'),
#             "status": session['status'],
#             "started_at": session.get('started_at')
#         })
    
#     return jsonify({
#         "sessions": sessions,
#         "total": len(sessions)
#     }), 200

# @monitoring_bp.route('/status/<identifier>', methods=['GET'])
# def get_monitoring_status(identifier):
#     """
#     Get monitoring status by packer_id or camera_id
#     Returns active session info if monitoring is active
#     """
#     from routes.packer_routes import get_packers_db
    
#     packers_db = get_packers_db()
    
#     # Check if identifier is a packer_id
#     if identifier in packers_db:
#         packer_data = packers_db[identifier]
#         return jsonify({
#             "is_active": packer_data.get('status') == 'active',
#             "session_id": packer_data.get('session_id'),
#             "packer_id": identifier,
#             "camera_id": packer_data.get('camera_id')
#         }), 200
    
#     # Check if identifier is a camera_id
#     for packer_id, packer_data in packers_db.items():
#         if packer_data.get('camera_id') == identifier:
#             return jsonify({
#                 "is_active": packer_data.get('status') == 'active',
#                 "session_id": packer_data.get('session_id'),
#                 "packer_id": packer_id,
#                 "camera_id": identifier
#             }), 200
    
#     return jsonify({
#         "error": "Not found",
#         "message": "No packer or camera found with this identifier"
#     }), 404

# @monitoring_bp.route('/pause/<session_id>', methods=['POST'])
# def pause_monitoring(session_id):
#     """Pause monitoring session"""
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
    
#     session = active_sessions[session_id]
#     if session['status'] == 'running':
#         session['status'] = 'paused'
#         return jsonify({
#             "message": "Monitoring paused",
#             "session_id": session_id
#         }), 200
#     else:
#         return jsonify({
#             "error": "Session is not running",
#             "current_status": session['status']
#         }), 400

# @monitoring_bp.route('/resume/<session_id>', methods=['POST'])
# def resume_monitoring(session_id):
#     """Resume paused monitoring session"""
#     if session_id not in active_sessions:
#         return jsonify({"error": "Session not found"}), 404
    
#     session = active_sessions[session_id]
#     if session['status'] == 'paused':
#         session['status'] = 'running'
#         return jsonify({
#             "message": "Monitoring resumed",
#             "session_id": session_id
#         }), 200
#     else:
#         return jsonify({
#             "error": "Session is not paused",
#             "current_status": session['status']
#         }), 400



# """
# Camera Management Routes
# Handles registration of physical hardware (RTSP, IP Cameras)
# """

# from flask import Blueprint, request, jsonify
# from datetime import datetime
# import uuid

# # Create Blueprint
# camera_bp = Blueprint('camera', __name__, url_prefix='/api/cameras')

# # In-memory storage for testing (integrate with DB for production)
# cameras_db = {}

# @camera_bp.route('', methods=['POST'])
# def add_camera():
#     """
#     Register a new physical camera source
#     Request Body: { "name": "Cam-01", "rtsp_url": "rtsp://..." }
#     """
#     data = request.json
    
#     if not data.get('name') or not data.get('rtsp_url'):
#         return jsonify({"error": "Camera name and RTSP URL are required"}), 400

#     camera_id = str(uuid.uuid4())
#     camera_config = {
#         "id": camera_id,
#         "name": data.get('name'),
#         "rtsp_url": data.get('rtsp_url'),
#         "status": "online",
#         "created_at": datetime.now().isoformat()
#     }
    
#     cameras_db[camera_id] = camera_config
    
#     return jsonify({
#         "message": "Camera registered successfully",
#         "camera_id": camera_id,
#         "camera": camera_config
#     }), 201

# @camera_bp.route('', methods=['GET'])
# def list_cameras():
#     """Returns all registered cameras with assignment status"""
#     from routes.packer_routes import get_packers_db
    
#     # Get all packers to check assignments
#     packers_db = get_packers_db()
    
#     # Build a map of camera_id -> packer info
#     camera_assignments = {}
#     for packer_id, packer_data in packers_db.items():
#         cam_id = packer_data.get('camera_id')
#         if cam_id:
#             camera_assignments[cam_id] = {
#                 "packer_id": packer_id,
#                 "packer_name": packer_data.get('name', 'Unknown Packer'),
#                 "is_active": packer_data.get('status') == 'active'
#             }
    
#     # Update cameras with assignment info
#     cameras_list = []
#     for cam_id, cam_data in cameras_db.items():
#         camera_info = cam_data.copy()
        
#         if cam_id in camera_assignments:
#             assignment = camera_assignments[cam_id]
#             camera_info['assigned_to'] = assignment['packer_name']
#             camera_info['packer_id'] = assignment['packer_id']
#             camera_info['is_assigned'] = True
#             camera_info['is_monitoring'] = assignment['is_active']
#         else:
#             camera_info['assigned_to'] = None
#             camera_info['packer_id'] = None
#             camera_info['is_assigned'] = False
#             camera_info['is_monitoring'] = False
        
#         cameras_list.append(camera_info)
    
#     return jsonify({
#         "cameras": cameras_list,
#         "total": len(cameras_list)
#     }), 200

# @camera_bp.route('/<camera_id>', methods=['GET'])
# def get_camera(camera_id):
#     """Get specific camera details"""
#     if camera_id not in cameras_db:
#         return jsonify({"error": "Camera not found"}), 404
    
#     from routes.packer_routes import get_packers_db
#     packers_db = get_packers_db()
    
#     camera_data = cameras_db[camera_id].copy()
    
#     # Find assigned packer
#     assigned_packer = None
#     for packer_id, packer_data in packers_db.items():
#         if packer_data.get('camera_id') == camera_id:
#             assigned_packer = {
#                 "packer_id": packer_id,
#                 "packer_name": packer_data.get('name'),
#                 "status": packer_data.get('status'),
#                 "is_active": packer_data.get('status') == 'active',
#                 "session_id": packer_data.get('session_id')
#             }
#             break
    
#     camera_data['assigned_packer'] = assigned_packer
#     camera_data['is_assigned'] = assigned_packer is not None
#     camera_data['can_start_monitoring'] = assigned_packer is not None and not assigned_packer['is_active']
    
#     return jsonify(camera_data), 200

# @camera_bp.route('/<camera_id>', methods=['PUT'])
# def update_camera(camera_id):
#     """Update camera configuration"""
#     if camera_id not in cameras_db:
#         return jsonify({"error": "Camera not found"}), 404
    
#     data = request.json
#     camera_data = cameras_db[camera_id]
    
#     # Check if camera is currently in use
#     from routes.packer_routes import get_packers_db
#     packers_db = get_packers_db()
    
#     for packer_id, packer_data in packers_db.items():
#         if packer_data.get('camera_id') == camera_id and packer_data.get('status') == 'active':
#             return jsonify({
#                 "error": "Camera in use",
#                 "message": f"Cannot update camera while it's being used by '{packer_data.get('name')}'"
#             }), 400
    
#     # Update fields
#     if 'name' in data:
#         camera_data['name'] = data['name']
#     if 'rtsp_url' in data:
#         camera_data['rtsp_url'] = data['rtsp_url']
#     if 'status' in data:
#         camera_data['status'] = data['status']
    
#     camera_data['updated_at'] = datetime.now().isoformat()
    
#     return jsonify({
#         "message": "Camera updated successfully",
#         "camera": camera_data
#     }), 200

# @camera_bp.route('/<camera_id>', methods=['DELETE'])
# def delete_camera(camera_id):
#     """Removes a camera registration"""
#     if camera_id not in cameras_db:
#         return jsonify({"error": "Camera not found"}), 404
    
#     # Check if camera is assigned to any packer
#     from routes.packer_routes import get_packers_db
#     packers_db = get_packers_db()
    
#     for packer_id, packer_data in packers_db.items():
#         if packer_data.get('camera_id') == camera_id:
#             return jsonify({
#                 "error": "Camera is assigned to a packer",
#                 "message": f"Please unassign from '{packer_data.get('name')}' before deleting",
#                 "packer_id": packer_id,
#                 "packer_name": packer_data.get('name')
#             }), 400
    
#     del cameras_db[camera_id]
#     return jsonify({"message": "Camera removed successfully"}), 200

# @camera_bp.route('/<camera_id>/status', methods=['GET'])
# def get_camera_status(camera_id):
#     """Get camera assignment and monitoring status"""
#     if camera_id not in cameras_db:
#         return jsonify({"error": "Camera not found"}), 404
    
#     from routes.packer_routes import get_packers_db
#     packers_db = get_packers_db()
    
#     assigned_packer = None
#     is_monitoring = False
    
#     for packer_id, packer_data in packers_db.items():
#         if packer_data.get('camera_id') == camera_id:
#             assigned_packer = {
#                 "packer_id": packer_id,
#                 "packer_name": packer_data.get('name'),
#                 "status": packer_data.get('status')
#             }
#             is_monitoring = packer_data.get('status') == 'active'
#             break
    
#     camera_data = cameras_db[camera_id].copy()
#     camera_data['assigned_packer'] = assigned_packer
#     camera_data['is_assigned'] = assigned_packer is not None
#     camera_data['is_monitoring'] = is_monitoring
#     camera_data['can_start_monitoring'] = assigned_packer is not None and not is_monitoring
    
#     return jsonify(camera_data), 200

# @camera_bp.route('/<camera_id>/start-monitoring', methods=['POST'])
# def start_camera_monitoring(camera_id):
#     """
#     Convenience endpoint to start monitoring from camera
#     Finds the assigned packer and starts monitoring
#     """
#     if camera_id not in cameras_db:
#         return jsonify({"error": "Camera not found"}), 404
    
#     from routes.packer_routes import get_packers_db
#     packers_db = get_packers_db()
    
#     # Find packer assigned to this camera
#     packer_id = None
#     for p_id, p_data in packers_db.items():
#         if p_data.get('camera_id') == camera_id:
#             packer_id = p_id
#             break
    
#     if not packer_id:
#         return jsonify({
#             "error": "Camera not assigned",
#             "message": "This camera is not assigned to any packer. Please assign it in Packer Master.",
#             "camera_id": camera_id
#         }), 400
    
#     # Forward to monitoring start endpoint
#     from flask import current_app
#     with current_app.test_request_context(
#         '/api/monitor/start',
#         method='POST',
#         json={'camera_id': camera_id}
#     ):
#         from routes.monitoring_routes import start_monitoring
#         return start_monitoring()

# def get_cameras_db():
#     """Helper to access cameras from other modules"""
#     return cameras_db

"""
Camera Routes - SQLite Version with Video File Support
Handles camera (RTSP/Video) management
"""

from flask import Blueprint, jsonify, request
import uuid
from database import get_db_connection

camera_bp = Blueprint('camera', __name__, url_prefix='/api/cameras')

# In-memory cache (populated from DB on first access)
cameras_db = {}
cameras_loaded = False

def get_cameras_db():
    """Load cameras from SQLite into memory if not already loaded"""
    global cameras_db, cameras_loaded
    if not cameras_loaded:
        conn = get_db_connection()
        # Fetch all cameras from the database
        rows = conn.execute('SELECT * FROM cameras').fetchall()
        conn.close()
        
        cameras_db.clear()
        for row in rows:
            # CRITICAL FIX: Explicitly convert the sqlite3.Row to a dictionary
            cameras_db[row['id']] = dict(row) 
            
        cameras_loaded = True
    return cameras_db

def refresh_cameras_cache():
    """Force reload from DB"""
    global cameras_loaded
    cameras_loaded = False
    return get_cameras_db()

# def get_cameras_db():
#     """Load cameras from SQLite into memory if not already loaded"""
#     global cameras_db, cameras_loaded
#     if not cameras_loaded:
#         conn = get_db_connection()
#         rows = conn.execute('SELECT * FROM cameras').fetchall()
#         conn.close()
        
#         cameras_db.clear()
#         for row in rows:
#             cameras_db[row['id']] = {
#                 'id': row['id'],
#                 'name': row['name'],
#                 'rtsp_url': row['rtsp_url'],
#                 'is_video_file': bool(row.get('is_video_file', 0)),
#                 'packer_id': row['packer_id']
#             }
#         cameras_loaded = True
#     return cameras_db

def refresh_cameras_cache():
    """Force reload from DB"""
    global cameras_loaded
    cameras_loaded = False
    return get_cameras_db()
@camera_bp.route('', methods=['GET'])
def list_cameras():
    """Get all cameras with assignment status for UI mapping"""
    conn = get_db_connection()
    
    # Use a JOIN to get the name of the packer assigned to the camera
    query = """
        SELECT c.*, p.name as packer_name
        FROM cameras c
        LEFT JOIN packers p ON c.packer_id = p.id
    """
    
    rows = conn.execute(query).fetchall()
    conn.close()
    
    formatted = []
    for row in rows:
        cam = dict(row) # Ensure it is a dictionary
        formatted.append({
            'id': cam['id'],
            'name': cam['name'],
            'rtsp_url': cam['rtsp_url'],
            'is_video_file': bool(cam.get('is_video_file', 0)),
            'packer_id': cam['packer_id'],
            # These fields drive the "Assigned Packer" column in CameraMaster.jsx
            'is_assigned': cam['packer_id'] is not None,
            'assigned_to': cam['packer_name'] if cam['packer_id'] else None
        })
    
    return jsonify({'cameras': formatted}), 200

# @camera_bp.route('', methods=['POST'])
# def add_camera():
#     """Add a new camera or video source"""
#     data = request.json
    
#     if not data.get('name') or not data.get('rtsp_url'):
#         return jsonify({'error': 'name and rtsp_url required'}), 400
    
#     camera_id = str(uuid.uuid4())
#     is_video_file = data.get('is_video_file', False)
    
#     try:
#         conn = get_db_connection()
#         conn.execute('''
#             INSERT INTO cameras (id, name, rtsp_url, is_video_file, packer_id)
#             VALUES (?, ?, ?, ?, ?)
#         ''', (
#             camera_id,
#             data['name'],
#             data['rtsp_url'],
#             1 if is_video_file else 0,
#             data.get('packer_id')
#         ))
#         conn.commit()
        
#         # If assigned to packer, update packer's camera_id
#         if data.get('packer_id'):
#             conn.execute('''
#                 UPDATE packers SET camera_id = ? WHERE id = ?
#             ''', (camera_id, data['packer_id']))
#             conn.commit()
        
#         conn.close()
        
#         # Refresh cache
#         refresh_cameras_cache()
        
#         return jsonify({
#             'message': 'Camera added successfully',
#             'camera_id': camera_id
#         }), 201
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@camera_bp.route('', methods=['POST'])
def add_camera():
    """Add a new camera or video source"""
    data = request.json
    
    # Validation for core fields
    if not data.get('name') or not data.get('rtsp_url'):
        return jsonify({'error': 'name and rtsp_url required'}), 400
    
    camera_id = str(uuid.uuid4())
    is_video_file = data.get('is_video_file', False)
    
    # Normalize packer_id: convert empty strings from UI to None for DB NULL
    packer_id = data.get('packer_id')
    if packer_id == "" or not packer_id:
        packer_id = None
    
    try:
        conn = get_db_connection()
        
        # 1. Insert into cameras table
        conn.execute('''
            INSERT INTO cameras (id, name, rtsp_url, is_video_file, packer_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            camera_id,
            data['name'],
            data['rtsp_url'],
            1 if is_video_file else 0,
            packer_id
        ))
        
        # 2. Sync logic: If assigned to a packer, update that packer's camera link
        if packer_id:
            conn.execute('''
                UPDATE packers SET camera_id = ? WHERE id = ?
            ''', (camera_id, packer_id))
            
        conn.commit()
        conn.close()
        
        # Refresh the in-memory cache used by other routes
        refresh_cameras_cache()
        
        return jsonify({
            'message': 'Source added successfully',
            'camera_id': camera_id,
            'is_video_file': is_video_file
        }), 201
        
    except Exception as e:
        # Log the actual error for debugging
        print(f"Error in add_camera: {e}")
        return jsonify({'error': str(e)}), 500
    
    
@camera_bp.route('/<camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Get camera details"""
    conn = get_db_connection()
    camera = conn.execute('SELECT * FROM cameras WHERE id = ?', (camera_id,)).fetchone()
    conn.close()
    
    if not camera:
        return jsonify({'error': 'Camera not found'}), 404
    
    return jsonify({
        'id': camera['id'],
        'name': camera['name'],
        'rtsp_url': camera['rtsp_url'],
        'is_video_file': bool(camera.get('is_video_file', 0)),
        'packer_id': camera['packer_id']
    }), 200

@camera_bp.route('/<camera_id>', methods=['PUT'])
def update_camera(camera_id):
    """Update camera details"""
    data = request.json
    
    conn = get_db_connection()
    camera = conn.execute('SELECT * FROM cameras WHERE id = ?', (camera_id,)).fetchone()
    
    if not camera:
        conn.close()
        return jsonify({'error': 'Camera not found'}), 404
    
    try:
        conn.execute('''
            UPDATE cameras 
            SET name = ?, rtsp_url = ?, is_video_file = ?, packer_id = ?
            WHERE id = ?
        ''', (
            data.get('name', camera['name']),
            data.get('rtsp_url', camera['rtsp_url']),
            1 if data.get('is_video_file', camera.get('is_video_file', 0)) else 0,
            data.get('packer_id', camera['packer_id']),
            camera_id
        ))
        conn.commit()
        conn.close()
        
        refresh_cameras_cache()
        
        return jsonify({'message': 'Camera updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@camera_bp.route('/<camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Delete a camera"""
    conn = get_db_connection()
    
    # Unlink from any packers first
    conn.execute('UPDATE packers SET camera_id = NULL WHERE camera_id = ?', (camera_id,))
    
    # Delete camera
    result = conn.execute('DELETE FROM cameras WHERE id = ?', (camera_id,))
    
    if result.rowcount == 0:
        conn.close()
        return jsonify({'error': 'Camera not found'}), 404
    
    conn.commit()
    conn.close()
    
    refresh_cameras_cache()
    
    return jsonify({'message': 'Camera deleted successfully'}), 200