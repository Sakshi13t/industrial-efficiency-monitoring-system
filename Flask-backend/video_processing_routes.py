"""
Video Processing Routes
Handles video upload and batch processing
"""

from flask import Blueprint, request, jsonify, send_file
import os
import uuid
import cv2
import json
import threading
from datetime import datetime
from werkzeug.utils import secure_filename

# Create Blueprint
video_bp = Blueprint('video', __name__, url_prefix='/api/process')

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Processing jobs storage
processing_jobs = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@video_bp.route('/upload', methods=['POST'])
def upload_video():
    """
    Upload video file for processing - packer_id is now optional
    to allow uploading test files before creating a packer entry.
    """
    from routes.packer_routes import get_packers_db
    
    # 1. Basic File Validation
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(video_file.filename):
        return jsonify({
            "error": "Invalid file format",
            "allowed_formats": list(ALLOWED_EXTENSIONS)
        }), 400
    
    # 2. Flexible Packer ID Validation
    packer_id = request.form.get('packer_id')
    
    # If a packer_id is provided, verify it; otherwise, ignore it
    if packer_id and packer_id != "" and packer_id != "undefined":
        packers_db = get_packers_db()
        if packer_id not in packers_db:
            # We will log a warning instead of erroring out to allow test uploads
            print(f"Warning: Uploaded video with unknown packer_id: {packer_id}")
    
    # 3. Save File
    video_id = str(uuid.uuid4())
    original_filename = secure_filename(video_file.filename)
    file_extension = original_filename.rsplit('.', 1)[1].lower()
    saved_filename = f"{video_id}.{file_extension}"
    video_path = os.path.join(UPLOAD_FOLDER, saved_filename)
    
    try:
        video_file.save(video_path)
        
        # Verify video can be opened by OpenCV
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
        
        file_size = os.path.getsize(video_path)
        
        return jsonify({
            "message": "Video uploaded successfully",
            "video_id": video_id,
            "filename": saved_filename,
            "original_filename": original_filename,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "video_info": {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2)
            },
            "packer_id": packer_id if packer_id else None
        }), 201
        
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({"error": "Upload failed", "message": str(e)}), 500
    
# @video_bp.route('/upload', methods=['POST'])
# def upload_video():
#     """
#     Upload video file for processing
    
#     Form Data:
#         video: Video file
#         packer_id: UUID of packer configuration
#         description: Optional description
    
#     Returns:
#         Upload confirmation with video_id
#     """
#     from routes.packer_routes import get_packers_db
    
#     # Check if video file is in request
#     if 'video' not in request.files:
#         return jsonify({"error": "No video file provided"}), 400
    
#     video_file = request.files['video']
    
#     if video_file.filename == '':
#         return jsonify({"error": "No file selected"}), 400
    
#     if not allowed_file(video_file.filename):
#         return jsonify({
#             "error": "Invalid file format",
#             "allowed_formats": list(ALLOWED_EXTENSIONS)
#         }), 400
    
#     # Get packer_id
#     packer_id = request.form.get('packer_id')
#     if not packer_id:
#         return jsonify({"error": "packer_id is required"}), 400
    
#     packers_db = get_packers_db()
#     if packer_id not in packers_db:
#         return jsonify({"error": "Invalid packer_id"}), 400
    
#     # Generate unique video ID
#     video_id = str(uuid.uuid4())
    
#     # Secure filename and save
#     original_filename = secure_filename(video_file.filename)
#     file_extension = original_filename.rsplit('.', 1)[1].lower()
#     saved_filename = f"{video_id}.{file_extension}"
#     video_path = os.path.join(UPLOAD_FOLDER, saved_filename)
    
#     try:
#         video_file.save(video_path)
        
#         # Get video info
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             os.remove(video_path)
#             return jsonify({"error": "Invalid video file"}), 400
        
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         duration = total_frames / fps if fps > 0 else 0
        
#         cap.release()
        
#         # Get file size
#         file_size = os.path.getsize(video_path)
        
#         return jsonify({
#             "message": "Video uploaded successfully",
#             "video_id": video_id,
#             "filename": saved_filename,
#             "original_filename": original_filename,
#             "file_size_bytes": file_size,
#             "file_size_mb": round(file_size / (1024 * 1024), 2),
#             "video_info": {
#                 "width": width,
#                 "height": height,
#                 "fps": fps,
#                 "total_frames": total_frames,
#                 "duration_seconds": round(duration, 2)
#             },
#             "packer_id": packer_id
#         }), 201
        
#     except Exception as e:
#         if os.path.exists(video_path):
#             os.remove(video_path)
#         return jsonify({
#             "error": "Failed to upload video",
#             "message": str(e)
#         }), 500


@video_bp.route('/start', methods=['POST'])
def start_processing():
    """
    Start video processing job
    
    Request Body:
        {
            "video_id": "uuid",
            "packer_id": "uuid",
            "save_output_video": true/false,
            "generate_report": true/false
        }
    
    Returns:
        Job info with job_id
    """
    from routes.packer_routes import get_packers_db
    from models.packer_monitor import PackerEfficiencyMonitor
    from app import MODEL_PATH
    
    data = request.json
    video_id = data.get('video_id')
    packer_id = data.get('packer_id')
    
    if not video_id or not packer_id:
        return jsonify({"error": "video_id and packer_id are required"}), 400
    
    packers_db = get_packers_db()
    
    if packer_id in packers_db:
        packers_db[packer_id]['status'] = 'active'
    else:
        return jsonify({"error": "Invalid packer_id"}), 400
    
    # Find video file
    video_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(video_id)]
    if not video_files:
        return jsonify({"error": "Video not found"}), 404
    
    video_path = os.path.join(UPLOAD_FOLDER, video_files[0])
    
    # Check if video exists
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    # Create job
    job_id = str(uuid.uuid4())
    
    packer_config = packers_db[packer_id]
    save_output = data.get('save_output_video', False)
    generate_report = data.get('generate_report', True)
    
    # Initialize job
    processing_jobs[job_id] = {
        "job_id": job_id,
        "packer_id": packer_id,
        "packer_name": packer_config.get('name'),
        "video_id": video_id,
        "status": "queued",
        "progress": 0,
        "frames_processed": 0,
        "total_frames": 0,
        "created_at": datetime.now().isoformat(),
        "save_output": save_output,
        "generate_report": generate_report
    }
    
    # Process video in background thread
    def process_video():
        try:
            processing_jobs[job_id]['status'] = 'processing'
            processing_jobs[job_id]['started_at'] = datetime.now().isoformat()
            
            # Create monitor
            monitor = PackerEfficiencyMonitor(
                model_path=MODEL_PATH,
                line_position=packer_config.get('line_position', 0.7),
                start_line_position=packer_config.get('start_line_position', 0.2),
                confidence_threshold=packer_config.get('confidence_threshold', 0.5),
                spouts=packer_config.get('spouts', 8)
            )
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                processing_jobs[job_id]['status'] = 'failed'
                processing_jobs[job_id]['error'] = 'Could not open video'
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            processing_jobs[job_id]['total_frames'] = total_frames
            
            # Setup output video writer if needed
            output_writer = None
            output_path = None
            
            if save_output:
                output_filename = f"output_{job_id}.mp4"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process frames
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = monitor.process_frame(frame)
                
                # Write to output if needed
                if output_writer:
                    output_writer.write(processed_frame)
                
                frame_count += 1
                
                # Update progress
                progress = (frame_count / total_frames) * 100
                processing_jobs[job_id]['progress'] = round(progress, 2)
                processing_jobs[job_id]['frames_processed'] = frame_count
                
                if frame_count % 30 == 0 or frame_count == total_frames:
                    processing_jobs[job_id]['summary'] = monitor.get_summary()
            
            # Cleanup
            cap.release()
            if output_writer:
                output_writer.release()
            
            # Get final summary
            summary = monitor.get_summary()
            
            processing_jobs[job_id]['status'] = 'completed'
            processing_jobs[job_id]['summary'] = summary
            processing_jobs[job_id]['completed_at'] = datetime.now().isoformat()
            
            if save_output:
                processing_jobs[job_id]['output_video'] = output_filename
            
            # Generate report if requested
            if generate_report:
                report_filename = f"report_{job_id}.json"
                report_path = os.path.join(REPORTS_FOLDER, report_filename)
                
                report_data = {
                    "job_id": job_id,
                    "packer_id": packer_id,
                    "packer_name": packer_config.get('name'),
                    "video_id": video_id,
                    "timestamp": datetime.now().isoformat(),
                    "video_info": {
                        "total_frames": total_frames,
                        "fps": fps,
                        "width": width,
                        "height": height
                    },
                    "summary": summary,
                    "configuration": {
                        "line_position": packer_config.get('line_position'),
                        "start_line_position": packer_config.get('start_line_position'),
                        "confidence_threshold": packer_config.get('confidence_threshold')
                    }
                }
                
                with open(report_path, 'w') as f:
                    json.dump(report_data, f, indent=2)
                
                processing_jobs[job_id]['report_file'] = report_filename
            
        except Exception as e:
            processing_jobs[job_id]['status'] = 'failed'
            processing_jobs[job_id]['error'] = str(e)
            processing_jobs[job_id]['failed_at'] = datetime.now().isoformat()
    
    # Start processing thread
    thread = threading.Thread(target=process_video)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "message": "Processing started",
        "job_id": job_id,
        "status": "queued"
    }), 202


@video_bp.route('/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Get processing job status and progress
    
    Args:
        job_id: UUID of processing job
    
    Returns:
        Job status, progress, and metrics
    """
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(processing_jobs[job_id]), 200


@video_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """
    List all processing jobs
    
    Query Parameters:
        status: Filter by status (queued/processing/completed/failed)
        limit: Limit number of results
    
    Returns:
        List of jobs
    """
    status_filter = request.args.get('status')
    limit = request.args.get('limit', type=int)
    
    jobs = list(processing_jobs.values())
    
    # Filter by status
    if status_filter:
        jobs = [j for j in jobs if j.get('status') == status_filter]
    
    # Sort by created_at (newest first)
    jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    # Limit results
    if limit:
        jobs = jobs[:limit]
    
    return jsonify({
        "jobs": jobs,
        "total": len(jobs)
    }), 200


@video_bp.route('/download/<job_id>', methods=['GET'])
def download_output(job_id):
    """
    Download processed output video
    
    Args:
        job_id: UUID of processing job
    
    Returns:
        Video file download
    """
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = processing_jobs[job_id]
    
    if job.get('status') != 'completed':
        return jsonify({"error": "Job not completed"}), 400
    
    if not job.get('output_video'):
        return jsonify({"error": "No output video available"}), 404
    
    output_path = os.path.join(OUTPUT_FOLDER, job['output_video'])
    
    if not os.path.exists(output_path):
        return jsonify({"error": "Output file not found"}), 404
    
    return send_file(output_path, as_attachment=True)

@video_bp.route('/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """
    Cancel a processing job
    
    Args:
        job_id: UUID of processing job
    
    Returns:
        Cancellation confirmation
    """
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = processing_jobs[job_id]
    
    if job.get('status') in ['completed', 'failed']:
        return jsonify({"error": "Cannot cancel completed or failed job"}), 400
    
    processing_jobs[job_id]['status'] = 'cancelled'
    processing_jobs[job_id]['cancelled_at'] = datetime.now().isoformat()
    
    return jsonify({
        "message": "Job cancelled",
        "job_id": job_id
    }), 200