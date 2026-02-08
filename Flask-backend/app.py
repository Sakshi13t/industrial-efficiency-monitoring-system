"""
Flask API Backend for Packer Efficiency Monitor
Main Application File - Integrates all route modules
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_mail import Mail, Message
import time
import os
import sqlite3
from database import init_db, get_db_connection

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize the DB file and tables on startup
init_db()

def reset_stale_statuses():
    """Reset all packers to idle on server startup"""
    try:
        conn = get_db_connection()
        conn.execute("UPDATE packers SET status = 'idle', session_id = NULL")
        conn.commit()
        conn.close()
        print("✓ System Check: All packer statuses reset to idle.")
    except Exception as e:
        print(f"✗ Startup Reset Error: {e}")

# Call the reset function immediately after init_db()
reset_stale_statuses()

# --- EMAIL CONFIGURATION ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'sakshitandon1193@gmail.com'
app.config['MAIL_PASSWORD'] = 'xpye gnab pfkm ctna'
app.config['MAIL_DEFAULT_SENDER'] = 'sakshi.tandon@amzbizsol.in'

mail = Mail(app)

@app.route('/api/send_feedback', methods=['POST'])
def send_feedback():
    data = request.json
    overall = data.get('overallExperience', 0)
    ease = data.get('easeOfUse', 0)
    performance = data.get('applicationPerformance', 0)
    comments = data.get('comments', "")

    try:
        msg = Message(
            subject="New PackerVision AI Feedback",
            recipients=["recipient-sakshitandon1193@gmail.com"],
            body=f"""
            New Feedback Received:
            Overall Experience: {overall}/5 Stars
            Ease of Use: {ease}/5 Stars
            Application Performance: {performance}/5 Stars
            User Comments:
            {comments}
            """
        )
        mail.send(msg)
        return jsonify({"status": "success", "message": "Feedback sent via email"}), 200
    except Exception as e:
        print(f"Mail Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['REPORTS_FOLDER'] = 'reports'

# Model configuration
MODEL_PATH = 'best.pt'

# Application start time (for uptime tracking)
app_start_time = time.time()

# Create necessary folders
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('evidence', exist_ok=True)

# Import Blueprints
from routes.dashboard_routes import dashboard_bp
from routes.packer_routes import packer_bp
from routes.monitoring_routes import monitoring_bp
from routes.video_processing_routes import video_bp
from routes.reports_routes import reports_bp
from routes.camera_routes import camera_bp
from routes.auth_routes import auth_bp

# Register Blueprints
app.register_blueprint(dashboard_bp)
app.register_blueprint(auth_bp)  
app.register_blueprint(packer_bp)
app.register_blueprint(monitoring_bp)
app.register_blueprint(video_bp)
app.register_blueprint(reports_bp)
app.register_blueprint(camera_bp)


# --- FIXED REPORTS ENDPOINT (PAGINATION SUPPORT) ---
@app.route('/api/reports', methods=['GET'])
def get_reports_paginated():
    """
    Fetch reports with server-side pagination, filtering, and sorting.
    """
    # Get Query Parameters
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 50))
    sort_order = request.args.get('sort', 'newest')
    date_from = request.args.get('from')
    date_to = request.args.get('to')
    search = request.args.get('search', '')
   
    # Build SQL Query
    query = "SELECT * FROM events WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM events WHERE 1=1"
    params = []

    # Apply Date Filters
    if date_from:
        query += " AND date(timestamp) >= ?"
        count_query += " AND date(timestamp) >= ?"
        params.append(date_from)
   
    if date_to:
        query += " AND date(timestamp) <= ?"
        count_query += " AND date(timestamp) <= ?"
        params.append(date_to)

    # Apply Search
    if search:
        query += " AND camera_name LIKE ?"
        count_query += " AND camera_name LIKE ?"
        params.append(f"%{search}%")

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
   
    try:
        # Get Total Count
        total_records = conn.execute(count_query, params).fetchone()[0]

        # Apply Sorting
        if sort_order == 'oldest':
            query += " ORDER BY timestamp ASC"
        else:
            query += " ORDER BY timestamp DESC"

        # Apply Pagination
        offset = (page - 1) * limit
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        # Execute Query
        events = conn.execute(query, params).fetchall()

        # Map Data
        reports_data = []
        for e in events:
            reports_data.append({
                "id": str(e['id']),
                "created_at": e['timestamp'],
                "packer_name": e['camera_name'],
                "location": "Zone A",
                "summary": {
                    "total_events": 1,
                    "total_cycles": 0,
                    "packer_efficiency": 100,
                    "manual_efficiency": 100,
                    "bags_placed": 0,
                    "stuck_bags": 0,
                    "bags_missed": 0
                }
            })

        return jsonify({
            "reports": reports_data,
            "total": total_records,
            "total_pages": (total_records + limit - 1) // limit,
            "current_page": page
        })

    except Exception as e:
        print(f"Error fetching reports: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# Root endpoint
@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        "service": "PackerPro Efficiency Monitor API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "dashboard": "/api/dashboard",
            "packers": "/api/packers",
            "monitoring": "/api/monitor",
            "video_processing": "/api/process",
            "reports": "/api/reports",
            "cameras": "/api/cameras"
        },
        "documentation": "/api/docs"
    }), 200


@app.route('/api/static/evidence/<session_id>/<filename>')
def serve_evidence(session_id, filename):
    """Serve evidence images"""
    evidence_dir = os.path.join('evidence', session_id)
   
    if not os.path.exists(evidence_dir):
        return jsonify({"error": "Evidence directory not found"}), 404
   
    try:
        return send_from_directory(evidence_dir, filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check"""
    uptime_seconds = time.time() - app_start_time
   
    return jsonify({
        "status": "healthy",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "uptime_seconds": round(uptime_seconds, 2),
        "uptime_hours": round(uptime_seconds / 3600, 2),
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH
    }), 200


# API documentation endpoint
@app.route('/api/docs', methods=['GET'])
def api_docs():
    """API documentation"""
    return jsonify({
        "title": "PackerPro API Documentation",
        "version": "1.0.0",
        "description": "REST API for Packer Efficiency Monitoring System",
        "base_url": "http://localhost:5000/api",
        "endpoints": {
            "Dashboard": {
                "GET /api/dashboard/stats": "Get dashboard statistics",
                "GET /api/dashboard/recent-reports": "Get recent reports",
                "GET /api/dashboard/overview": "Get comprehensive overview",
                "GET /api/dashboard/performance-comparison": "Get comparison data for bar charts"
            },
            "Packer Management": {
                "GET /api/packers": "List all packers",
                "POST /api/packers": "Create new packer",
                "GET /api/packers/{id}": "Get packer details",
                "PUT /api/packers/{id}": "Update packer",
                "DELETE /api/packers/{id}": "Delete packer",
                "GET /api/packers/count": "Get packer count"
            },
            "Live Monitoring": {
                "POST /api/monitor/start": "Start monitoring",
                "POST /api/monitor/stop/{id}": "Stop monitoring",
                "GET /api/monitor/metrics/{id}": "Get live metrics",
                "GET /api/monitor/active-sessions": "List active sessions",
                "GET /api/monitor/test-camera/{id}": "Test camera connection"
            },
            "Video Processing": {
                "POST /api/process/upload": "Upload video",
                "POST /api/process/start": "Start processing",
                "GET /api/process/status/{job_id}": "Get job status",
                "GET /api/process/jobs": "List all jobs",
                "GET /api/process/download/{job_id}": "Download output",
                "POST /api/process/cancel/{job_id}": "Cancel job"
            },
            "Reports": {
                "GET /api/reports": "List all reports (Paginated)",
                "GET /api/reports/{id}": "Get report details",
                "DELETE /api/reports/{id}": "Delete report",
                "GET /api/reports/{id}/download": "Download report"
            },
            "Cameras": {
                "GET /api/cameras": "List all cameras",
                "POST /api/cameras": "Add new camera",
                "PUT /api/cameras/{id}": "Update camera",
                "DELETE /api/cameras/{id}": "Delete camera"
            }
        }
    }), 200


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested resource was not found",
        "status": 404
    }), 404


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "error": "Bad Request",
        "message": "The request was invalid or cannot be served",
        "status": 400
    }), 400


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "message": "An internal server error occurred",
        "status": 500
    }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "File Too Large",
        "message": "The uploaded file exceeds the maximum allowed size (500MB)",
        "status": 413
    }), 413


# Run application
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("PackerPro Efficiency Monitor - Backend API")
    print("=" * 60)
    print(f"Server starting on: http://localhost:5000")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {'✓' if os.path.exists(MODEL_PATH) else '✗'}")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  Dashboard:     /api/dashboard")
    print("  Packers:       /api/packers")
    print("  Cameras:       /api/cameras")
    print("  Monitoring:    /api/monitor")
    print("  Processing:    /api/process")
    print("  Reports:       /api/reports")
    print("  Health Check:  /api/health")
    print("  Documentation: /api/docs")
    print("=" * 60)
    print("\nDiagnostic Tools:")
    print("  Test Camera:   GET /api/monitor/test-camera/<camera_id>")
    print("  Active Sessions: GET /api/monitor/active-sessions")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server\n")
   
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )

# """
# Flask API Backend for Packer Efficiency Monitor
# Main Application File - Integrates all route modules
# """

# from flask import Flask, jsonify, request, send_from_directory
# from flask_cors import CORS
# from flask_mail import Mail, Message
# import time
# import os
# import sqlite3
# from database import init_db, get_db_connection

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Initialize the DB file and tables on startup
# init_db()

# def reset_stale_statuses():
#     """Reset all packers to idle on server startup"""
#     try:
#         conn = get_db_connection()
#         conn.execute("UPDATE packers SET status = 'idle', session_id = NULL")
#         conn.commit()
#         conn.close()
#         print("System Check: All packer statuses reset to idle.")
#     except Exception as e:
#         print(f"Startup Reset Error: {e}")

# # Call the reset function immediately after init_db()
# reset_stale_statuses()

# # --- EMAIL CONFIGURATION ---
# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'sakshitandon1193@gmail.com' # Your email
# app.config['MAIL_PASSWORD'] = 'xpye gnab pfkm ctna'    # Your Gmail App Password
# app.config['MAIL_DEFAULT_SENDER'] = 'sakshi.tandon@amzbizsol.in'

# mail = Mail(app)

# @app.route('/api/send_feedback', methods=['POST'])
# def send_feedback():
#     data = request.json
#     overall = data.get('overallExperience', 0)
#     ease = data.get('easeOfUse', 0)
#     performance = data.get('applicationPerformance', 0)
#     comments = data.get('comments', "")

#     try:
#         # Create the email message
#         msg = Message(
#             subject="New PackerVision AI Feedback",
#             recipients=["recipient-sakshitandon1193@gmail.com"],
#             body=f"""
#             New Feedback Received:
#             Overall Experience: {overall}/5 Stars
#             Ease of Use: {ease}/5 Stars
#             Application Performance: {performance}/5 Stars
#             User Comments:
#             {comments}
#             """
#         )
#         mail.send(msg)
#         return jsonify({"status": "success", "message": "Feedback sent via email"}), 200
#     except Exception as e:
#         print(f"Mail Error: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500


# # Configuration
# app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['OUTPUT_FOLDER'] = 'outputs'
# app.config['REPORTS_FOLDER'] = 'reports'

# # Model configuration
# MODEL_PATH = 'best.pt'

# # Application start time (for uptime tracking)
# app_start_time = time.time()

# # Create necessary folders
# os.makedirs('uploads', exist_ok=True)
# os.makedirs('outputs', exist_ok=True)
# os.makedirs('reports', exist_ok=True)

# # Import Blueprints
# from routes.dashboard_routes import dashboard_bp
# from routes.packer_routes import packer_bp
# from routes.monitoring_routes import monitoring_bp
# from routes.video_processing_routes import video_bp
# from routes.reports_routes import reports_bp
# from routes.camera_routes import camera_bp
# from routes.auth_routes import auth_bp

# # Register Blueprints
# app.register_blueprint(dashboard_bp)
# app.register_blueprint(auth_bp)  
# app.register_blueprint(packer_bp)
# app.register_blueprint(monitoring_bp)
# app.register_blueprint(video_bp)
# app.register_blueprint(reports_bp)
# app.register_blueprint(camera_bp)


# # --- FIXED REPORTS ENDPOINT (PAGINATION SUPPORT) ---
# @app.route('/api/reports', methods=['GET'])
# def get_reports_paginated():
#     """
#     Fetch reports with server-side pagination, filtering, and sorting.
#     Overrides any default logic to ensure pagination works correctly.
#     """
#     # 1. Get Query Parameters
#     page = int(request.args.get('page', 1))
#     limit = int(request.args.get('limit', 50))
#     sort_order = request.args.get('sort', 'newest')
#     date_from = request.args.get('from')
#     date_to = request.args.get('to')
#     search = request.args.get('search', '')
   
#     # 2. Build SQL Query
#     query = "SELECT * FROM events WHERE 1=1"
#     count_query = "SELECT COUNT(*) FROM events WHERE 1=1"
#     params = []

#     # Apply Date Filters
#     if date_from:
#         query += " AND date(timestamp) >= ?"
#         count_query += " AND date(timestamp) >= ?"
#         params.append(date_from)
   
#     if date_to:
#         query += " AND date(timestamp) <= ?"
#         count_query += " AND date(timestamp) <= ?"
#         params.append(date_to)

#     # Apply Search (Filter by Camera Name/Packer)
#     if search:
#         query += " AND camera_name LIKE ?"
#         count_query += " AND camera_name LIKE ?"
#         params.append(f"%{search}%")

#     conn = get_db_connection()
#     conn.row_factory = sqlite3.Row  # Access columns by name
   
#     try:
#         # 3. Get Total Count (For Pagination Calculation)
#         total_records = conn.execute(count_query, params).fetchone()[0]

#         # 4. Apply Sorting
#         if sort_order == 'oldest':
#             query += " ORDER BY timestamp ASC"
#         else:
#             query += " ORDER BY timestamp DESC"

#         # 5. Apply Pagination (Limit & Offset)
#         offset = (page - 1) * limit
#         query += " LIMIT ? OFFSET ?"
#         params.extend([limit, offset])

#         # 6. Execute Data Query
#         events = conn.execute(query, params).fetchall()

#         # 7. Map Data for Frontend
#         reports_data = []
#         for e in events:
#             reports_data.append({
#                 "id": str(e['id']),
#                 "created_at": e['timestamp'],
#                 "packer_name": e['camera_name'], # Maps camera name to packer name
#                 "location": "Zone A",
#                 "summary": {
#                     "total_events": 1,
#                     "total_cycles": 0,
#                     "packer_efficiency": 100, # Default values if not in DB
#                     "manual_efficiency": 100,
#                     "bags_placed": 0,
#                     "stuck_bags": 0,
#                     "bags_missed": 0
#                 }
#             })

#         return jsonify({
#             "reports": reports_data,
#             "total": total_records,
#             "total_pages": (total_records + limit - 1) // limit,
#             "current_page": page
#         })

#     except Exception as e:
#         print(f"Error fetching reports: {e}")
#         return jsonify({"error": str(e)}), 500
#     finally:
#         conn.close()


# # Root endpoint
# @app.route('/', methods=['GET'])
# def index():
#     """Root endpoint with API information"""
#     return jsonify({
#         "service": "PackerPro Efficiency Monitor API",
#         "version": "1.0.0",
#         "status": "running",
#         "endpoints": {
#             "dashboard": "/api/dashboard",
#             "packers": "/api/packers",
#             "monitoring": "/api/monitor",
#             "video_processing": "/api/process",
#             "reports": "/api/reports"
#         },
#         "documentation": "/api/docs"
#     }), 200

# @app.route('/api/static/evidence/<session_id>/<filename>')
# def serve_evidence(session_id, filename):
#     """Serve evidence images"""
#     evidence_dir = os.path.join('evidence', session_id)
#     # print(f"Attempting to serve: {evidence_dir}/{filename}")  # Debug log
   
#     if not os.path.exists(evidence_dir):
#         return jsonify({"error": "Evidence directory not found"}), 404
   
#     try:
#         return send_from_directory(evidence_dir, filename)
#     except FileNotFoundError:
#         return jsonify({"error": "File not found"}), 404

# # Health check endpoint
# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """API health check"""
#     uptime_seconds = time.time() - app_start_time
   
#     return jsonify({
#         "status": "healthy",
#         "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
#         "uptime_seconds": round(uptime_seconds, 2),
#         "uptime_hours": round(uptime_seconds / 3600, 2),
#         "model_loaded": os.path.exists(MODEL_PATH),
#         "model_path": MODEL_PATH
#     }), 200


# # API documentation endpoint
# @app.route('/api/docs', methods=['GET'])
# def api_docs():
#     """API documentation"""
#     return jsonify({
#         "title": "PackerPro API Documentation",
#         "version": "1.0.0",
#         "description": "REST API for Packer Efficiency Monitoring System",
#         "base_url": "http://localhost:5000/api",
#         "endpoints": {
#             "Dashboard": {
#                 "GET /api/dashboard/stats": "Get dashboard statistics",
#                 "GET /api/dashboard/recent-reports": "Get recent reports",
#                 "GET /api/dashboard/overview": "Get comprehensive overview",
#                 "GET /api/dashboard/performance-comparison": "Get comparison data for bar charts"
#             },
#             "Packer Management": {
#                 "GET /api/packers": "List all packers",
#                 "POST /api/packers": "Create new packer",
#                 "GET /api/packers/{id}": "Get packer details",
#                 "PUT /api/packers/{id}": "Update packer",
#                 "DELETE /api/packers/{id}": "Delete packer",
#                 "GET /api/packers/count": "Get packer count"
#             },
#             "Live Monitoring": {
#                 "POST /api/monitor/start": "Start monitoring",
#                 "POST /api/monitor/stop/{id}": "Stop monitoring",
#                 "GET /api/monitor/metrics/{id}": "Get live metrics",
#                 "GET /api/monitor/sessions": "List active sessions",
#                 "POST /api/monitor/pause/{id}": "Pause monitoring",
#                 "POST /api/monitor/resume/{id}": "Resume monitoring"
#             },
#             "Video Processing": {
#                 "POST /api/process/upload": "Upload video",
#                 "POST /api/process/start": "Start processing",
#                 "GET /api/process/status/{job_id}": "Get job status",
#                 "GET /api/process/jobs": "List all jobs",
#                 "GET /api/process/download/{job_id}": "Download output",
#                 "POST /api/process/cancel/{job_id}": "Cancel job"
#             },
#             "Reports": {
#                 "GET /api/reports": "List all reports (Paginated)",
#                 "GET /api/reports/{id}": "Get report details",
#                 "DELETE /api/reports/{id}": "Delete report",
#                 "GET /api/reports/{id}/download": "Download report"
#             }
#         }
#     }), 200


# # Error handlers
# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({
#         "error": "Not Found",
#         "message": "The requested resource was not found",
#         "status": 404
#     }), 404


# @app.errorhandler(400)
# def bad_request(error):
#     return jsonify({
#         "error": "Bad Request",
#         "message": "The request was invalid or cannot be served",
#         "status": 400
#     }), 400


# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({
#         "error": "Internal Server Error",
#         "message": "An internal server error occurred",
#         "status": 500
#     }), 500

# @app.errorhandler(413)
# def request_entity_too_large(error):
#     return jsonify({
#         "error": "File Too Large",
#         "message": "The uploaded file exceeds the maximum allowed size (500MB)",
#         "status": 413
#     }), 413


# # Run application
# if __name__ == '__main__':
#     print("=" * 60)
#     print("PackerPro Efficiency Monitor - Backend API")
#     print("=" * 60)
#     print(f"Server starting on: http://localhost:5000")
#     print(f"Model path: {MODEL_PATH}")
#     print(f"Model exists: {os.path.exists(MODEL_PATH)}")
#     print("=" * 60)
#     print("\nAvailable endpoints:")
#     print("  Dashboard:     /api/dashboard")
#     print("  Packers:       /api/packers")
#     print("  Monitoring:    /api/monitor")
#     print("  Processing:    /api/process")
#     print("  Reports:       /api/reports")
#     print("  Health Check:  /api/health")
#     print("  Documentation: /api/docs")
#     print("=" * 60)
#     print("\nPress CTRL+C to stop the server\n")
   
#     app.run(
#         host='0.0.0.0',
#         port=5000,
#         debug=True,
#         threaded=True
#     )