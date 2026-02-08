# """
# Reports Routes - SQLite Version (FIXED)
# Handles report management, retrieval, and CSV exporting.
# """

# from flask import Blueprint, jsonify, request, send_file
# import json
# import os
# import pandas as pd
# from datetime import datetime
# from database import get_db_connection

# # Create Blueprint
# reports_bp = Blueprint('reports', __name__, url_prefix='/api/reports')

# def save_report_to_db(report_data):
#     """
#     Helper used by Monitoring and Video routes to save final session results.
#     """
#     summary = report_data.get('summary', {})
#     try:
#         conn = get_db_connection()
#         conn.execute('''
#             INSERT INTO reports (
#                 id, packer_id, packer_name, total_events, total_cycles, 
#                 bags_placed, bags_missed, stuck_bags, 
#                 packer_efficiency, manual_efficiency, 
#                 elapsed_time, timestamp
#             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         ''', (
#             report_data['id'],
#             report_data['packer_id'],
#             report_data.get('packer_name', 'Unknown'),
#             summary.get('total_events', 0),
#             summary.get('total_cycles', 0),
#             summary.get('bags_placed', 0),
#             summary.get('bags_missed', 0),
#             summary.get('stuck_bags', 0),
#             summary.get('packer_efficiency', 0),
#             summary.get('manual_efficiency', 0),
#             summary.get('elapsed_time', 0),
#             datetime.now().isoformat()
#         ))
#         conn.commit()
#         conn.close()
#         return True
#     except Exception as e:
#         print(f"Error saving report to SQLite: {e}")
#         return False

# @reports_bp.route('', methods=['GET'])
# def list_reports():
#     """List reports from SQLite with optional filters"""
#     limit = request.args.get('limit', 100, type=int)
#     packer_id = request.args.get('packer_id')
#     sort_order = request.args.get('sort', 'newest')
    
#     conn = get_db_connection()
    
#     # Build query with JOIN to get packer name
#     query = """
#         SELECT r.*, p.name as packer_name, p.location
#         FROM reports r 
#         LEFT JOIN packers p ON r.packer_id = p.id
#         WHERE 1=1
#     """
#     params = []
    
#     if packer_id:
#         query += " AND r.packer_id = ?"
#         params.append(packer_id)
    
#     # Sort order
#     if sort_order == 'newest':
#         query += " ORDER BY r.timestamp DESC"
#     else:
#         query += " ORDER BY r.timestamp ASC"
        
#     query += " LIMIT ?"
#     params.append(limit)
    
#     reports = conn.execute(query, params).fetchall()
#     conn.close()
    
#     formatted = []
#     for r in reports:
#         formatted.append({
#             "id": r['id'],
#             "packer_id": r['packer_id'],
#             "packer_name": r['packer_name'] or 'Unknown',
#             "location": r['location'] or 'N/A',
#             "created_at": r['timestamp'],
#             "summary": {
#                 "total_events": r['total_events'],
#                 "total_cycles": round(r['total_cycles'], 2),
#                 "bags_placed": r['bags_placed'],
#                 "bags_missed": r['bags_missed'],
#                 "stuck_bags": r['stuck_bags'],
#                 "packer_efficiency": round(r['packer_efficiency'], 2),
#                 "manual_efficiency": round(r['manual_efficiency'], 2)
#             },
#             "has_evidence": os.path.exists(f"evidence/{r['id']}")
#         })

#     return jsonify({
#         "reports": formatted, 
#         "total": len(formatted)
#     }), 200

# @reports_bp.route('/<report_id>/evidence', methods=['GET'])
# def get_report_evidence(report_id):
#     """Returns list of proof-of-work images for the eye-icon modal"""
#     # Note: For simplicity, session_id is used as report_id in your stop logic
#     evidence_path = os.path.join('evidence', report_id)
    
#     if not os.path.exists(evidence_path):
#         return jsonify({"evidence": []}), 200
    
#     # Return URLs or filenames that the frontend can use to display images
#     files = [f for f in os.listdir(evidence_path) if f.endswith('.jpg')]
#     return jsonify({
#         "report_id": report_id,
#         "evidence": sorted(files)
#     }), 200
    
# # @reports_bp.route('/<report_id>', methods=['GET'])
# # def get_report(report_id):
# #     """Get single report details"""
# #     conn = get_db_connection()
# #     query = """
# #         SELECT r.*, p.name as packer_name, p.location, p.spouts
# #         FROM reports r
# #         LEFT JOIN packers p ON r.packer_id = p.id
# #         WHERE r.id = ?
# #     """
# #     report = conn.execute(query, (report_id,)).fetchone()
# #     conn.close()
    
# #     if not report:
# #         return jsonify({"error": "Report not found"}), 404
    
# #     return jsonify({
# #         "id": report['id'],
# #         "packer_id": report['packer_id'],
# #         "packer_name": report['packer_name'],
# #         "location": report['location'],
# #         "spouts": report['spouts'],
# #         "timestamp": report['timestamp'],
# #         "summary": {
# #             "total_events": report['total_events'],
# #             "total_cycles": round(report['total_cycles'], 2),
# #             "bags_placed": report['bags_placed'],
# #             "bags_missed": report['bags_missed'],
# #             "stuck_bags": report['stuck_bags'],
# #             "packer_efficiency": round(report['packer_efficiency'], 2),
# #             "manual_efficiency": round(report['manual_efficiency'], 2),
# #             "elapsed_time": round(report['elapsed_time'], 2)
# #         }
# #     }), 200

# # routes/reports_routes.py

# @reports_bp.route('/<report_id>', methods=['GET'])
# def get_report(report_id):
#     """Get single report with Stuck Bags and Evidence check"""
#     conn = get_db_connection()
#     query = """
#         SELECT r.*, p.name as packer_name, p.location, p.spouts
#         FROM reports r
#         LEFT JOIN packers p ON r.packer_id = p.id
#         WHERE r.id = ?
#     """
#     report = conn.execute(query, (report_id,)).fetchone()
#     conn.close()
    
#     if not report:
#         return jsonify({"error": "Report not found"}), 404
    
#     report_dict = dict(report)
#     return jsonify({
#         **report_dict,
#         "summary": {
#             "total_events": report_dict['total_events'],
#             "total_cycles": round(report_dict['total_cycles'], 2),
#             "bags_placed": report_dict['bags_placed'],
#             "bags_missed": report_dict['bags_missed'],
#             "stuck_bags": report_dict['stuck_bags'], # Ensure this exists
#             "packer_efficiency": round(report_dict['packer_efficiency'], 2),
#             "manual_efficiency": round(report_dict['manual_efficiency'], 2),
#             "elapsed_time": round(report_dict['elapsed_time'], 2)
#         }
#     }), 200
    
# @reports_bp.route('/<report_id>', methods=['DELETE'])
# def delete_report(report_id):
#     """Delete a report"""
#     conn = get_db_connection()
    
#     # Check if exists
#     exists = conn.execute('SELECT id FROM reports WHERE id = ?', (report_id,)).fetchone()
#     if not exists:
#         conn.close()
#         return jsonify({"error": "Report not found"}), 404
    
#     conn.execute('DELETE FROM reports WHERE id = ?', (report_id,))
#     conn.commit()
#     conn.close()
    
#     return jsonify({
#         "message": "Report deleted successfully",
#         "report_id": report_id
#     }), 200

# @reports_bp.route('/stats', methods=['GET'])
# def get_report_stats():
#     """Get aggregate statistics from all reports"""
#     conn = get_db_connection()
    
#     query = """
#         SELECT 
#             COUNT(*) as total_reports,
#             SUM(total_events) as total_events,
#             SUM(total_cycles) as total_cycles,
#             SUM(bags_placed) as total_bags_placed,
#             SUM(bags_missed) as total_bags_missed,
#             SUM(stuck_bags) as total_stuck_bags,
#             AVG(packer_efficiency) as avg_packer_efficiency,
#             AVG(manual_efficiency) as avg_manual_efficiency
#         FROM reports
#     """
    
#     stats = conn.execute(query).fetchone()
#     conn.close()
    
#     if not stats or stats['total_reports'] == 0:
#         return jsonify({
#             "total_reports": 0,
#             "total_events": 0,
#             "total_cycles": 0,
#             "total_bags_placed": 0,
#             "total_bags_missed": 0,
#             "total_stuck_bags": 0,
#             "average_packer_efficiency": 0,
#             "average_manual_efficiency": 0
#         }), 200
    
#     return jsonify({
#         "total_reports": stats['total_reports'],
#         "total_events": stats['total_events'] or 0,
#         "total_cycles": round(stats['total_cycles'] or 0, 2),
#         "total_bags_placed": stats['total_bags_placed'] or 0,
#         "total_bags_missed": stats['total_bags_missed'] or 0,
#         "total_stuck_bags": stats['total_stuck_bags'] or 0,
#         "average_packer_efficiency": round(stats['avg_packer_efficiency'] or 0, 2),
#         "average_manual_efficiency": round(stats['avg_manual_efficiency'] or 0, 2)
#     }), 200

# @reports_bp.route('/by-packer/<packer_id>', methods=['GET'])
# def get_packer_reports(packer_id):
#     """Get all reports for a specific packer"""
#     conn = get_db_connection()
    
#     query = """
#         SELECT * FROM reports 
#         WHERE packer_id = ?
#         ORDER BY timestamp DESC
#     """
    
#     reports = conn.execute(query, (packer_id,)).fetchall()
#     conn.close()
    
#     formatted = []
#     for r in reports:
#         formatted.append({
#             "id": r['id'],
#             "created_at": r['timestamp'],
#             "summary": {
#                 "total_events": r['total_events'],
#                 "total_cycles": round(r['total_cycles'], 2),
#                 "bags_placed": r['bags_placed'],
#                 "bags_missed": r['bags_missed'],
#                 "stuck_bags": r['stuck_bags'],
#                 "packer_efficiency": round(r['packer_efficiency'], 2),
#                 "manual_efficiency": round(r['manual_efficiency'], 2)
#             }
#         })
    
#     return jsonify({
#         "packer_id": packer_id,
#         "reports": formatted,
#         "total": len(formatted)
#     }), 200

# @reports_bp.route('/export-csv', methods=['GET'])
# def export_csv():
#     """Export reports to CSV with date filtering"""
#     start_date = request.args.get('from')  # YYYY-MM-DD
#     end_date = request.args.get('to')      # YYYY-MM-DD
    
#     conn = get_db_connection()
    
#     query = """
#         SELECT 
#             r.id, r.timestamp, r.packer_name, p.location,
#             r.total_events, r.total_cycles, r.bags_placed, 
#             r.bags_missed, r.stuck_bags, r.packer_efficiency, 
#             r.manual_efficiency, r.elapsed_time
#         FROM reports r
#         LEFT JOIN packers p ON r.packer_id = p.id
#     """
#     params = []
    
#     if start_date and end_date:
#         query += " WHERE date(r.timestamp) BETWEEN ? AND ?"
#         params = [start_date, end_date]
    
#     query += " ORDER BY r.timestamp DESC"
    
#     df = pd.read_sql_query(query, conn, params=params)
#     conn.close()
    
#     # Create exports folder
#     os.makedirs('exports', exist_ok=True)
    
#     # Generate filename
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f"packer_reports_{timestamp}.csv"
#     file_path = os.path.join('exports', filename)
    
#     # Save CSV
#     df.to_csv(file_path, index=False)
    
#     return send_file(
#         file_path, 
#         as_attachment=True,
#         download_name=filename,
#         mimetype='text/csv'
#     )

# @reports_bp.route('/create', methods=['POST'])
# def create_report():
#     """
#     Manual endpoint to create a report
#     (Usually called automatically by monitoring system)
#     """
#     data = request.get_json()
    
#     if not data:
#         return jsonify({"error": "No data provided"}), 400
    
#     # Validate required fields
#     required = ['id', 'packer_id']
#     for field in required:
#         if field not in data:
#             return jsonify({"error": f"Missing required field: {field}"}), 400
    
#     success = save_report_to_db(data)
    
#     if success:
#         return jsonify({
#             "message": "Report created successfully",
#             "report_id": data['id']
#         }), 201
#     else:
#         return jsonify({
#             "error": "Failed to create report"
#         }), 500



"""
Reports Routes - SQLite Version (FIXED)
Handles report management, retrieval, pagination, and CSV exporting.
"""

from flask import Blueprint, jsonify, request, send_file
import json
import os
import math
import pandas as pd
from datetime import datetime
from database import get_db_connection

# Create Blueprint
reports_bp = Blueprint('reports', __name__, url_prefix='/api/reports')

def save_report_to_db(report_data):
    """
    Helper used by Monitoring and Video routes to save final session results.
    """
    summary = report_data.get('summary', {})
    try:
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO reports (
                id, packer_id, packer_name, total_events, total_cycles,
                bags_placed, bags_missed, stuck_bags,
                packer_efficiency, manual_efficiency,
                elapsed_time, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report_data['id'],
            report_data['packer_id'],
            report_data.get('packer_name', 'Unknown'),
            summary.get('total_events', 0),
            summary.get('total_cycles', 0),
            summary.get('bags_placed', 0),
            summary.get('bags_missed', 0),
            summary.get('stuck_bags', 0),
            summary.get('packer_efficiency', 0),
            summary.get('manual_efficiency', 0),
            summary.get('elapsed_time', 0),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving report to SQLite: {e}")
        return False

@reports_bp.route('', methods=['GET'])
def list_reports():
    """
    List reports from SQLite with Pagination, Date Filters, and Sorting.
    """
    # 1. Get Query Parameters
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    sort_order = request.args.get('sort', 'newest')
    packer_id = request.args.get('packer_id')
    date_from = request.args.get('from')
    date_to = request.args.get('to')
    search = request.args.get('search', '')

    conn = get_db_connection()
   
    # 2. Build Base Query and Count Query
    # We allow searching by Packer Name (joined from packers table) or Report ID
    query = """
        SELECT r.*, p.name as packer_name_joined, p.location
        FROM reports r
        LEFT JOIN packers p ON r.packer_id = p.id
        WHERE 1=1
    """
   
    count_query = """
        SELECT COUNT(*)
        FROM reports r
        LEFT JOIN packers p ON r.packer_id = p.id
        WHERE 1=1
    """
   
    params = []

    # 3. Apply Filters
    if packer_id:
        filter_clause = " AND r.packer_id = ?"
        query += filter_clause
        count_query += filter_clause
        params.append(packer_id)
   
    if date_from:
        # Assumes timestamp is ISO string (YYYY-MM-DD...)
        filter_clause = " AND date(r.timestamp) >= ?"
        query += filter_clause
        count_query += filter_clause
        params.append(date_from)
       
    if date_to:
        filter_clause = " AND date(r.timestamp) <= ?"
        query += filter_clause
        count_query += filter_clause
        params.append(date_to)

    if search:
        # Search in packer name (from DB or Joined) or ID
        filter_clause = " AND (r.packer_name LIKE ? OR p.name LIKE ? OR r.id LIKE ?)"
        query += filter_clause
        count_query += filter_clause
        term = f"%{search}%"
        params.extend([term, term, term])

    # 4. Get Total Count (for pagination)
    total_records = conn.execute(count_query, params).fetchone()[0]

    # 5. Apply Sorting
    if sort_order == 'newest':
        query += " ORDER BY r.timestamp DESC"
    else:
        query += " ORDER BY r.timestamp ASC"
       
    # 6. Apply Pagination (Limit & Offset)
    offset = (page - 1) * limit
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])
   
    # 7. Execute Final Query
    reports = conn.execute(query, params).fetchall()
    conn.close()
   
    # 8. Format Data
    formatted = []
    for r in reports:
        # Use joined name if available, fallback to stored name, fallback to 'Unknown'
        p_name = r['packer_name_joined'] if r['packer_name_joined'] else (r['packer_name'] or 'Unknown')
       
        formatted.append({
            "id": r['id'],
            "packer_id": r['packer_id'],
            "packer_name": p_name,
            "location": r['location'] or 'N/A',
            "created_at": r['timestamp'],
            "summary": {
                "total_events": r['total_events'],
                "total_cycles": round(r['total_cycles'], 2),
                "bags_placed": r['bags_placed'],
                "bags_missed": r['bags_missed'],
                "stuck_bags": r['stuck_bags'],
                "packer_efficiency": round(r['packer_efficiency'], 2),
                "manual_efficiency": round(r['manual_efficiency'], 2)
            },
            # Check for evidence images
            "has_evidence": os.path.exists(os.path.join("evidence", str(r['id'])))
        })

    # 9. Calculate Pagination Metadata
    total_pages = math.ceil(total_records / limit) if limit > 0 else 0

    return jsonify({
        "reports": formatted,
        "total": total_records,
        "total_pages": total_pages,
        "current_page": page,
        "limit": limit
    }), 200

@reports_bp.route('/<report_id>/evidence', methods=['GET'])
def get_report_evidence(report_id):
    """Returns list of proof-of-work images for the eye-icon modal"""
    evidence_path = os.path.join('evidence', report_id)
   
    if not os.path.exists(evidence_path):
        return jsonify({"evidence": []}), 200
   
    # Return filenames that the frontend can use to display images
    files = [f for f in os.listdir(evidence_path) if f.endswith('.jpg')]
    return jsonify({
        "report_id": report_id,
        "evidence": sorted(files)
    }), 200
   
@reports_bp.route('/<report_id>', methods=['GET'])
def get_report(report_id):
    """Get single report with Stuck Bags and Evidence check"""
    conn = get_db_connection()
    query = """
        SELECT r.*, p.name as packer_name, p.location, p.spouts
        FROM reports r
        LEFT JOIN packers p ON r.packer_id = p.id
        WHERE r.id = ?
    """
    report = conn.execute(query, (report_id,)).fetchone()
    conn.close()
   
    if not report:
        return jsonify({"error": "Report not found"}), 404
   
    report_dict = dict(report)
    return jsonify({
        **report_dict,
        "summary": {
            "total_events": report_dict['total_events'],
            "total_cycles": round(report_dict['total_cycles'], 2),
            "bags_placed": report_dict['bags_placed'],
            "bags_missed": report_dict['bags_missed'],
            "stuck_bags": report_dict['stuck_bags'],
            "packer_efficiency": round(report_dict['packer_efficiency'], 2),
            "manual_efficiency": round(report_dict['manual_efficiency'], 2),
            "elapsed_time": round(report_dict['elapsed_time'], 2)
        }
    }), 200
   
@reports_bp.route('/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    """Delete a report"""
    conn = get_db_connection()
   
    # Check if exists
    exists = conn.execute('SELECT id FROM reports WHERE id = ?', (report_id,)).fetchone()
    if not exists:
        conn.close()
        return jsonify({"error": "Report not found"}), 404
   
    conn.execute('DELETE FROM reports WHERE id = ?', (report_id,))
    conn.commit()
    conn.close()
   
    return jsonify({
        "message": "Report deleted successfully",
        "report_id": report_id
    }), 200

@reports_bp.route('/stats', methods=['GET'])
def get_report_stats():
    """Get aggregate statistics from all reports"""
    conn = get_db_connection()
   
    query = """
        SELECT
            COUNT(*) as total_reports,
            SUM(total_events) as total_events,
            SUM(total_cycles) as total_cycles,
            SUM(bags_placed) as total_bags_placed,
            SUM(bags_missed) as total_bags_missed,
            SUM(stuck_bags) as total_stuck_bags,
            AVG(packer_efficiency) as avg_packer_efficiency,
            AVG(manual_efficiency) as avg_manual_efficiency
        FROM reports
    """
   
    stats = conn.execute(query).fetchone()
    conn.close()
   
    if not stats or stats['total_reports'] == 0:
        return jsonify({
            "total_reports": 0,
            "total_events": 0,
            "total_cycles": 0,
            "total_bags_placed": 0,
            "total_bags_missed": 0,
            "total_stuck_bags": 0,
            "average_packer_efficiency": 0,
            "average_manual_efficiency": 0
        }), 200
   
    return jsonify({
        "total_reports": stats['total_reports'],
        "total_events": stats['total_events'] or 0,
        "total_cycles": round(stats['total_cycles'] or 0, 2),
        "total_bags_placed": stats['total_bags_placed'] or 0,
        "total_bags_missed": stats['total_bags_missed'] or 0,
        "total_stuck_bags": stats['total_stuck_bags'] or 0,
        "average_packer_efficiency": round(stats['avg_packer_efficiency'] or 0, 2),
        "average_manual_efficiency": round(stats['avg_manual_efficiency'] or 0, 2)
    }), 200

@reports_bp.route('/by-packer/<packer_id>', methods=['GET'])
def get_packer_reports(packer_id):
    """Get all reports for a specific packer"""
    conn = get_db_connection()
   
    query = """
        SELECT * FROM reports
        WHERE packer_id = ?
        ORDER BY timestamp DESC
    """
   
    reports = conn.execute(query, (packer_id,)).fetchall()
    conn.close()
   
    formatted = []
    for r in reports:
        formatted.append({
            "id": r['id'],
            "created_at": r['timestamp'],
            "summary": {
                "total_events": r['total_events'],
                "total_cycles": round(r['total_cycles'], 2),
                "bags_placed": r['bags_placed'],
                "bags_missed": r['bags_missed'],
                "stuck_bags": r['stuck_bags'],
                "packer_efficiency": round(r['packer_efficiency'], 2),
                "manual_efficiency": round(r['manual_efficiency'], 2)
            }
        })
   
    return jsonify({
        "packer_id": packer_id,
        "reports": formatted,
        "total": len(formatted)
    }), 200

@reports_bp.route('/export-csv', methods=['GET'])
def export_csv():
    """Export reports to CSV with date filtering"""
    start_date = request.args.get('from')  # YYYY-MM-DD
    end_date = request.args.get('to')      # YYYY-MM-DD
   
    conn = get_db_connection()
   
    query = """
        SELECT
            r.id, r.timestamp, r.packer_name, p.location,
            r.total_events, r.total_cycles, r.bags_placed,
            r.bags_missed, r.stuck_bags, r.packer_efficiency,
            r.manual_efficiency, r.elapsed_time
        FROM reports r
        LEFT JOIN packers p ON r.packer_id = p.id
    """
    params = []
   
    if start_date and end_date:
        query += " WHERE date(r.timestamp) BETWEEN ? AND ?"
        params = [start_date, end_date]
   
    query += " ORDER BY r.timestamp DESC"
   
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
   
    # Create exports folder
    os.makedirs('exports', exist_ok=True)
   
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"packer_reports_{timestamp}.csv"
    file_path = os.path.join('exports', filename)
   
    # Save CSV
    df.to_csv(file_path, index=False)
   
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv'
    )

@reports_bp.route('/create', methods=['POST'])
def create_report():
    """
    Manual endpoint to create a report
    (Usually called automatically by monitoring system)
    """
    data = request.get_json()
   
    if not data:
        return jsonify({"error": "No data provided"}), 400
   
    # Validate required fields
    required = ['id', 'packer_id']
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
   
    success = save_report_to_db(data)
   
    if success:
        return jsonify({
            "message": "Report created successfully",
            "report_id": data['id']
        }), 201
    else:
        return jsonify({
            "error": "Failed to create report"
        }), 500