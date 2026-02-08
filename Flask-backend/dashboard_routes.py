"""
Dashboard Routes
Handles dashboard overview and statistics endpoints for the home page

Endpoints:
    GET /api/dashboard/stats           - Get dashboard statistics (supports packer_id filter)
    GET /api/dashboard/recent-reports  - Get recent reports
    GET /api/dashboard/overview        - Get comprehensive overview
    GET /api/dashboard/system-health   - Get system health status
"""

from flask import Blueprint, jsonify, request
import os
import json
import time
from datetime import datetime
from routes.monitoring_routes import active_sessions

# Create Blueprint
dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/api/dashboard')

# Paths
REPORTS_FOLDER = 'reports'

def get_active_monitor_summary(packer_data):
    """
    Helper function to safely get monitor summary from active packer
    Returns empty dict if monitor is not available
    """
    try:
        if packer_data.get('monitor') and hasattr(packer_data['monitor'], 'get_summary'):
            return packer_data['monitor'].get_summary()
    except Exception as e:
        print(f"Error getting monitor summary: {e}")
    return {}


@dashboard_bp.route('/stats', methods=['GET'])
def get_dashboard_stats():
    """
    Get high-level production KPIs for the dashboard top cards.
    Supports filtering by packer_id query parameter.
    If packer_id='all' or not provided, aggregates across all active sessions.
    """
    from routes.monitoring_routes import active_sessions
    
    packer_id = request.args.get('packer_id', 'all')
    
    total_bags_placed = 0
    total_stuck_alerts = 0
    efficiency_sum = 0
    active_lines_count = 0

    # Filter sessions based on packer_id
    for session_id, session in active_sessions.items():
        if session.get('status') == 'running':
            # If specific packer selected, only include that packer's session
            if packer_id != 'all' and session.get('packer_id') != packer_id:
                continue
                
            metrics = session['monitor'].get_summary()
            total_bags_placed += metrics.get('bags_placed', 0)
            total_stuck_alerts += metrics.get('stuck_bags', 0)
            efficiency_sum += metrics.get('packer_efficiency', 0)
            active_lines_count += 1

    # Calculate Average Operational Efficiency
    avg_efficiency = round(efficiency_sum / active_lines_count, 2) if active_lines_count > 0 else 0
    
    # Determine system status based on activity
    system_status = "Optimal" if active_lines_count > 0 and total_stuck_alerts == 0 else "Idle"
    if total_stuck_alerts > 0:
        system_status = "Warning"

    return jsonify({
        "operational_efficiency": avg_efficiency,
        "total_bags_placed": total_bags_placed,
        "active_lines": active_lines_count,
        "total_alerts": total_stuck_alerts,
        "system_status": system_status
    }), 200
    

@dashboard_bp.route('/performance-comparison', methods=['GET'])
def get_performance_comparison():
    """
    Returns the latest performance metrics for each configured packer.
    Supports filtering by packer_id query parameter.
    If packer_id='all' or not provided, returns all packers.
    Prioritizes Live/Processing sessions, falls back to latest report.
    """
    from routes.packer_routes import get_packers_db
    from routes.monitoring_routes import active_sessions, get_active_monitor_summary
    from database import get_db_connection
    
    packer_id = request.args.get('packer_id', 'all')
    packers_db = get_packers_db()
    comparison_results = []

    # Create a map of active sessions by packer_id for quick lookup
    active_by_packer = {}
    for session_id, session in active_sessions.items():
        if session.get('status') == 'running':
            p_id = session.get('packer_id')
            active_by_packer[p_id] = session

    # Filter packers based on packer_id parameter
    filtered_packers = {}
    if packer_id == 'all':
        filtered_packers = packers_db
    elif packer_id in packers_db:
        filtered_packers = {packer_id: packers_db[packer_id]}
    
    for p_id, p_data in filtered_packers.items():
        # Initialize with zero metrics
        packer_metrics = {
            "packer_name": p_data.get('name'),
            "packer_efficiency": 0,
            "manual_efficiency": 0,
            "status": "idle"
        }

        # 1. Check for Live Monitor Session
        if p_id in active_by_packer:
            summary = get_active_monitor_summary(active_by_packer[p_id])
            if summary:
                packer_metrics.update({
                    "packer_efficiency": round(summary.get('packer_efficiency', 0), 2),
                    "manual_efficiency": round(summary.get('manual_efficiency', 0), 2),
                    "status": "active"
                })
        # 2. Fall back to latest historical report from SQLite
        else:
            try:
                conn = get_db_connection()
                report = conn.execute('''
                    SELECT packer_efficiency, manual_efficiency 
                    FROM reports 
                    WHERE packer_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (p_id,)).fetchone()
                conn.close()
                
                if report:
                    packer_metrics.update({
                        "packer_efficiency": round(report['packer_efficiency'], 2),
                        "manual_efficiency": round(report['manual_efficiency'], 2)
                    })
            except Exception as e:
                print(f"Error fetching report for packer {p_id}: {e}")

        comparison_results.append(packer_metrics)

    return jsonify({"by_packer": comparison_results}), 200


@dashboard_bp.route('/packer-stats/<packer_id>', methods=['GET'])
def get_specific_packer_stats(packer_id):
    """
    Retrieves historical averages and live status for a specific packer
    to update the Manual and Packer Efficiency gauges on the dashboard.
    """
    from routes.packer_routes import get_packers_db
    from routes.monitoring_routes import active_sessions, get_active_monitor_summary
    from database import get_db_connection
    
    packers_db = get_packers_db()
    if packer_id not in packers_db:
        return jsonify({"error": "Packer not found"}), 404
        
    packer_data = packers_db[packer_id]
    
    # Check if packer has an active session
    active_session = None
    for session_id, session in active_sessions.items():
        if session.get('packer_id') == packer_id and session.get('status') == 'running':
            active_session = session
            break
    
    is_active = active_session is not None
    total_bags_placed = 0
    total_events = 0
    total_stuck = 0
    report_count = 0

    # Get metrics from live monitor if active
    if is_active:
        live_metrics = get_active_monitor_summary(active_session)
        if live_metrics:
            total_bags_placed = live_metrics.get('bags_placed', 0)
            total_events = live_metrics.get('total_events', 0)
            total_stuck = live_metrics.get('stuck_bags', 0)
            
            # Calculate efficiencies from live data
            manual_eff = (total_bags_placed / total_events * 100) if total_events > 0 else 0
            total_ops = total_events + total_stuck
            packer_eff = (total_events / total_ops * 100) if total_ops > 0 else 0
            
            return jsonify({
                "packer_id": packer_id,
                "packer_name": packer_data.get('name'),
                "report_count": 0,
                "is_active_now": True,
                "status": "active",
                "metrics": {
                    "manual_efficiency": round(manual_eff, 2),
                    "packer_efficiency": round(packer_eff, 2),
                    "total_events": total_events,
                    "total_stuck": total_stuck,
                    "total_bags_placed": total_bags_placed
                }
            }), 200

    # If not active, aggregate from historical reports
    try:
        conn = get_db_connection()
        
        # Get aggregated historical data
        aggregated = conn.execute('''
            SELECT 
                COUNT(*) as report_count,
                SUM(bags_placed) as total_bags,
                SUM(total_events) as total_events,
                SUM(stuck_bags) as total_stuck
            FROM reports 
            WHERE packer_id = ?
        ''', (packer_id,)).fetchone()
        
        # Get latest efficiency values
        latest = conn.execute('''
            SELECT packer_efficiency, manual_efficiency
            FROM reports 
            WHERE packer_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (packer_id,)).fetchone()
        
        conn.close()
        
        if aggregated:
            report_count = aggregated['report_count'] or 0
            total_bags_placed = aggregated['total_bags'] or 0
            total_events = aggregated['total_events'] or 0
            total_stuck = aggregated['total_stuck'] or 0
        
        # Use latest efficiency values if available, otherwise calculate
        if latest:
            manual_eff = latest['manual_efficiency']
            packer_eff = latest['packer_efficiency']
        else:
            manual_eff = (total_bags_placed / total_events * 100) if total_events > 0 else 0
            total_ops = total_events + total_stuck
            packer_eff = (total_events / total_ops * 100) if total_ops > 0 else 0

    except Exception as e:
        print(f"Database error for packer {packer_id}: {e}")
        manual_eff = 0
        packer_eff = 0

    return jsonify({
        "packer_id": packer_id,
        "packer_name": packer_data.get('name'),
        "report_count": report_count,
        "is_active_now": False,
        "status": "idle",
        "metrics": {
            "manual_efficiency": round(manual_eff, 2),
            "packer_efficiency": round(packer_eff, 2),
            "total_events": total_events,
            "total_stuck": total_stuck,
            "total_bags_placed": total_bags_placed
        }
    }), 200


@dashboard_bp.route('/overview', methods=['GET'])
def get_dashboard_overview():
    """Get comprehensive dashboard overview with defined counts"""
    from routes.packer_routes import get_packers_db
    from routes.monitoring_routes import active_sessions, get_active_monitor_summary
    from database import get_db_connection
    from app import app_start_time
    
    packers_db = get_packers_db()
    
    # Calculate counts from active_sessions
    active_packers_count = sum(1 for s in active_sessions.values() if s.get('status') == 'running')
    
    # Get total reports count from database
    try:
        conn = get_db_connection()
        total_reports_count = conn.execute('SELECT COUNT(*) as count FROM reports').fetchone()['count']
        conn.close()
    except:
        total_reports_count = 0

    # Build active packers list
    active_packers_list = []
    for session_id, session in active_sessions.items():
        if session.get('status') == 'running':
            packer_id = session.get('packer_id')
            packer_data = packers_db.get(packer_id, {})
            
            active_packers_list.append({
                "id": packer_id,
                "name": packer_data.get('name', 'Unknown'),
                "session_id": session_id,
                "status": "live",
                "metrics": get_active_monitor_summary(session)
            })
    
    uptime_seconds = time.time() - app_start_time

    # Get recent reports from database
    recent_reports = []
    try:
        conn = get_db_connection()
        reports = conn.execute('''
            SELECT id, packer_name, timestamp, packer_efficiency
            FROM reports 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''').fetchall()
        conn.close()
        
        for report in reports:
            recent_reports.append({
                "id": report['id'],
                "packer_name": report['packer_name'],
                "created_at": report['timestamp'],
                "efficiency": round(report['packer_efficiency'], 2)
            })
    except Exception as e:
        print(f"Error fetching recent reports: {e}")

    return jsonify({
        "stats": {
            "total_packers": len(packers_db),
            "active_monitors": active_packers_count,
            "total_reports": total_reports_count
        },
        "system_status": {
            "uptime_formatted": _format_uptime(uptime_seconds),
            "status": "healthy" if active_packers_count > 0 else "idle"
        },
        "active_packers": active_packers_list,
        "recent_reports": recent_reports,
        "timestamp": datetime.now().isoformat()
    }), 200
    
@dashboard_bp.route('/system-health', methods=['GET'])
def get_system_health():
    """
    Get system health status
    """
    from app import app_start_time
    from routes.packer_routes import get_packers_db
    
    uptime_seconds = time.time() - app_start_time
    
    packers_db = get_packers_db()
    active_sessions = sum(1 for p in packers_db.values() if p.get('status') == 'active')
    
    # Determine system status
    status = "healthy"
    if active_sessions == 0 and len(packers_db) > 0:
        status = "warning"
    
    return jsonify({
        "status": status,
        "uptime": {
            "seconds": round(uptime_seconds, 2),
            "hours": int(uptime_seconds / 3600),
            "days": int(uptime_seconds / 86400),
            "formatted": _format_uptime(uptime_seconds)
        },
        "active_sessions": active_sessions,
        "total_packers": len(packers_db),
        "timestamp": datetime.now().isoformat()
    }), 200

@dashboard_bp.route('/activity-log', methods=['GET'])
def get_activity_log():
    """
    Get recent activity log
    """
    from routes.packer_routes import get_packers_db
    
    limit = request.args.get('limit', 20, type=int)
    packers_db = get_packers_db()
    activities = []
    
    # Add packer creation activities
    for packer_id, packer_data in packers_db.items():
        if packer_data.get('created_at'):
            activities.append({
                "type": "packer_created",
                "message": f"Packer '{packer_data.get('name')}' was created",
                "timestamp": packer_data.get('created_at'),
                "packer_id": packer_id
            })
        
        # Add monitoring activities
        if packer_data.get('monitoring_started_at'):
            activities.append({
                "type": "monitoring_started",
                "message": f"Monitoring started for '{packer_data.get('name')}'",
                "timestamp": packer_data.get('monitoring_started_at'),
                "packer_id": packer_id
            })
    
    # Add report generation activities
    if os.path.exists(REPORTS_FOLDER):
        for filename in os.listdir(REPORTS_FOLDER):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(REPORTS_FOLDER, filename), 'r') as f:
                        report_data = json.load(f)
                    
                    activities.append({
                        "type": "report_generated",
                        "message": f"Report generated for '{report_data.get('packer_name')}'",
                        "timestamp": report_data.get('timestamp'),
                        "report_id": filename.replace('.json', '').replace('report_', '')
                    })
                except:
                    continue
    
    # Sort by timestamp (newest first)
    activities.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return jsonify({
        "activities": activities[:limit],
        "count": len(activities[:limit])
    }), 200

@dashboard_bp.route('/performance-metrics', methods=['GET'])
def get_performance_metrics():
    """
    Get aggregated performance metrics across all packers
    """
    from routes.packer_routes import get_packers_db
    
    packers_db = get_packers_db()
    
    # Aggregate metrics from all active monitors
    total_cycles = 0
    total_bags_placed = 0
    total_bags_missed = 0
    total_stuck_bags = 0
    
    by_packer = []
    
    for packer_id, packer_data in packers_db.items():
        if packer_data.get('monitor'):
            metrics = get_active_monitor_summary(packer_data)
            
            if metrics:
                total_cycles += metrics.get('total_cycles', 0)
                total_bags_placed += metrics.get('bags_placed', 0)
                total_bags_missed += metrics.get('bags_missed', 0)
                total_stuck_bags += metrics.get('stuck_bags', 0)
                
                by_packer.append({
                    "packer_id": packer_id,
                    "packer_name": packer_data.get('name'),
                    "metrics": metrics
                })
    
    # Calculate overall efficiencies
    overall_packer_efficiency = 0
    overall_manual_efficiency = 0
    
    if total_cycles > 0:
        overall_manual_efficiency = (total_bags_placed / total_cycles) * 100
    
    total_operations = total_cycles + total_stuck_bags
    if total_operations > 0:
        overall_packer_efficiency = ((total_bags_placed + total_bags_missed) / total_operations) * 100
    
    return jsonify({
        "total_cycles": total_cycles,
        "total_bags_placed": total_bags_placed,
        "total_bags_missed": total_bags_missed,
        "total_stuck_bags": total_stuck_bags,
        "overall_packer_efficiency": round(overall_packer_efficiency, 2),
        "overall_manual_efficiency": round(overall_manual_efficiency, 2),
        "by_packer": by_packer
    }), 200

# Helper Functions

def _format_uptime(seconds):
    """Format uptime in human-readable format"""
    days = int(seconds / 86400)
    hours = int((seconds % 86400) / 3600)
    minutes = int((seconds % 3600) / 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    
    if not parts:
        return f"{int(seconds)}s"
    
    return " ".join(parts)

def _get_file_size_mb(filepath):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(filepath)
        return round(size_bytes / (1024 * 1024), 2)
    except:
        return 0

# """
# Dashboard Routes
# Handles dashboard overview and statistics endpoints for the home page

# Endpoints:
#     GET /api/dashboard/stats           - Get dashboard statistics
#     GET /api/dashboard/recent-reports  - Get recent reports
#     GET /api/dashboard/overview        - Get comprehensive overview
#     GET /api/dashboard/system-health   - Get system health status
# """

# from flask import Blueprint, jsonify, request
# import os
# import json
# import time
# from datetime import datetime
# from routes.monitoring_routes import active_sessions

# # Create Blueprint
# dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/api/dashboard')

# # Paths
# REPORTS_FOLDER = 'reports'

# def get_active_monitor_summary(packer_data):
#     """
#     Helper function to safely get monitor summary from active packer
#     Returns empty dict if monitor is not available
#     """
#     try:
#         if packer_data.get('monitor') and hasattr(packer_data['monitor'], 'get_summary'):
#             return packer_data['monitor'].get_summary()
#     except Exception as e:
#         print(f"Error getting monitor summary: {e}")
#     return {}


# @dashboard_bp.route('/stats', methods=['GET'])
# def get_dashboard_stats():
#     """
#     Get high-level production KPIs for the dashboard top cards.
#     Aggregates data across all active sessions in monitoring_routes.
#     """
#     from routes.monitoring_routes import active_sessions
    
#     total_bags_placed = 0
#     total_stuck_alerts = 0
#     efficiency_sum = 0
#     active_lines_count = 0

#     # Iterate through live monitoring sessions to get real-time video test data
#     for session_id, session in active_sessions.items():
#         if session.get('status') == 'running':
#             metrics = session['monitor'].get_summary()
#             total_bags_placed += metrics.get('bags_placed', 0)
#             total_stuck_alerts += metrics.get('stuck_bags', 0)
#             efficiency_sum += metrics.get('packer_efficiency', 0)
#             active_lines_count += 1

#     # Calculate Average Operational Efficiency
#     avg_efficiency = round(efficiency_sum / active_lines_count, 2) if active_lines_count > 0 else 0
    
#     # Determine system status based on activity
#     system_status = "Optimal" if active_lines_count > 0 and total_stuck_alerts == 0 else "Idle"
#     if total_stuck_alerts > 0:
#         system_status = "Warning"

#     return jsonify({
#         "operational_efficiency": avg_efficiency,
#         "total_bags_placed": total_bags_placed,
#         "active_lines": active_lines_count,
#         "total_alerts": total_stuck_alerts,
#         "system_status": system_status
#     }), 200
    

# @dashboard_bp.route('/performance-comparison', methods=['GET'])
# def get_performance_comparison():
#     """
#     Returns the latest performance metrics for each configured packer.
#     Prioritizes Live/Processing sessions, falls back to latest report.
#     """
#     from routes.packer_routes import get_packers_db
#     from routes.monitoring_routes import active_sessions, get_active_monitor_summary
#     from database import get_db_connection
    
#     packers_db = get_packers_db()
#     comparison_results = []

#     # Create a map of active sessions by packer_id for quick lookup
#     active_by_packer = {}
#     for session_id, session in active_sessions.items():
#         if session.get('status') == 'running':
#             packer_id = session.get('packer_id')
#             active_by_packer[packer_id] = session

#     for p_id, p_data in packers_db.items():
#         # Initialize with zero metrics
#         packer_metrics = {
#             "packer_name": p_data.get('name'),
#             "packer_efficiency": 0,
#             "manual_efficiency": 0,
#             "status": "idle"
#         }

#         # 1. Check for Live Monitor Session
#         if p_id in active_by_packer:
#             summary = get_active_monitor_summary(active_by_packer[p_id])
#             if summary:
#                 packer_metrics.update({
#                     "packer_efficiency": round(summary.get('packer_efficiency', 0), 2),
#                     "manual_efficiency": round(summary.get('manual_efficiency', 0), 2),
#                     "status": "active"
#                 })
#         # 2. Fall back to latest historical report from SQLite
#         else:
#             try:
#                 conn = get_db_connection()
#                 report = conn.execute('''
#                     SELECT packer_efficiency, manual_efficiency 
#                     FROM reports 
#                     WHERE packer_id = ? 
#                     ORDER BY timestamp DESC 
#                     LIMIT 1
#                 ''', (p_id,)).fetchone()
#                 conn.close()
                
#                 if report:
#                     packer_metrics.update({
#                         "packer_efficiency": round(report['packer_efficiency'], 2),
#                         "manual_efficiency": round(report['manual_efficiency'], 2)
#                     })
#             except Exception as e:
#                 print(f"Error fetching report for packer {p_id}: {e}")

#         comparison_results.append(packer_metrics)

#     return jsonify({"by_packer": comparison_results}), 200


# @dashboard_bp.route('/packer-stats/<packer_id>', methods=['GET'])
# def get_specific_packer_stats(packer_id):
#     """
#     Retrieves historical averages and live status for a specific packer
#     to update the Manual and Packer Efficiency gauges on the dashboard.
#     """
#     from routes.packer_routes import get_packers_db
#     from routes.monitoring_routes import active_sessions, get_active_monitor_summary
#     from database import get_db_connection
    
#     packers_db = get_packers_db()
#     if packer_id not in packers_db:
#         return jsonify({"error": "Packer not found"}), 404
        
#     packer_data = packers_db[packer_id]
    
#     # Check if packer has an active session
#     active_session = None
#     for session_id, session in active_sessions.items():
#         if session.get('packer_id') == packer_id and session.get('status') == 'running':
#             active_session = session
#             break
    
#     is_active = active_session is not None
#     total_bags_placed = 0
#     total_events = 0
#     total_stuck = 0
#     report_count = 0

#     # Get metrics from live monitor if active
#     if is_active:
#         live_metrics = get_active_monitor_summary(active_session)
#         if live_metrics:
#             total_bags_placed = live_metrics.get('bags_placed', 0)
#             total_events = live_metrics.get('total_events', 0)
#             total_stuck = live_metrics.get('stuck_bags', 0)
            
#             # Calculate efficiencies from live data
#             manual_eff = (total_bags_placed / total_events * 100) if total_events > 0 else 0
#             total_ops = total_events + total_stuck
#             packer_eff = (total_events / total_ops * 100) if total_ops > 0 else 0
            
#             return jsonify({
#                 "packer_id": packer_id,
#                 "packer_name": packer_data.get('name'),
#                 "report_count": 0,
#                 "is_active_now": True,
#                 "status": "active",
#                 "metrics": {
#                     "manual_efficiency": round(manual_eff, 2),
#                     "packer_efficiency": round(packer_eff, 2),
#                     "total_events": total_events,
#                     "total_stuck": total_stuck,
#                     "total_bags_placed": total_bags_placed
#                 }
#             }), 200

#     # If not active, aggregate from historical reports
#     try:
#         conn = get_db_connection()
        
#         # Get aggregated historical data
#         aggregated = conn.execute('''
#             SELECT 
#                 COUNT(*) as report_count,
#                 SUM(bags_placed) as total_bags,
#                 SUM(total_events) as total_events,
#                 SUM(stuck_bags) as total_stuck
#             FROM reports 
#             WHERE packer_id = ?
#         ''', (packer_id,)).fetchone()
        
#         # Get latest efficiency values
#         latest = conn.execute('''
#             SELECT packer_efficiency, manual_efficiency
#             FROM reports 
#             WHERE packer_id = ?
#             ORDER BY timestamp DESC
#             LIMIT 1
#         ''', (packer_id,)).fetchone()
        
#         conn.close()
        
#         if aggregated:
#             report_count = aggregated['report_count'] or 0
#             total_bags_placed = aggregated['total_bags'] or 0
#             total_events = aggregated['total_events'] or 0
#             total_stuck = aggregated['total_stuck'] or 0
        
#         # Use latest efficiency values if available, otherwise calculate
#         if latest:
#             manual_eff = latest['manual_efficiency']
#             packer_eff = latest['packer_efficiency']
#         else:
#             manual_eff = (total_bags_placed / total_events * 100) if total_events > 0 else 0
#             total_ops = total_events + total_stuck
#             packer_eff = (total_events / total_ops * 100) if total_ops > 0 else 0

#     except Exception as e:
#         print(f"Database error for packer {packer_id}: {e}")
#         manual_eff = 0
#         packer_eff = 0

#     return jsonify({
#         "packer_id": packer_id,
#         "packer_name": packer_data.get('name'),
#         "report_count": report_count,
#         "is_active_now": False,
#         "status": "idle",
#         "metrics": {
#             "manual_efficiency": round(manual_eff, 2),
#             "packer_efficiency": round(packer_eff, 2),
#             "total_events": total_events,
#             "total_stuck": total_stuck,
#             "total_bags_placed": total_bags_placed
#         }
#     }), 200


# @dashboard_bp.route('/overview', methods=['GET'])
# def get_dashboard_overview():
#     """Get comprehensive dashboard overview with defined counts"""
#     from routes.packer_routes import get_packers_db
#     from routes.monitoring_routes import active_sessions, get_active_monitor_summary
#     from database import get_db_connection
#     from app import app_start_time
    
#     packers_db = get_packers_db()
    
#     # Calculate counts from active_sessions
#     active_packers_count = sum(1 for s in active_sessions.values() if s.get('status') == 'running')
    
#     # Get total reports count from database
#     try:
#         conn = get_db_connection()
#         total_reports_count = conn.execute('SELECT COUNT(*) as count FROM reports').fetchone()['count']
#         conn.close()
#     except:
#         total_reports_count = 0

#     # Build active packers list
#     active_packers_list = []
#     for session_id, session in active_sessions.items():
#         if session.get('status') == 'running':
#             packer_id = session.get('packer_id')
#             packer_data = packers_db.get(packer_id, {})
            
#             active_packers_list.append({
#                 "id": packer_id,
#                 "name": packer_data.get('name', 'Unknown'),
#                 "session_id": session_id,
#                 "status": "live",
#                 "metrics": get_active_monitor_summary(session)
#             })
    
#     uptime_seconds = time.time() - app_start_time

#     # Get recent reports from database
#     recent_reports = []
#     try:
#         conn = get_db_connection()
#         reports = conn.execute('''
#             SELECT id, packer_name, timestamp, packer_efficiency
#             FROM reports 
#             ORDER BY timestamp DESC 
#             LIMIT 5
#         ''').fetchall()
#         conn.close()
        
#         for report in reports:
#             recent_reports.append({
#                 "id": report['id'],
#                 "packer_name": report['packer_name'],
#                 "created_at": report['timestamp'],
#                 "efficiency": round(report['packer_efficiency'], 2)
#             })
#     except Exception as e:
#         print(f"Error fetching recent reports: {e}")

#     return jsonify({
#         "stats": {
#             "total_packers": len(packers_db),
#             "active_monitors": active_packers_count,
#             "total_reports": total_reports_count
#         },
#         "system_status": {
#             "uptime_formatted": _format_uptime(uptime_seconds),
#             "status": "healthy" if active_packers_count > 0 else "idle"
#         },
#         "active_packers": active_packers_list,
#         "recent_reports": recent_reports,
#         "timestamp": datetime.now().isoformat()
#     }), 200
    
# @dashboard_bp.route('/system-health', methods=['GET'])
# def get_system_health():
#     """
#     Get system health status
#     """
#     from app import app_start_time
#     from routes.packer_routes import get_packers_db
    
#     uptime_seconds = time.time() - app_start_time
    
#     packers_db = get_packers_db()
#     active_sessions = sum(1 for p in packers_db.values() if p.get('status') == 'active')
    
#     # Determine system status
#     status = "healthy"
#     if active_sessions == 0 and len(packers_db) > 0:
#         status = "warning"
    
#     return jsonify({
#         "status": status,
#         "uptime": {
#             "seconds": round(uptime_seconds, 2),
#             "hours": int(uptime_seconds / 3600),
#             "days": int(uptime_seconds / 86400),
#             "formatted": _format_uptime(uptime_seconds)
#         },
#         "active_sessions": active_sessions,
#         "total_packers": len(packers_db),
#         "timestamp": datetime.now().isoformat()
#     }), 200

# @dashboard_bp.route('/activity-log', methods=['GET'])
# def get_activity_log():
#     """
#     Get recent activity log
#     """
#     from routes.packer_routes import get_packers_db
    
#     limit = request.args.get('limit', 20, type=int)
#     packers_db = get_packers_db()
#     activities = []
    
#     # Add packer creation activities
#     for packer_id, packer_data in packers_db.items():
#         if packer_data.get('created_at'):
#             activities.append({
#                 "type": "packer_created",
#                 "message": f"Packer '{packer_data.get('name')}' was created",
#                 "timestamp": packer_data.get('created_at'),
#                 "packer_id": packer_id
#             })
        
#         # Add monitoring activities
#         if packer_data.get('monitoring_started_at'):
#             activities.append({
#                 "type": "monitoring_started",
#                 "message": f"Monitoring started for '{packer_data.get('name')}'",
#                 "timestamp": packer_data.get('monitoring_started_at'),
#                 "packer_id": packer_id
#             })
    
#     # Add report generation activities
#     if os.path.exists(REPORTS_FOLDER):
#         for filename in os.listdir(REPORTS_FOLDER):
#             if filename.endswith('.json'):
#                 try:
#                     with open(os.path.join(REPORTS_FOLDER, filename), 'r') as f:
#                         report_data = json.load(f)
                    
#                     activities.append({
#                         "type": "report_generated",
#                         "message": f"Report generated for '{report_data.get('packer_name')}'",
#                         "timestamp": report_data.get('timestamp'),
#                         "report_id": filename.replace('.json', '').replace('report_', '')
#                     })
#                 except:
#                     continue
    
#     # Sort by timestamp (newest first)
#     activities.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
#     return jsonify({
#         "activities": activities[:limit],
#         "count": len(activities[:limit])
#     }), 200

# @dashboard_bp.route('/performance-metrics', methods=['GET'])
# def get_performance_metrics():
#     """
#     Get aggregated performance metrics across all packers
#     """
#     from routes.packer_routes import get_packers_db
    
#     packers_db = get_packers_db()
    
#     # Aggregate metrics from all active monitors
#     total_cycles = 0
#     total_bags_placed = 0
#     total_bags_missed = 0
#     total_stuck_bags = 0
    
#     by_packer = []
    
#     for packer_id, packer_data in packers_db.items():
#         if packer_data.get('monitor'):
#             metrics = get_active_monitor_summary(packer_data)
            
#             if metrics:
#                 total_cycles += metrics.get('total_cycles', 0)
#                 total_bags_placed += metrics.get('bags_placed', 0)
#                 total_bags_missed += metrics.get('bags_missed', 0)
#                 total_stuck_bags += metrics.get('stuck_bags', 0)
                
#                 by_packer.append({
#                     "packer_id": packer_id,
#                     "packer_name": packer_data.get('name'),
#                     "metrics": metrics
#                 })
    
#     # Calculate overall efficiencies
#     overall_packer_efficiency = 0
#     overall_manual_efficiency = 0
    
#     if total_cycles > 0:
#         overall_manual_efficiency = (total_bags_placed / total_cycles) * 100
    
#     total_operations = total_cycles + total_stuck_bags
#     if total_operations > 0:
#         overall_packer_efficiency = ((total_bags_placed + total_bags_missed) / total_operations) * 100
    
#     return jsonify({
#         "total_cycles": total_cycles,
#         "total_bags_placed": total_bags_placed,
#         "total_bags_missed": total_bags_missed,
#         "total_stuck_bags": total_stuck_bags,
#         "overall_packer_efficiency": round(overall_packer_efficiency, 2),
#         "overall_manual_efficiency": round(overall_manual_efficiency, 2),
#         "by_packer": by_packer
#     }), 200

# # Helper Functions

# def _format_uptime(seconds):
#     """Format uptime in human-readable format"""
#     days = int(seconds / 86400)
#     hours = int((seconds % 86400) / 3600)
#     minutes = int((seconds % 3600) / 60)
    
#     parts = []
#     if days > 0:
#         parts.append(f"{days}d")
#     if hours > 0:
#         parts.append(f"{hours}h")
#     if minutes > 0:
#         parts.append(f"{minutes}m")
    
#     if not parts:
#         return f"{int(seconds)}s"
    
#     return " ".join(parts)

# def _get_file_size_mb(filepath):
#     """Get file size in MB"""
#     try:
#         size_bytes = os.path.getsize(filepath)
#         return round(size_bytes / (1024 * 1024), 2)
#     except:
#         return 0