"""
Reports Routes - SQLite Version
Handles report management, retrieval, pagination, and CSV exporting.
Updated: shift column support + shift-wise filtering and CSV export.
"""

from flask import Blueprint, jsonify, request, send_file
import json
import os
import math
import pandas as pd
from datetime import datetime
from typing import Optional
from database import get_db_connection

# Create Blueprint
reports_bp = Blueprint('reports', __name__, url_prefix='/api/reports')


def save_report_to_db(report_data):
    """
    Helper used by Monitoring and Video routes to save final session results.
    Now persists the `shift` field (A / B / C) when provided.
    For manually-triggered sessions shift is None / omitted → stored as NULL.
    """
    from shift_scheduler import get_shift_for_time

    summary = report_data.get('summary', {})

    # Determine shift: use explicit value if provided, else derive from timestamp
    shift = report_data.get('shift')
    if not shift:
        ts_str = report_data.get('timestamp') or datetime.now().isoformat()
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            ts = datetime.now()
        shift = get_shift_for_time(ts)

    try:
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO reports (
                id, packer_id, packer_name, shift,
                total_events, total_cycles,
                bags_placed, bags_missed, stuck_bags,
                packer_efficiency, manual_efficiency,
                elapsed_time, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report_data['id'],
            report_data['packer_id'],
            report_data.get('packer_name', 'Unknown'),
            shift,
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
    List reports from SQLite with Pagination, Date Filters, Shift Filter, and Sorting.
    New query param: ?shift=A|B|C
    """
    page         = request.args.get('page', 1, type=int)
    limit        = request.args.get('limit', 50, type=int)
    sort_order   = request.args.get('sort', 'newest')
    packer_id    = request.args.get('packer_id')
    date_from    = request.args.get('from')
    date_to      = request.args.get('to')
    search       = request.args.get('search', '')
    shift_filter = request.args.get('shift', '').upper().strip()  # '', 'A', 'B', 'C'

    conn = get_db_connection()

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

    if packer_id:
        clause = " AND r.packer_id = ?"
        query += clause; count_query += clause; params.append(packer_id)

    if date_from:
        clause = " AND date(r.timestamp) >= ?"
        query += clause; count_query += clause; params.append(date_from)

    if date_to:
        clause = " AND date(r.timestamp) <= ?"
        query += clause; count_query += clause; params.append(date_to)

    if shift_filter in ('A', 'B', 'C'):
        clause = " AND r.shift = ?"
        query += clause; count_query += clause; params.append(shift_filter)

    if search:
        clause = " AND (r.packer_name LIKE ? OR p.name LIKE ? OR r.id LIKE ?)"
        query += clause; count_query += clause
        term = f"%{search}%"
        params.extend([term, term, term])

    total_records = conn.execute(count_query, params).fetchone()[0]

    query += " ORDER BY r.timestamp DESC" if sort_order == 'newest' else " ORDER BY r.timestamp ASC"
    offset = (page - 1) * limit
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    reports = conn.execute(query, params).fetchall()
    conn.close()

    formatted = []
    for r in reports:
        p_name = r['packer_name_joined'] if r['packer_name_joined'] else (r['packer_name'] or 'Unknown')
        formatted.append({
            "id":          r['id'],
            "packer_id":   r['packer_id'],
            "packer_name": p_name,
            "location":    r['location'] or 'N/A',
            "shift":       r['shift'] or 'Manual',
            "shift_label": _shift_label(r['shift']),
            "created_at":  r['timestamp'],
            "summary": {
                "total_events":       r['total_events'],
                "total_cycles":       round(r['total_cycles'], 2),
                "bags_placed":        r['bags_placed'],
                "bags_missed":        r['bags_missed'],
                "stuck_bags":         r['stuck_bags'],
                "packer_efficiency":  round(r['packer_efficiency'], 2),
                "manual_efficiency":  round(r['manual_efficiency'], 2)
            },
            "has_evidence": os.path.exists(os.path.join("/media/amazin/store/evidences", str(r['id'])))
        })

    total_pages = math.ceil(total_records / limit) if limit > 0 else 0
    return jsonify({
        "reports":      formatted,
        "total":        total_records,
        "total_pages":  total_pages,
        "current_page": page,
        "limit":        limit
    }), 200


def _shift_label(shift: Optional[str]) -> str:
    labels = {"A": "Shift A (6AM–2PM)", "B": "Shift B (2PM–10PM)", "C": "Shift C (10PM–6AM)"}
    return labels.get(shift, "Manual")


@reports_bp.route('/<report_id>/evidence', methods=['GET'])
def get_report_evidence(report_id):
    """Returns list of proof-of-work images for the eye-icon modal"""
    evidence_path = os.path.join("/media/amazin/store/evidences", report_id)
    if not os.path.exists(evidence_path):
        return jsonify({"evidence": []}), 200
    files = [f for f in os.listdir(evidence_path) if f.endswith('.jpg')]
    return jsonify({"report_id": report_id, "evidence": sorted(files)}), 200


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

    rd = dict(report)
    rd['shift_label'] = _shift_label(rd.get('shift'))
    return jsonify({
        **rd,
        "summary": {
            "total_events":      rd['total_events'],
            "total_cycles":      round(rd['total_cycles'], 2),
            "bags_placed":       rd['bags_placed'],
            "bags_missed":       rd['bags_missed'],
            "stuck_bags":        rd['stuck_bags'],
            "packer_efficiency": round(rd['packer_efficiency'], 2),
            "manual_efficiency": round(rd['manual_efficiency'], 2),
            "elapsed_time":      round(rd['elapsed_time'], 2)
        }
    }), 200


@reports_bp.route('/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    """Delete a report"""
    conn = get_db_connection()
    exists = conn.execute('SELECT id FROM reports WHERE id = ?', (report_id,)).fetchone()
    if not exists:
        conn.close()
        return jsonify({"error": "Report not found"}), 404
    conn.execute('DELETE FROM reports WHERE id = ?', (report_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Report deleted successfully", "report_id": report_id}), 200


@reports_bp.route('/stats', methods=['GET'])
def get_report_stats():
    """Get aggregate statistics from all reports, optionally filtered by shift"""
    shift_filter = request.args.get('shift', '').upper().strip()
    conn = get_db_connection()

    where = f"WHERE shift = '{shift_filter}'" if shift_filter in ('A', 'B', 'C') else ""
    query = f"""
        SELECT
            COUNT(*) as total_reports,
            SUM(total_events) as total_events,
            SUM(total_cycles) as total_cycles,
            SUM(bags_placed) as total_bags_placed,
            SUM(bags_missed) as total_bags_missed,
            SUM(stuck_bags) as total_stuck_bags,
            AVG(packer_efficiency) as avg_packer_efficiency,
            AVG(manual_efficiency) as avg_manual_efficiency
        FROM reports {where}
    """
    stats = conn.execute(query).fetchone()
    conn.close()

    if not stats or stats['total_reports'] == 0:
        return jsonify({
            "total_reports": 0, "total_events": 0, "total_cycles": 0,
            "total_bags_placed": 0, "total_bags_missed": 0, "total_stuck_bags": 0,
            "average_packer_efficiency": 0, "average_manual_efficiency": 0
        }), 200

    return jsonify({
        "total_reports":             stats['total_reports'],
        "total_events":              stats['total_events'] or 0,
        "total_cycles":              round(stats['total_cycles'] or 0, 2),
        "total_bags_placed":         stats['total_bags_placed'] or 0,
        "total_bags_missed":         stats['total_bags_missed'] or 0,
        "total_stuck_bags":          stats['total_stuck_bags'] or 0,
        "average_packer_efficiency": round(stats['avg_packer_efficiency'] or 0, 2),
        "average_manual_efficiency": round(stats['avg_manual_efficiency'] or 0, 2)
    }), 200


@reports_bp.route('/by-packer/<packer_id>', methods=['GET'])
def get_packer_reports(packer_id):
    """Get all reports for a specific packer"""
    conn = get_db_connection()
    reports = conn.execute(
        "SELECT * FROM reports WHERE packer_id = ? ORDER BY timestamp DESC", (packer_id,)
    ).fetchall()
    conn.close()

    formatted = []
    for r in reports:
        formatted.append({
            "id":          r['id'],
            "shift":       r['shift'] or 'Manual',
            "shift_label": _shift_label(r['shift']),
            "created_at":  r['timestamp'],
            "summary": {
                "total_events":      r['total_events'],
                "total_cycles":      round(r['total_cycles'], 2),
                "bags_placed":       r['bags_placed'],
                "bags_missed":       r['bags_missed'],
                "stuck_bags":        r['stuck_bags'],
                "packer_efficiency": round(r['packer_efficiency'], 2),
                "manual_efficiency": round(r['manual_efficiency'], 2)
            }
        })

    return jsonify({"packer_id": packer_id, "reports": formatted, "total": len(formatted)}), 200


@reports_bp.route('/export-csv', methods=['GET'])
def export_csv():
    """
    Export reports to CSV.
    Supports ?from=YYYY-MM-DD &to=YYYY-MM-DD &shift=A|B|C filtering.
    The CSV now includes a 'shift' and 'shift_label' column.
    """
    start_date   = request.args.get('from')
    end_date     = request.args.get('to')
    shift_filter = request.args.get('shift', '').upper().strip()
    packer_id    = request.args.get('packer_id')

    conn = get_db_connection()

    query = """
        SELECT
            r.id, r.timestamp,
            r.shift,
            r.packer_name, p.location,
            r.total_events, r.total_cycles, r.bags_placed,
            r.bags_missed, r.stuck_bags, r.packer_efficiency,
            r.manual_efficiency, r.elapsed_time
        FROM reports r
        LEFT JOIN packers p ON r.packer_id = p.id
        WHERE 1=1
    """
    params = []

    if start_date and end_date:
        query += " AND date(r.timestamp) BETWEEN ? AND ?"
        params += [start_date, end_date]
    elif start_date:
        query += " AND date(r.timestamp) >= ?"
        params.append(start_date)
    elif end_date:
        query += " AND date(r.timestamp) <= ?"
        params.append(end_date)

    if shift_filter in ('A', 'B', 'C'):
        query += " AND r.shift = ?"
        params.append(shift_filter)

    if packer_id:
        query += " AND r.packer_id = ?"
        params.append(packer_id)

    query += " ORDER BY r.timestamp DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Add human-readable shift label column
    df['shift_label'] = df['shift'].map(
        lambda s: {"A": "Shift A (6AM-2PM)", "B": "Shift B (2PM-10PM)",
                   "C": "Shift C (10PM-6AM)"}.get(s, "Manual")
    )

    # Reorder columns for readability
    col_order = [
        'id', 'timestamp', 'shift', 'shift_label', 'packer_name', 'location',
        'total_events', 'total_cycles', 'bags_placed', 'bags_missed',
        'stuck_bags', 'packer_efficiency', 'manual_efficiency', 'elapsed_time'
    ]
    df = df[[c for c in col_order if c in df.columns]]

    os.makedirs('exports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Build descriptive filename
    parts = ["packer_reports"]
    if shift_filter in ('A', 'B', 'C'):
        parts.append(f"shift{shift_filter}")
    if start_date:
        parts.append(start_date)
    if end_date:
        parts.append(f"to_{end_date}")
    parts.append(timestamp)

    filename = "_".join(parts) + ".csv"
    file_path = os.path.join('exports', filename)
    df.to_csv(file_path, index=False)

    return send_file(file_path, as_attachment=True, download_name=filename, mimetype='text/csv')


@reports_bp.route('/create', methods=['POST'])
def create_report():
    """Manual endpoint to create a report"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    for field in ['id', 'packer_id']:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    success = save_report_to_db(data)
    if success:
        return jsonify({"message": "Report created successfully", "report_id": data['id']}), 201
    return jsonify({"error": "Failed to create report"}), 500


# ── Shift summary endpoint ─────────────────────────────────────────────────────

@reports_bp.route('/shift-summary', methods=['GET'])
def shift_summary():
    """
    Returns per-shift aggregated stats for all shifts (or a specific date range).
    Useful for a dashboard comparison view.

    Optional params: ?from=YYYY-MM-DD &to=YYYY-MM-DD &packer_id=...
    """
    date_from = request.args.get('from')
    date_to   = request.args.get('to')
    packer_id = request.args.get('packer_id')

    conn = get_db_connection()

    base_where = "WHERE 1=1"
    params = []
    if date_from:
        base_where += " AND date(timestamp) >= ?"; params.append(date_from)
    if date_to:
        base_where += " AND date(timestamp) <= ?"; params.append(date_to)
    if packer_id:
        base_where += " AND packer_id = ?"; params.append(packer_id)

    query = f"""
        SELECT
            COALESCE(shift, 'Manual') as shift,
            COUNT(*) as sessions,
            SUM(bags_placed) as total_bags_placed,
            SUM(bags_missed) as total_bags_missed,
            SUM(stuck_bags)  as total_stuck_bags,
            AVG(packer_efficiency) as avg_efficiency,
            SUM(elapsed_time) as total_minutes
        FROM reports
        {base_where}
        GROUP BY COALESCE(shift, 'Manual')
        ORDER BY shift ASC
    """
    rows = conn.execute(query, params).fetchall()
    conn.close()

    result = []
    for r in rows:
        result.append({
            "shift":             r['shift'],
            "shift_label":       _shift_label(r['shift']),
            "sessions":          r['sessions'],
            "total_bags_placed": r['total_bags_placed'] or 0,
            "total_bags_missed": r['total_bags_missed'] or 0,
            "total_stuck_bags":  r['total_stuck_bags'] or 0,
            "avg_efficiency":    round(r['avg_efficiency'] or 0, 2),
            "total_minutes":     round(r['total_minutes'] or 0, 1),
        })

    return jsonify({"shift_summary": result}), 200
