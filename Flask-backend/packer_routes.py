# """
# Packer Management Routes
# Handles all packer configuration and management endpoints
# """

# from flask import Blueprint, request, jsonify
# from datetime import datetime
# import uuid

# # Create Blueprint
# packer_bp = Blueprint('packer', __name__, url_prefix='/api/packers')

# # In-memory storage (replace with database in production)
# packers_db = {}

# # Maximum number of packers allowed
# MAX_PACKERS = 4

# # routes/packer_routes.py

# # @packer_bp.route('', methods=['POST'])
# # def create_packer():
# #     data = request.json
# #     packer_id = str(uuid.uuid4())
    
# #     # NEW FIELDS INTEGRATED HERE
# #     packer_config = {
# #         "name": data.get('name'),
# #         "location": data.get('location', 'Unknown'),
# #         "spouts": data.get('spouts', 8),
# #         "rpm": data.get('rpm', 5),
# #         "camera_id": data.get('camera_id'), # Linked from Cameras Page
# #         "line_position": data.get('line_position', 0.7), # bag counting line
# #         "start_line_position": data.get('start_line_position', 0.2), # filled bag counting
# #         "confidence_threshold": data.get('confidence_threshold', 0.5),
# #         "status": "idle",
# #         "created_at": datetime.now().isoformat(),
# #         "updated_at": datetime.now().isoformat(),
# #         "monitor": None
# #     }
    
# #     packers_db[packer_id] = packer_config
# #     return jsonify({
# #         "message": "Packer created successfully", 
# #         "packer_id": packer_id,
# #         "packer": {"id": packer_id, **packer_config}
# #     }), 201

# # routes/packer_routes.py

# @packer_bp.route('', methods=['POST'])
# def create_packer():
#     data = request.json
    
#     if len(packers_db) >= MAX_PACKERS:
#         return jsonify({"error": "Max limit reached"}), 400

#     packer_id = str(uuid.uuid4())
    
#     # ENSURE camera_id is saved from the request
#     packer_config = {
#         "name": data.get('name'),
#         "location": data.get('location', 'Unknown'),
#         "spouts": int(data.get('spouts', 8)),
#         "rpm": float(data.get('rpm', 5)),
#         "camera_id": data.get('camera_id'), # LINK CREATED HERE
#         "line_position": float(data.get('line_position', 0.7)),
#         "start_line_position": float(data.get('start_line_position', 0.2)),
#         "confidence_threshold": float(data.get('confidence_threshold', 0.5)),
#         "status": "idle",
#         "created_at": datetime.now().isoformat(),
#         "updated_at": datetime.now().isoformat(),
#         "monitor": None
#     }
    
#     packers_db[packer_id] = packer_config
#     return jsonify({
#         "message": "Packer created successfully", 
#         "packer_id": packer_id,
#         "packer": {"id": packer_id, **packer_config}
#     }), 201

# @packer_bp.route('', methods=['GET'])
# def list_packers():
#     packers_list = []
#     for packer_id, packer_data in packers_db.items():
#         packer_info = {
#             "id": packer_id,
#             "name": packer_data.get('name'),
#             "camera_id": packer_data.get('camera_id'), # SEND TO UI
#             "spouts": packer_data.get('spouts'),
#             "rpm": packer_data.get('rpm'),
#             "location": packer_data.get('location'),
#             "status": packer_data.get('status'),
#             "line_position": packer_data.get('line_position'),
#             "start_line_position": packer_data.get('start_line_position'),
#             "confidence_threshold": packer_data.get('confidence_threshold')
#         }
#         packers_list.append(packer_info)
    
#     return jsonify({
#         "packers": packers_list,
#         "total": len(packers_list),
#         "max_allowed": MAX_PACKERS
#     }), 200
    
# @packer_bp.route('/<packer_id>', methods=['PUT'])
# def update_packer(packer_id):
#     if packer_id not in packers_db:
#         return jsonify({"error": "Not found"}), 404
        
#     data = request.json
#     p = packers_db[packer_id]

#     # Update logic for all new fields
#     fields = ['name', 'location', 'spouts', 'rpm', 'camera_id', 
#               'line_position', 'start_line_position', 'confidence_threshold']
              
#     for field in fields:
#         if field in data:
#             p[field] = data[field]
            
#     p['updated_at'] = datetime.now().isoformat()
#     return jsonify({"message": "Updated successfully"}), 200


# # @packer_bp.route('', methods=['GET'])
# # def list_packers():
# #     """
# #     Get list of all configured packers
    
# #     Returns:
# #         JSON list of all packers with their configurations and status
# #     """
# #     packers_list = []
    
# #     for packer_id, packer_data in packers_db.items():
# #         packer_info = {
# #             "id": packer_id,
# #             "name": packer_data.get('name'),
# #             "spouts": packer_data.get('spouts', 8),
# #             "rpm": packer_data.get('rpm', 5),
# #             "location": packer_data.get('location', 'location1'),
# #             "status": packer_data.get('status', 'idle'),
# #             "line_position": packer_data.get('line_position', 0.7),
# #             "start_line_position": packer_data.get('start_line_position', 0.2),
# #             "confidence_threshold": packer_data.get('confidence_threshold', 0.5),
# #             "created_at": packer_data.get('created_at'),
# #             "updated_at": packer_data.get('updated_at')
# #         }
        
# #         # Add metrics if monitor exists
# #         if packer_data.get('monitor'):
# #             packer_info['current_metrics'] = packer_data['monitor'].get_summary()
        
# #         packers_list.append(packer_info)
    
# #     return jsonify({
# #         "packers": packers_list,
# #         "total": len(packers_list),
# #         "max_allowed": MAX_PACKERS
# #     }), 200


# # @packer_bp.route('', methods=['POST'])
# # def create_packer():
# #     """
# #     Create a new packer configuration
    
# #     Request Body:
# #         {
# #             "name": "Line-1",
# #             "spouts": 8,
# #             "rpm": 5,
# #             "location": "location1",
# #             "line_position": 0.7,
# #             "start_line_position": 0.2,
# #             "confidence_threshold": 0.5
# #         }
    
# #     Returns:
# #         Created packer configuration with ID
# #     """
# #     # Check if max limit reached
# #     if len(packers_db) >= MAX_PACKERS:
# #         return jsonify({
# #             "error": f"Maximum packer limit reached ({MAX_PACKERS})",
# #             "message": "Please delete an existing packer before adding a new one"
# #         }), 400
    
# #     data = request.json
    
# #     # Validate required fields
# #     if not data.get('name'):
# #         return jsonify({"error": "Packer name is required"}), 400
    
# #     # Generate unique ID
# #     packer_id = str(uuid.uuid4())
    
# #     # Create packer configuration
# #     packer_config = {
# #         "name": data.get('name'),
# #         "spouts": data.get('spouts', 8),
# #         "rpm": data.get('rpm', 5),
# #         "location": data.get('location', 'location1'),
# #         "line_position": data.get('line_position', 0.7),
# #         "start_line_position": data.get('start_line_position', 0.2),
# #         "confidence_threshold": data.get('confidence_threshold', 0.5),
# #         "status": "configured",
# #         "created_at": datetime.now().isoformat(),
# #         "updated_at": datetime.now().isoformat(),
# #         "monitor": None
# #     }
    
# #     # Store in database
# #     packers_db[packer_id] = packer_config
    
# #     return jsonify({
# #         "message": "Packer created successfully",
# #         "packer_id": packer_id,
# #         "packer": {
# #             "id": packer_id,
# #             **packer_config
# #         }
# #     }), 201


# @packer_bp.route('/<packer_id>', methods=['GET'])
# def get_packer(packer_id):
#     """
#     Get specific packer details
    
#     Args:
#         packer_id: UUID of the packer
    
#     Returns:
#         Packer configuration and current metrics
#     """
#     if packer_id not in packers_db:
#         return jsonify({"error": "Packer not found"}), 404
    
#     packer_data = packers_db[packer_id]
    
#     response = {
#         "id": packer_id,
#         "name": packer_data.get('name'),
#         "spouts": packer_data.get('spouts'),
#         "rpm": packer_data.get('rpm'),
#         "location": packer_data.get('location'),
#         "status": packer_data.get('status'),
#         "config": {
#             "line_position": packer_data.get('line_position'),
#             "start_line_position": packer_data.get('start_line_position'),
#             "confidence_threshold": packer_data.get('confidence_threshold')
#         },
#         "created_at": packer_data.get('created_at'),
#         "updated_at": packer_data.get('updated_at')
#     }
    
#     # Add current metrics if monitoring is active
#     if packer_data.get('monitor'):
#         response['current_metrics'] = packer_data['monitor'].get_summary()
    
#     return jsonify(response), 200


# # @packer_bp.route('/<packer_id>', methods=['PUT'])
# # def update_packer(packer_id):
# #     """
# #     Update packer configuration
    
# #     Args:
# #         packer_id: UUID of the packer
    
# #     Request Body:
# #         {
# #             "name": "Updated Line-1",
# #             "spouts": 10,
# #             "rpm": 6,
# #             "location": "location2",
# #             "line_position": 0.75,
# #             "start_line_position": 0.25,
# #             "confidence_threshold": 0.6
# #         }
    
# #     Returns:
# #         Updated packer configuration
# #     """
# #     if packer_id not in packers_db:
# #         return jsonify({"error": "Packer not found"}), 404
    
# #     data = request.json
# #     packer_data = packers_db[packer_id]
    
# #     # Check if packer is currently active
# #     if packer_data.get('status') == 'active':
# #         return jsonify({
# #             "error": "Cannot update active packer",
# #             "message": "Please stop monitoring before updating configuration"
# #         }), 400
    
# #     # Update fields
# #     if 'name' in data:
# #         packer_data['name'] = data['name']
# #     if 'spouts' in data:
# #         packer_data['spouts'] = data['spouts']
# #     if 'rpm' in data:
# #         packer_data['rpm'] = data['rpm']
# #     if 'location' in data:
# #         packer_data['location'] = data['location']
# #     if 'line_position' in data:
# #         packer_data['line_position'] = data['line_position']
# #     if 'start_line_position' in data:
# #         packer_data['start_line_position'] = data['start_line_position']
# #     if 'confidence_threshold' in data:
# #         packer_data['confidence_threshold'] = data['confidence_threshold']
    
# #     packer_data['updated_at'] = datetime.now().isoformat()
    
# #     return jsonify({
# #         "message": "Packer updated successfully",
# #         "packer": {
# #             "id": packer_id,
# #             **packer_data
# #         }
# #     }), 200


# @packer_bp.route('/<packer_id>', methods=['DELETE'])
# def delete_packer(packer_id):
#     """
#     Delete a packer configuration
    
#     Args:
#         packer_id: UUID of the packer
    
#     Returns:
#         Success message
#     """
#     if packer_id not in packers_db:
#         return jsonify({"error": "Packer not found"}), 404
    
#     packer_data = packers_db[packer_id]
    
#     # Check if packer is currently active
#     if packer_data.get('status') == 'active':
#         return jsonify({
#             "error": "Cannot delete active packer",
#             "message": "Please stop monitoring before deleting"
#         }), 400
    
#     # Delete from database
#     del packers_db[packer_id]
    
#     return jsonify({
#         "message": "Packer deleted successfully",
#         "packer_id": packer_id
#     }), 200


# @packer_bp.route('/<packer_id>/status', methods=['PUT'])
# def update_packer_status(packer_id):
#     """
#     Update packer status (active/idle/stopped)
    
#     Args:
#         packer_id: UUID of the packer
    
#     Request Body:
#         {
#             "status": "active" | "idle" | "stopped"
#         }
    
#     Returns:
#         Updated status
#     """
#     if packer_id not in packers_db:
#         return jsonify({"error": "Packer not found"}), 404
    
#     data = request.json
#     new_status = data.get('status')
    
#     if new_status not in ['active', 'idle', 'stopped']:
#         return jsonify({"error": "Invalid status. Must be 'active', 'idle', or 'stopped'"}), 400
    
#     packers_db[packer_id]['status'] = new_status
#     packers_db[packer_id]['updated_at'] = datetime.now().isoformat()
    
#     return jsonify({
#         "message": f"Packer status updated to {new_status}",
#         "packer_id": packer_id,
#         "status": new_status
#     }), 200


# @packer_bp.route('/count', methods=['GET'])
# def get_packer_count():
#     """
#     Get current packer count and max allowed
    
#     Returns:
#         Packer count information
#     """
#     return jsonify({
#         "current_count": len(packers_db),
#         "max_allowed": MAX_PACKERS,
#         "available_slots": MAX_PACKERS - len(packers_db),
#         "can_add_more": len(packers_db) < MAX_PACKERS
#     }), 200


# # Helper function to get packers_db (for use in other modules)
# def get_packers_db():
#     """Return reference to packers database"""
#     return packers_db



"""
Packer Management Routes - SQLite Version
Handles all packer configuration and management endpoints using persistent storage
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import uuid
from database import get_db_connection

# Create Blueprint
packer_bp = Blueprint('packer', __name__, url_prefix='/api/packers')

# Maximum number of packers allowed
MAX_PACKERS = 4

@packer_bp.route('', methods=['POST'])
def create_packer():
    """Create a new packer and link the camera source"""
    conn = get_db_connection()
    
    count = conn.execute('SELECT COUNT(*) FROM packers').fetchone()[0]
    if count >= MAX_PACKERS:
        conn.close()
        return jsonify({"error": "Max limit reached", "message": f"Maximum of {MAX_PACKERS} packers allowed"}), 400

    data = request.json
    if not data.get('name'):
        conn.close()
        return jsonify({"error": "Packer name is required"}), 400

    packer_id = str(uuid.uuid4())
    camera_id = data.get('camera_id') # Get the assigned camera ID
    
    try:
        # 1. Insert the Packer
        conn.execute('''
            INSERT INTO packers (
                id, name, location, spouts, rpm, camera_id, 
                line_position, start_line_position, confidence_threshold, 
                status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            packer_id, data.get('name'), data.get('location', 'Unknown'), 
            int(data.get('spouts', 8)), float(data.get('rpm', 5)), camera_id, 
            float(data.get('line_position', 0.7)), float(data.get('start_line_position', 0.2)), 
            float(data.get('confidence_threshold', 0.5)), 'idle', datetime.now().isoformat()
        ))

        # 2. SYNC: Update the camera table so it knows it is now assigned to this packer
        if camera_id:
            conn.execute('UPDATE cameras SET packer_id = ? WHERE id = ?', (packer_id, camera_id))
            
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"error": "Database error", "message": str(e)}), 500
    
    conn.close()
    return jsonify({"message": "Packer created successfully", "packer_id": packer_id}), 201

@packer_bp.route('', methods=['GET'])
def list_packers():
    """List all packers with their assigned camera info from the DB"""
    conn = get_db_connection()
    
    # Join with cameras table to get camera name/url if it exists
    query = '''
        SELECT p.*, c.name as camera_name, c.rtsp_url 
        FROM packers p
        LEFT JOIN cameras c ON p.camera_id = c.id
    '''
    rows = conn.execute(query).fetchall()
    conn.close()

    packers_list = []
    for row in rows:
        packer_info = dict(row)
        # Structure camera info for UI consistency
        packer_info['camera_info'] = {
            "id": row['camera_id'],
            "name": row['camera_name'],
            "rtsp_url": row['rtsp_url']
        } if row['camera_id'] else None
        packers_list.append(packer_info)
    
    return jsonify({
        "packers": packers_list,
        "total": len(packers_list),
        "max_allowed": MAX_PACKERS
    }), 200

@packer_bp.route('/<packer_id>', methods=['GET'])
def get_packer(packer_id):
    """Get specific packer details from DB"""
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM packers WHERE id = ?', (packer_id,)).fetchone()
    conn.close()
    
    if not row:
        return jsonify({"error": "Packer not found"}), 404
    
    packer_data = dict(row)
    # Wrap config fields for UI compatibility
    packer_data['config'] = {
        "line_position": row['line_position'],
        "start_line_position": row['start_line_position'],
        "confidence_threshold": row['confidence_threshold']
    }
    
    return jsonify(packer_data), 200

@packer_bp.route('/<packer_id>', methods=['PUT'])
def update_packer(packer_id):
    """Update packer details and sync camera assignment status"""
    data = request.json
    conn = get_db_connection()
    
    packer = conn.execute('SELECT status, camera_id FROM packers WHERE id = ?', (packer_id,)).fetchone()
    if not packer:
        conn.close()
        return jsonify({"error": "Packer not found"}), 404
    
    if packer['status'] == 'active':
        conn.close()
        return jsonify({"error": "Cannot update active packer"}), 400

    fields = ['name', 'location', 'spouts', 'rpm', 'camera_id', 
              'line_position', 'start_line_position', 'confidence_threshold']
    
    updates = []
    values = []
    for field in fields:
        if field in data:
            updates.append(f"{field} = ?")
            values.append(data[field])
    
    if not updates:
        conn.close()
        return jsonify({"message": "No changes"}), 200

    try:
        # 1. Update Packer Table
        values.append(packer_id)
        conn.execute(f"UPDATE packers SET {', '.join(updates)} WHERE id = ?", values)

        # 2. SYNC: If camera was changed, update the cameras table
        if 'camera_id' in data:
            new_camera_id = data['camera_id']
            # Clear ANY camera currently linked to THIS packer
            conn.execute('UPDATE cameras SET packer_id = NULL WHERE packer_id = ?', (packer_id,))
            # Link the NEW camera to this packer
            if new_camera_id:
                conn.execute('UPDATE cameras SET packer_id = ? WHERE id = ?', (packer_id, new_camera_id))

        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500

    conn.close()
    return jsonify({"message": "Updated successfully"}), 200

@packer_bp.route('/<packer_id>', methods=['DELETE'])
def delete_packer(packer_id):
    """Delete a packer from DB"""
    conn = get_db_connection()
    packer = conn.execute('SELECT status FROM packers WHERE id = ?', (packer_id,)).fetchone()
    
    if not packer:
        conn.close()
        return jsonify({"error": "Packer not found"}), 404
    
    # THIS IS TRIGGERING THE 400 ERROR
    if packer['status'] == 'active':
        conn.close()
        return jsonify({"error": "Cannot delete active packer", "message": "Please stop monitoring before deleting"}), 400
    
    conn.execute('DELETE FROM packers WHERE id = ?', (packer_id,))
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Packer deleted successfully", "packer_id": packer_id}), 200

@packer_bp.route('/count', methods=['GET'])
def get_packer_count():
    conn = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM packers').fetchone()[0]
    conn.close()
    return jsonify({
        "current_count": count,
        "max_allowed": MAX_PACKERS,
        "available_slots": MAX_PACKERS - count,
        "can_add_more": count < MAX_PACKERS
    }), 200

# Helper to maintain compatibility with monitoring_routes
def get_packers_db():
    """Returns all packers as a dictionary indexed by ID"""
    conn = get_db_connection()
    rows = conn.execute('SELECT * FROM packers').fetchall()
    conn.close()
    return {row['id']: dict(row) for row in rows}

def update_packer_status_internal(packer_id, status, session_id):
    conn = get_db_connection()
    conn.execute('UPDATE packers SET status = ?, session_id = ? WHERE id = ?', (status, session_id, packer_id))
    conn.commit()
    conn.close()