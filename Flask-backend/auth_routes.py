"""
Authentication Routes
Handles user registration, login, and session management
"""

from flask import Blueprint, request, jsonify
import hashlib
import secrets
from datetime import datetime
from database import get_db_connection

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token():
    """Generate a secure random token"""
    return secrets.token_urlsafe(32)

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.json
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')

    # Validation
    if not username or not email or not password:
        return jsonify({"message": "All fields are required"}), 400

    if len(password) < 6:
        return jsonify({"message": "Password must be at least 6 characters"}), 400

    try:
        conn = get_db_connection()
        
        # Check if username already exists
        existing_user = conn.execute(
            'SELECT id FROM users WHERE username = ?', (username,)
        ).fetchone()
        
        if existing_user:
            conn.close()
            return jsonify({"message": "Username already exists"}), 400

        # Check if email already exists
        existing_email = conn.execute(
            'SELECT id FROM users WHERE email = ?', (email,)
        ).fetchone()
        
        if existing_email:
            conn.close()
            return jsonify({"message": "Email already registered"}), 400

        # Hash password and create user
        hashed_password = hash_password(password)
        
        conn.execute('''
            INSERT INTO users (username, email, password, created_at)
            VALUES (?, ?, ?, ?)
        ''', (username, email, hashed_password, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()

        return jsonify({
            "message": "Registration successful",
            "username": username
        }), 201

    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({"message": "Registration failed"}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """Login user"""
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"message": "Username and password required"}), 400

    try:
        conn = get_db_connection()
        
        # Get user from database
        user = conn.execute('''
            SELECT id, username, email, password, created_at
            FROM users 
            WHERE username = ?
        ''', (username,)).fetchone()

        if not user:
            conn.close()
            return jsonify({"message": "Invalid username or password"}), 401

        # Verify password
        hashed_password = hash_password(password)
        if user['password'] != hashed_password:
            conn.close()
            return jsonify({"message": "Invalid username or password"}), 401

        # Generate session token
        token = generate_token()
        
        # Update last login
        conn.execute('''
            UPDATE users 
            SET last_login = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), user['id']))
        
        conn.commit()
        conn.close()

        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {
                "id": user['id'],
                "username": user['username'],
                "email": user['email'],
                "created_at": user['created_at']
            }
        }), 200

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"message": "Login failed"}), 500


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """Logout user (client-side token removal)"""
    return jsonify({"message": "Logout successful"}), 200


@auth_bp.route('/verify', methods=['GET'])
def verify_token():
    """Verify if user is authenticated (client-side check)"""
    # In a production app, you'd verify the token from headers
    # For now, this is a simple endpoint for token validation
    return jsonify({"authenticated": True}), 200