"""
Database Module - SQLite Version
Handles database initialization and connections for PackerVision AI
Updated: Added `shift` column to reports table for shift-wise tracking.
"""

import sqlite3
import os

DB_PATH = 'packer_monitoring.db'

def get_db_connection():
    """Create and return a database connection with Row factory"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with all required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Users Table - FOR AUTHENTICATION
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    ''')
    
    # 2. Packers Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS packers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT,
            spouts INTEGER,
            rpm REAL,
            camera_id TEXT,
            line_position REAL,
            start_line_position REAL,
            confidence_threshold REAL,
            status TEXT DEFAULT 'idle',
            session_id TEXT,
            created_at TEXT
        )
    ''')

    # 3. Cameras Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            rtsp_url TEXT NOT NULL,
            location TEXT,
            status TEXT DEFAULT 'online',
            is_assigned INTEGER DEFAULT 0,
            assigned_packer_id TEXT,
            is_video_file INTEGER DEFAULT 0,
            created_at TEXT
        )
    ''')

    # 4. Reports Table
    # `shift` column: 'A' / 'B' / 'C'  — NULL for manually-triggered reports
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            packer_id TEXT,
            packer_name TEXT,
            shift TEXT,
            total_events INTEGER,
            total_cycles REAL,
            bags_placed INTEGER,
            bags_missed INTEGER,
            stuck_bags INTEGER,
            packer_efficiency REAL,
            manual_efficiency REAL,
            elapsed_time REAL,
            timestamp TEXT,
            FOREIGN KEY (packer_id) REFERENCES packers (id)
        )
    ''')
    
    # 5. Monitoring Sessions Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS monitoring_sessions (
            session_id TEXT PRIMARY KEY,
            packer_id TEXT,
            camera_id TEXT,
            shift TEXT,
            started_at TEXT,
            ended_at TEXT,
            status TEXT,
            FOREIGN KEY (packer_id) REFERENCES packers (id),
            FOREIGN KEY (camera_id) REFERENCES cameras (id)
        )
    ''')
    
    conn.commit()

    # ── Migration: add shift column to existing DBs ────────────────────────────
    # Safe to run on a fresh DB too — PRAGMA returns [] if the column exists.
    _add_column_if_missing(conn, "reports", "shift", "TEXT")
    _add_column_if_missing(conn, "monitoring_sessions", "shift", "TEXT")

    conn.close()
    print("✓ Database initialized successfully with all tables")


def _add_column_if_missing(conn, table: str, column: str, col_type: str):
    """Add a column to an existing table only if it does not already exist."""
    existing = [
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    ]
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()
        print(f"  ↳ Migration: added '{column}' column to '{table}'")


def create_default_admin():
    """Create a default admin user if no users exist"""
    import hashlib
    from datetime import datetime
    
    conn = get_db_connection()
    
    # Check if any users exist
    user_count = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
    
    if user_count == 0:
        # Create default admin user
        default_password = hashlib.sha256('admin123'.encode()).hexdigest()
        
        conn.execute('''
            INSERT INTO users (username, email, password, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', 'admin@packervision.ai', default_password, 'admin', datetime.now().isoformat()))
        
        conn.commit()
        print("✓ Default admin user created (username: admin, password: admin123)")
    
    conn.close()

if __name__ == "__main__":
    init_db()
    create_default_admin()
    print("\n" + "="*60)
    print("Database setup complete!")
    print("="*60)
    print("\nDefault Admin Credentials:")
    print("  Username: admin")
    print("  Password: admin123")
    print("\n⚠️  Please change the default password after first login!")
    print("="*60)
