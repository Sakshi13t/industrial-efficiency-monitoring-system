"""
Database Utilities
Common database operations and helper functions
"""

import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple, Any

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',  # Update with your password
    'database': 'packer_efficiency',
    'autocommit': False
}

@contextmanager
def get_db_cursor(dictionary=True, buffered=True):
    """
    Context manager for database connections and cursors
    
    Usage:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM table")
            results = cursor.fetchall()
    
    Args:
        dictionary: Return results as dictionaries
        buffered: Use buffered cursor
    
    Yields:
        cursor: Database cursor
    """
    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor(dictionary=dictionary, buffered=buffered)
        yield cursor
        connection.commit()
    except Error as e:
        if connection:
            connection.rollback()
        print(f"Database error: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def execute_query(query: str, params: Optional[Tuple] = None, fetch: str = 'all') -> Optional[Any]:
    """
    Execute a SELECT query and return results
    
    Args:
        query: SQL query string
        params: Query parameters
        fetch: 'all', 'one', or 'many'
    
    Returns:
        Query results or None on error
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute(query, params or ())
            
            if fetch == 'all':
                return cursor.fetchall()
            elif fetch == 'one':
                return cursor.fetchone()
            elif fetch == 'many':
                return cursor.fetchmany()
            else:
                return cursor.fetchall()
    except Error as e:
        print(f"Query execution error: {e}")
        return None


def execute_insert(query: str, params: Tuple, return_id: bool = False) -> Optional[int]:
    """
    Execute an INSERT query
    
    Args:
        query: SQL INSERT query
        params: Query parameters
        return_id: Return the last inserted ID
    
    Returns:
        Last inserted ID if return_id=True, else number of affected rows
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute(query, params)
            if return_id:
                return cursor.lastrowid
            return cursor.rowcount
    except Error as e:
        print(f"Insert error: {e}")
        return None


def execute_update(query: str, params: Tuple) -> Optional[int]:
    """
    Execute an UPDATE query
    
    Args:
        query: SQL UPDATE query
        params: Query parameters
    
    Returns:
        Number of affected rows or None on error
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount
    except Error as e:
        print(f"Update error: {e}")
        return None


def execute_delete(query: str, params: Tuple) -> Optional[int]:
    """
    Execute a DELETE query
    
    Args:
        query: SQL DELETE query
        params: Query parameters
    
    Returns:
        Number of affected rows or None on error
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount
    except Error as e:
        print(f"Delete error: {e}")
        return None


def execute_many(query: str, params_list: List[Tuple]) -> Optional[int]:
    """
    Execute a query multiple times with different parameters
    
    Args:
        query: SQL query
        params_list: List of parameter tuples
    
    Returns:
        Number of affected rows or None on error
    """
    try:
        with get_db_cursor() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount
    except Error as e:
        print(f"Batch execution error: {e}")
        return None


def call_procedure(proc_name: str, args: Optional[Tuple] = None) -> Optional[List[Dict]]:
    """
    Call a stored procedure
    
    Args:
        proc_name: Procedure name
        args: Procedure arguments
    
    Returns:
        List of result dictionaries or None on error
    """
    try:
        with get_db_cursor() as cursor:
            cursor.callproc(proc_name, args or ())
            
            # Fetch results from all result sets
            results = []
            for result in cursor.stored_results():
                results.extend(result.fetchall())
            
            return results
    except Error as e:
        print(f"Procedure call error: {e}")
        return None


def table_exists(table_name: str) -> bool:
    """
    Check if a table exists in the database
    
    Args:
        table_name: Name of the table
    
    Returns:
        True if table exists, False otherwise
    """
    query = """
        SELECT COUNT(*) as count
        FROM information_schema.tables 
        WHERE table_schema = DATABASE() 
        AND table_name = %s
    """
    result = execute_query(query, (table_name,), fetch='one')
    return result and result['count'] > 0 if result else False


def get_table_info(table_name: str) -> Optional[List[Dict]]:
    """
    Get column information for a table
    
    Args:
        table_name: Name of the table
    
    Returns:
        List of column information dictionaries
    """
    query = """
        SELECT 
            COLUMN_NAME as column_name,
            COLUMN_TYPE as column_type,
            IS_NULLABLE as is_nullable,
            COLUMN_KEY as column_key,
            COLUMN_DEFAULT as column_default,
            EXTRA as extra
        FROM information_schema.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE() 
        AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
    """
    return execute_query(query, (table_name,))


def backup_table(table_name: str, backup_suffix: Optional[str] = None) -> bool:
    """
    Create a backup copy of a table
    
    Args:
        table_name: Name of the table to backup
        backup_suffix: Optional suffix for backup table name
    
    Returns:
        True if successful, False otherwise
    """
    from datetime import datetime
    
    if not backup_suffix:
        backup_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    backup_table_name = f"{table_name}_backup_{backup_suffix}"
    
    try:
        with get_db_cursor() as cursor:
            # Create backup table structure
            cursor.execute(f"CREATE TABLE {backup_table_name} LIKE {table_name}")
            # Copy data
            cursor.execute(f"INSERT INTO {backup_table_name} SELECT * FROM {table_name}")
            print(f"Backup created: {backup_table_name}")
            return True
    except Error as e:
        print(f"Backup error: {e}")
        return False


def get_database_stats() -> Optional[Dict]:
    """
    Get overall database statistics
    
    Returns:
        Dictionary with database statistics
    """
    try:
        with get_db_cursor() as cursor:
            # Get table sizes
            cursor.execute("""
                SELECT 
                    TABLE_NAME as table_name,
                    TABLE_ROWS as row_count,
                    ROUND(((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024), 2) as size_mb
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
                ORDER BY (DATA_LENGTH + INDEX_LENGTH) DESC
            """)
            tables = cursor.fetchall()
            
            # Get total size
            cursor.execute("""
                SELECT 
                    ROUND(SUM((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024), 2) as total_size_mb
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
            """)
            total = cursor.fetchone()
            
            return {
                'tables': tables,
                'total_size_mb': total['total_size_mb'] if total else 0,
                'table_count': len(tables)
            }
    except Error as e:
        print(f"Stats error: {e}")
        return None


def test_connection() -> Dict[str, Any]:
    """
    Test database connection and return status
    
    Returns:
        Dictionary with connection status and info
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("SELECT VERSION() as version, DATABASE() as database")
        info = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        return {
            'connected': True,
            'version': info['version'],
            'database': info['database'],
            'host': DB_CONFIG['host']
        }
    except Error as e:
        return {
            'connected': False,
            'error': str(e)
        }


# Migration utilities
def run_migration(migration_sql: str) -> bool:
    """
    Run a database migration script
    
    Args:
        migration_sql: SQL migration script
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_cursor() as cursor:
            # Split and execute multiple statements
            statements = migration_sql.split(';')
            for statement in statements:
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
            print("Migration completed successfully")
            return True
    except Error as e:
        print(f"Migration error: {e}")
        return False


# Example usage functions
def get_recent_reports(limit: int = 10) -> Optional[List[Dict]]:
    """Get recent reports"""
    query = """
        SELECT * FROM vw_reports_complete 
        ORDER BY report_timestamp DESC 
        LIMIT %s
    """
    return execute_query(query, (limit,))


def get_packer_by_id(packer_id: str) -> Optional[Dict]:
    """Get packer details by ID"""
    query = "SELECT * FROM packers WHERE packer_id = %s"
    return execute_query(query, (packer_id,), fetch='one')


def update_packer_status(packer_id: str, status: str) -> bool:
    """Update packer status"""
    query = "UPDATE packers SET status = %s WHERE packer_id = %s"
    result = execute_update(query, (status, packer_id))
    return result is not None and result > 0


if __name__ == '__main__':
    # Test connection
    print("Testing database connection...")
    status = test_connection()
    print(f"Connection status: {status}")
    
    if status['connected']:
        print("\nDatabase statistics:")
        stats = get_database_stats()
        if stats:
            print(f"Total size: {stats['total_size_mb']} MB")
            print(f"Table count: {stats['table_count']}")
            print("\nTables:")
            for table in stats['tables']:
                print(f"  {table['table_name']}: {table['row_count']} rows, {table['size_mb']} MB")