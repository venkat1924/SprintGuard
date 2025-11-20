"""
Database initialization and connection management

DEPRECATION NOTICE:
    SQLite is now optional and primarily kept for potential future use cases
    such as user annotations, custom story tracking, or offline caching.
    
    The primary data source for SprintGuard is the augmented NeoDataset CSV.
    Use CSVDataLoader from src.data_loader for accessing story data.
"""
import sqlite3
from pathlib import Path

from config import DATABASE_PATH


def init_database():
    """
    Initialize the SQLite database with schema.
    
    Note: This is optional and kept for potential future features like
    user annotations, custom stories, or caching.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create stories table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            estimated_points INTEGER NOT NULL,
            actual_points INTEGER NOT NULL,
            days_to_complete INTEGER NOT NULL,
            caused_spillover BOOLEAN NOT NULL,
            risk_level TEXT NOT NULL,
            epic TEXT,
            reporter TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized at {DATABASE_PATH}")


def get_database_connection():
    """Get a connection to the SQLite database"""
    return sqlite3.connect(DATABASE_PATH)


if __name__ == '__main__':
    print("="*70)
    print("DEPRECATION NOTICE")
    print("="*70)
    print("SQLite database is now optional.")
    print("Primary data source: Augmented NeoDataset CSV")
    print("\nTo set up SprintGuard:")
    print("  1. Install ML dependencies: pip install -r requirements-ml.txt")
    print("  2. Run augmentation: python scripts/augment_neodataset.py")
    print("  3. Start app: python app.py")
    print("="*70)

