"""Database initialization and connection management"""
import sqlite3
import json
from typing import List
from pathlib import Path

from config import DATABASE_PATH, SEED_DATA_PATH
from src.models.story import Story


def init_database():
    """Initialize the SQLite database with schema"""
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


def seed_database_from_json():
    """Load seed data from JSON file into database"""
    if not Path(SEED_DATA_PATH).exists():
        print(f"✗ Seed data file not found at {SEED_DATA_PATH}")
        print("  Run seed_data_generator.py first to create sample data")
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Clear existing data
    cursor.execute('DELETE FROM stories')
    
    # Load from JSON
    with open(SEED_DATA_PATH, 'r') as f:
        stories_data = json.load(f)
    
    # Insert stories
    for story_data in stories_data:
        cursor.execute('''
            INSERT INTO stories 
            (id, description, estimated_points, actual_points, days_to_complete, 
             caused_spillover, risk_level, epic, reporter)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            story_data['id'],
            story_data['description'],
            story_data['estimated_points'],
            story_data['actual_points'],
            story_data['days_to_complete'],
            story_data['caused_spillover'],
            story_data['risk_level'],
            story_data.get('epic'),
            story_data.get('reporter')
        ))
    
    conn.commit()
    count = cursor.execute('SELECT COUNT(*) FROM stories').fetchone()[0]
    conn.close()
    
    print(f"✓ Database seeded with {count} stories from {SEED_DATA_PATH}")


def get_database_connection():
    """Get a connection to the SQLite database"""
    return sqlite3.connect(DATABASE_PATH)


if __name__ == '__main__':
    # For standalone execution
    Path('data').mkdir(exist_ok=True)
    init_database()
    seed_database_from_json()

