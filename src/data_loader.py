"""
Data loader abstraction layer for accessing story data.
This abstraction allows for future integration with external sources (e.g., OpenProject API).
"""
import sqlite3
from typing import List, Optional
from abc import ABC, abstractmethod

from config import DATABASE_PATH
from src.models.story import Story


class DataLoaderInterface(ABC):
    """Abstract interface for data loading - future-proof for API integration"""
    
    @abstractmethod
    def get_all_stories(self) -> List[Story]:
        """Retrieve all historical stories"""
        pass
    
    @abstractmethod
    def get_story_by_id(self, story_id: int) -> Optional[Story]:
        """Retrieve a specific story by ID"""
        pass
    
    @abstractmethod
    def get_stories_by_risk_level(self, risk_level: str) -> List[Story]:
        """Retrieve stories filtered by risk level (Low/Medium/High)"""
        pass
    
    @abstractmethod
    def get_story_count(self) -> int:
        """Get total number of stories in the system"""
        pass


class SQLiteDataLoader(DataLoaderInterface):
    """SQLite implementation of data loader"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _row_to_story(self, row: sqlite3.Row) -> Story:
        """Convert database row to Story object"""
        return Story(
            id=row['id'],
            description=row['description'],
            estimated_points=row['estimated_points'],
            actual_points=row['actual_points'],
            days_to_complete=row['days_to_complete'],
            caused_spillover=bool(row['caused_spillover']),
            risk_level=row['risk_level'],
            epic=row['epic'],
            reporter=row['reporter']
        )
    
    def get_all_stories(self) -> List[Story]:
        """Retrieve all historical stories"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM stories ORDER BY id')
        stories = [self._row_to_story(row) for row in cursor.fetchall()]
        conn.close()
        return stories
    
    def get_story_by_id(self, story_id: int) -> Optional[Story]:
        """Retrieve a specific story by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM stories WHERE id = ?', (story_id,))
        row = cursor.fetchone()
        conn.close()
        
        return self._row_to_story(row) if row else None
    
    def get_stories_by_risk_level(self, risk_level: str) -> List[Story]:
        """Retrieve stories filtered by risk level"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM stories WHERE risk_level = ? ORDER BY id',
            (risk_level,)
        )
        stories = [self._row_to_story(row) for row in cursor.fetchall()]
        conn.close()
        return stories
    
    def get_story_count(self) -> int:
        """Get total number of stories"""
        conn = self._get_connection()
        cursor = conn.cursor()
        count = cursor.execute('SELECT COUNT(*) FROM stories').fetchone()[0]
        conn.close()
        return count
    
    def get_average_completion_time(self, story_points: int) -> float:
        """Get average completion time for stories with given points"""
        conn = self._get_connection()
        cursor = conn.cursor()
        result = cursor.execute(
            'SELECT AVG(days_to_complete) FROM stories WHERE estimated_points = ?',
            (story_points,)
        ).fetchone()[0]
        conn.close()
        return result if result else 0.0


# Factory function for easy instantiation
def get_data_loader() -> DataLoaderInterface:
    """
    Factory function to get the appropriate data loader.
    Currently returns SQLite loader, but can be extended for API sources.
    """
    return SQLiteDataLoader()

