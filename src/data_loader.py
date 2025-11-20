"""
Data loader abstraction layer for accessing story data.
This abstraction allows for future integration with external sources (e.g., OpenProject API).
"""
import sqlite3
import pandas as pd
import os
from typing import List, Optional
from abc import ABC, abstractmethod

from config import DATABASE_PATH, NEODATASET_PATH, NEODATASET_HIGH_CONF_PATH
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


class CSVDataLoader(DataLoaderInterface):
    """
    CSV implementation of data loader for augmented NeoDataset.
    This is the primary data source for SprintGuard.
    """
    
    def __init__(self, csv_path: str = NEODATASET_PATH, high_conf_only: bool = False):
        """
        Initialize CSV data loader.
        
        Args:
            csv_path: Path to augmented NeoDataset CSV
            high_conf_only: If True, use only high-confidence subset
        """
        if high_conf_only:
            csv_path = NEODATASET_HIGH_CONF_PATH
        
        self.csv_path = csv_path
        self._df = None
        self._load_data()
    
    def _load_data(self):
        """Load data from CSV file"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"ERROR: Augmented NeoDataset not found at:\n"
                f"  {self.csv_path}\n\n"
                f"Please run the augmentation pipeline first:\n"
                f"  1. Install ML dependencies: pip install -r requirements-ml.txt\n"
                f"  2. Run augmentation: python scripts/augment_neodataset.py\n"
                f"{'='*70}\n"
            )
        
        self._df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(self._df)} stories from {os.path.basename(self.csv_path)}")
    
    def _row_to_story(self, row) -> Story:
        """Convert CSV row (pandas Series) to Story object"""
        return Story(
            id=int(row.get('id', row.name)),  # Use index if no id column
            description=row.get('description', ''),
            estimated_points=int(row.get('story_points', 0)) if pd.notna(row.get('story_points')) else None,
            actual_points=int(row.get('story_points', 0)) if pd.notna(row.get('story_points')) else None,
            days_to_complete=None,  # Not in NeoDataset
            caused_spillover=None,  # Not in NeoDataset
            risk_level=row.get('risk_label', 'Unknown'),  # From augmentation
            epic=None,
            reporter=None
        )
    
    def get_all_stories(self) -> List[Story]:
        """Retrieve all historical stories"""
        return [self._row_to_story(row) for _, row in self._df.iterrows()]
    
    def get_story_by_id(self, story_id: int) -> Optional[Story]:
        """Retrieve a specific story by ID"""
        matching = self._df[self._df['id'] == story_id]
        if matching.empty:
            return None
        return self._row_to_story(matching.iloc[0])
    
    def get_stories_by_risk_level(self, risk_level: str) -> List[Story]:
        """Retrieve stories filtered by risk level"""
        # Map risk_level to risk_label in NeoDataset
        # risk_level can be "Low", "Medium", "High"
        # risk_label in NeoDataset is "SAFE" or "RISK"
        
        if risk_level.lower() in ['low', 'safe']:
            filtered = self._df[self._df['risk_label'] == 'SAFE']
        elif risk_level.lower() in ['high', 'risk']:
            filtered = self._df[self._df['risk_label'] == 'RISK']
        else:  # Medium - return subset of RISK stories
            filtered = self._df[self._df['risk_label'] == 'RISK']
        
        return [self._row_to_story(row) for _, row in filtered.iterrows()]
    
    def get_story_count(self) -> int:
        """Get total number of stories"""
        return len(self._df)
    
    def get_average_completion_time(self, story_points: int) -> float:
        """Get average completion time (not available in NeoDataset)"""
        return 0.0  # NeoDataset doesn't have completion time data


# Factory function for easy instantiation
def get_data_loader(use_neodataset: bool = True, high_conf_only: bool = False) -> DataLoaderInterface:
    """
    Factory function to get the appropriate data loader.
    
    Args:
        use_neodataset: If True, use CSV loader for NeoDataset (default)
        high_conf_only: If True, use only high-confidence subset
    
    Returns:
        DataLoaderInterface implementation
    """
    if use_neodataset:
        return CSVDataLoader(high_conf_only=high_conf_only)
    else:
        return SQLiteDataLoader()

