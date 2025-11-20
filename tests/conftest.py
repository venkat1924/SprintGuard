"""Pytest configuration and fixtures"""
import pytest
import sys
import pandas as pd
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.story import Story
from src.data_loader import CSVDataLoader
from config import NEODATASET_PATH


@pytest.fixture
def sample_neodataset_stories() -> List[Story]:
    """
    Fixture providing a small sample of stories from augmented NeoDataset.
    Useful for testing without loading the entire dataset.
    """
    # Create sample stories mimicking NeoDataset structure
    sample_data = [
        {
            'id': 1,
            'description': 'Implement user authentication with OAuth2',
            'story_points': 8,
            'risk_label': 'RISK',
            'risk_confidence': 0.85
        },
        {
            'id': 2,
            'description': 'Fix typo in button label',
            'story_points': 1,
            'risk_label': 'SAFE',
            'risk_confidence': 0.92
        },
        {
            'id': 3,
            'description': 'Refactor legacy database schema',
            'story_points': 13,
            'risk_label': 'RISK',
            'risk_confidence': 0.78
        }
    ]
    
    stories = []
    for data in sample_data:
        story = Story(
            id=data['id'],
            description=data['description'],
            estimated_points=data['story_points'],
            actual_points=data['story_points'],
            days_to_complete=None,
            caused_spillover=None,
            risk_level=data['risk_label'],
            epic=None,
            reporter=None
        )
        stories.append(story)
    
    return stories


@pytest.fixture
def neodataset_loader():
    """
    Fixture providing CSVDataLoader for testing.
    Skips test if augmented NeoDataset not available.
    """
    try:
        loader = CSVDataLoader()
        return loader
    except FileNotFoundError:
        pytest.skip("Augmented NeoDataset not available. Run augmentation pipeline first.")

