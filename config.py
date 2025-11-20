"""Configuration settings for SprintGuard PoC"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Database configuration (optional - for future user annotations)
DATABASE_PATH = os.path.join(DATA_DIR, 'sprintguard.db')

# NeoDataset paths (primary data source)
NEODATASET_PATH = os.path.join(DATA_DIR, 'neodataset_augmented.csv')
NEODATASET_HIGH_CONF_PATH = os.path.join(DATA_DIR, 'neodataset_augmented_high_confidence.csv')

# Flask configuration
DEBUG = True
HOST = '0.0.0.0'
PORT = 5001

# Health Check thresholds
HEALTH_CHECK_MIN_STORIES = 100
HEALTH_CHECK_MIN_DESCRIPTION_LENGTH = 5
HEALTH_CHECK_IDEAL_DESCRIPTION_LENGTH = 15

