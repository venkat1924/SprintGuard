"""Configuration settings for SprintGuard PoC"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database configuration
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'sprintguard.db')
SEED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'seed_stories.json')

# Flask configuration
DEBUG = True
HOST = '0.0.0.0'
PORT = 5001

# Health Check thresholds
HEALTH_CHECK_MIN_STORIES = 100
HEALTH_CHECK_MIN_DESCRIPTION_LENGTH = 5
HEALTH_CHECK_IDEAL_DESCRIPTION_LENGTH = 15

# Risk Assessment keywords
HIGH_RISK_KEYWORDS = [
    'api', 'integration', 'migrate', 'migration', 'database', 'db',
    'third-party', 'legacy', 'security', 'authentication', 'auth',
    'performance', 'refactor', 'refactoring', 'architecture',
    'infrastructure', 'deploy', 'deployment', 'scale', 'scaling'
]

MEDIUM_RISK_KEYWORDS = [
    'implement', 'create', 'build', 'develop', 'feature',
    'backend', 'frontend', 'service', 'component', 'module',
    'test', 'testing', 'validation', 'error', 'bug'
]

