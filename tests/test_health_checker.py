"""Tests for HealthChecker module"""
import pytest
from src.models.story import Story
from src.analyzers.health_checker import HealthChecker


def create_sample_stories(count=10):
    """Helper to create sample stories for testing"""
    stories = []
    for i in range(count):
        story = Story(
            id=i + 1,
            description=f"Test story {i + 1} with sufficient detail",
            estimated_points=3,
            actual_points=3,
            days_to_complete=2,
            caused_spillover=False,
            risk_level="Low"
        )
        stories.append(story)
    return stories


def test_health_checker_initialization():
    """Test HealthChecker can be initialized with stories"""
    stories = create_sample_stories(5)
    checker = HealthChecker(stories)
    assert checker.story_count == 5


def test_volume_score_calculation():
    """Test volume score is calculated correctly"""
    # With 100+ stories, should get 100%
    stories = create_sample_stories(120)
    checker = HealthChecker(stories)
    assert checker._calculate_volume_score() == 100.0
    
    # With 50 stories (half of minimum), should get 50%
    stories = create_sample_stories(50)
    checker = HealthChecker(stories)
    assert checker._calculate_volume_score() == 50.0


def test_completeness_score():
    """Test completeness score calculation"""
    stories = create_sample_stories(10)
    checker = HealthChecker(stories)
    
    # All stories are complete, should get 100%
    score = checker._calculate_completeness_score()
    assert score == 100.0


def test_score_to_grade_conversion():
    """Test score to grade conversion"""
    stories = create_sample_stories(1)
    checker = HealthChecker(stories)
    
    assert checker._score_to_grade(95) == 'A'
    assert checker._score_to_grade(80) == 'B'
    assert checker._score_to_grade(65) == 'C'
    assert checker._score_to_grade(45) == 'D'
    assert checker._score_to_grade(30) == 'F'


def test_full_assessment():
    """Test full health check assessment"""
    stories = create_sample_stories(100)
    checker = HealthChecker(stories)
    result = checker.assess()
    
    assert result.overall_grade in ['A', 'B', 'C', 'D', 'F']
    assert 0 <= result.overall_score <= 100
    assert result.story_count == 100
    assert len(result.recommendations) > 0
    
    # Can convert to dict
    result_dict = result.to_dict()
    assert 'overall_grade' in result_dict
    assert 'recommendations' in result_dict

