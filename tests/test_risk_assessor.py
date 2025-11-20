"""Tests for Risk Assessor implementations"""
import pytest
from src.models.story import Story
from src.analyzers.keyword_risk_assessor import KeywordRiskAssessor


def create_historical_stories():
    """Create sample historical stories for testing"""
    stories = [
        Story(
            id=1,
            description="Implement API authentication endpoint",
            estimated_points=5,
            actual_points=8,
            days_to_complete=5,
            caused_spillover=True,
            risk_level="High"
        ),
        Story(
            id=2,
            description="Update button color",
            estimated_points=1,
            actual_points=1,
            days_to_complete=1,
            caused_spillover=False,
            risk_level="Low"
        ),
        Story(
            id=3,
            description="Create user profile page",
            estimated_points=3,
            actual_points=3,
            days_to_complete=2,
            caused_spillover=False,
            risk_level="Medium"
        ),
    ]
    return stories


def test_keyword_assessor_initialization():
    """Test KeywordRiskAssessor can be initialized"""
    stories = create_historical_stories()
    assessor = KeywordRiskAssessor(stories)
    assert assessor.get_name() == "KeywordRiskAssessor"
    assert len(assessor.historical_stories) == 3


def test_high_risk_detection():
    """Test detection of high-risk stories"""
    stories = create_historical_stories()
    assessor = KeywordRiskAssessor(stories)
    
    # Story with high-risk keywords
    result = assessor.assess("Implement OAuth2 API integration with third-party service")
    assert result.risk_level == "High"
    assert result.confidence > 50


def test_low_risk_detection():
    """Test detection of low-risk stories"""
    stories = create_historical_stories()
    assessor = KeywordRiskAssessor(stories)
    
    # Simple story
    result = assessor.assess("Update copyright year in footer")
    assert result.risk_level == "Low"
    assert len(result.explanation) > 0


def test_result_to_dict():
    """Test RiskResult can be converted to dictionary"""
    stories = create_historical_stories()
    assessor = KeywordRiskAssessor(stories)
    
    result = assessor.assess("Test story")
    result_dict = result.to_dict()
    
    assert 'risk_level' in result_dict
    assert 'confidence' in result_dict
    assert 'explanation' in result_dict


def test_empty_description_handling():
    """Test handling of edge cases"""
    stories = create_historical_stories()
    assessor = KeywordRiskAssessor(stories)
    
    # Very short description
    result = assessor.assess("Fix")
    assert result.risk_level in ["Low", "Medium", "High"]

