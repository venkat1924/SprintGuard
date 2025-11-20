"""Tests for Scope Simulator"""
import pytest
from datetime import datetime, timedelta
from src.analyzers.scope_simulator import ScopeSimulator


def test_scope_simulator_initialization():
    """Test ScopeSimulator can be initialized"""
    simulator = ScopeSimulator(team_velocity=2.5)
    assert simulator.team_velocity == 2.5


def test_days_calculation():
    """Test calculation of days needed"""
    simulator = ScopeSimulator(team_velocity=2.0)
    
    # 4 points at 2 points/day = 2 days
    days = simulator._calculate_days_needed(4, 2.0)
    assert days == 2
    
    # 5 points at 2 points/day = 3 days (rounded up)
    days = simulator._calculate_days_needed(5, 2.0)
    assert days == 3


def test_business_days_addition():
    """Test adding business days (skip weekends)"""
    simulator = ScopeSimulator()
    
    # Start on Monday (2025-11-17)
    start_date = datetime(2025, 11, 17)
    
    # Add 3 business days should give us Thursday
    new_date = simulator._add_business_days(start_date, 3)
    assert new_date.weekday() == 3  # Thursday


def test_full_simulation():
    """Test complete simulation"""
    simulator = ScopeSimulator(team_velocity=2.0)
    
    result = simulator.simulate_scope_addition(
        current_end_date="2025-12-15",
        current_story_points=20,
        new_story_points=4,
        team_velocity=2.0
    )
    
    assert result.days_added == 2
    assert result.points_added == 4
    assert result.new_points == 24
    assert len(result.recommendations) > 0
    
    # Can convert to dict
    result_dict = result.to_dict()
    assert 'original_end_date' in result_dict
    assert 'new_end_date' in result_dict


def test_invalid_velocity():
    """Test handling of invalid velocity"""
    simulator = ScopeSimulator()
    
    with pytest.raises(ValueError):
        simulator._calculate_days_needed(5, 0)


def test_velocity_calculation():
    """Test team velocity calculation from historical data"""
    simulator = ScopeSimulator()
    
    # 20 points in 10 days = 2 points/day
    velocity = simulator.calculate_team_velocity([5, 5, 10], 10)
    assert velocity == 2.0

