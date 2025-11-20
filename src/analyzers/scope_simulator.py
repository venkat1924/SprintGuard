"""
Scope Impact Simulator - Models the timeline impact of adding work to a sprint.
Visualizes the true cost of scope creep.
"""
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SimulationResult:
    """Result of scope impact simulation"""
    original_end_date: str
    new_end_date: str
    days_added: int
    original_points: int
    new_points: int
    points_added: int
    impact_summary: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            'original_end_date': self.original_end_date,
            'new_end_date': self.new_end_date,
            'days_added': self.days_added,
            'original_points': self.original_points,
            'new_points': self.new_points,
            'points_added': self.points_added,
            'impact_summary': self.impact_summary,
            'recommendations': self.recommendations
        }


class ScopeSimulator:
    """
    Simulates the timeline impact of adding scope to a sprint.
    
    This makes the consequences of scope creep tangible and immediate,
    enabling data-driven conversations about trade-offs.
    """
    
    def __init__(self, team_velocity: float = 2.0):
        """
        Initialize simulator with team's velocity.
        
        Args:
            team_velocity: Story points completed per day (default: 2.0)
        """
        self.team_velocity = team_velocity
    
    def simulate_scope_addition(
        self,
        current_end_date: str,  # ISO format: YYYY-MM-DD
        current_story_points: int,
        new_story_points: int,
        team_velocity: float = None
    ) -> SimulationResult:
        """
        Simulate adding new story points to the sprint.
        
        Args:
            current_end_date: Current planned sprint end date
            current_story_points: Current committed story points
            new_story_points: Story points to add
            team_velocity: Override default velocity (optional)
            
        Returns:
            SimulationResult with impact analysis
        """
        velocity = team_velocity if team_velocity is not None else self.team_velocity
        
        # Parse dates
        try:
            end_date = datetime.fromisoformat(current_end_date)
        except ValueError:
            # Try parsing different formats
            end_date = datetime.strptime(current_end_date, '%Y-%m-%d')
        
        # Calculate additional days needed
        days_needed = self._calculate_days_needed(new_story_points, velocity)
        
        # Calculate new end date (skip weekends)
        new_end_date = self._add_business_days(end_date, days_needed)
        
        # Generate impact summary
        total_points = current_story_points + new_story_points
        impact_summary = self._generate_impact_summary(
            current_story_points,
            new_story_points,
            days_needed,
            velocity
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            new_story_points,
            days_needed,
            current_story_points
        )
        
        return SimulationResult(
            original_end_date=end_date.strftime('%Y-%m-%d'),
            new_end_date=new_end_date.strftime('%Y-%m-%d'),
            days_added=days_needed,
            original_points=current_story_points,
            new_points=total_points,
            points_added=new_story_points,
            impact_summary=impact_summary,
            recommendations=recommendations
        )
    
    def _calculate_days_needed(self, story_points: int, velocity: float) -> int:
        """Calculate days needed for given story points"""
        if velocity <= 0:
            raise ValueError("Team velocity must be greater than 0")
        
        import math
        days = story_points / velocity
        return max(1, math.ceil(days))  # Round up, at least 1 day
    
    def _add_business_days(self, start_date: datetime, days_to_add: int) -> datetime:
        """
        Add business days to a date (skip weekends).
        
        Args:
            start_date: Starting date
            days_to_add: Number of business days to add
            
        Returns:
            New date after adding business days
        """
        current_date = start_date
        days_added = 0
        
        while days_added < days_to_add:
            current_date += timedelta(days=1)
            # Skip weekends (5=Saturday, 6=Sunday)
            if current_date.weekday() < 5:
                days_added += 1
        
        return current_date
    
    def _generate_impact_summary(
        self,
        current_points: int,
        added_points: int,
        days_added: int,
        velocity: float
    ) -> str:
        """Generate human-readable impact summary"""
        percentage_increase = (added_points / current_points * 100) if current_points > 0 else 0
        
        summary = (
            f"Adding {added_points} story points to the current {current_points} points "
            f"({percentage_increase:.1f}% increase) will require approximately "
            f"{days_added} additional business day{'s' if days_added != 1 else ''} "
            f"at your team's velocity of {velocity:.1f} points/day."
        )
        
        return summary
    
    def _generate_recommendations(
        self,
        added_points: int,
        days_added: int,
        current_points: int
    ) -> List[str]:
        """Generate actionable recommendations based on impact"""
        recommendations = []
        
        if days_added > 5:
            recommendations.append(
                "⚠️ Major timeline impact detected. Consider splitting this into a separate sprint."
            )
        elif days_added > 2:
            recommendations.append(
                "⚠️ Significant delay expected. Evaluate if this change is critical for this sprint."
            )
        
        if current_points > 0:
            percentage = (added_points / current_points) * 100
            if percentage > 30:
                recommendations.append(
                    f"💡 This represents a {percentage:.0f}% increase in scope. "
                    "Consider removing lower-priority items to maintain sprint predictability."
                )
        
        if added_points >= 8:
            recommendations.append(
                "💡 Large story detected. Can it be broken into smaller, more manageable pieces?"
            )
        
        recommendations.append(
            "✓ Document this scope change decision and share with all stakeholders."
        )
        
        if not recommendations:
            recommendations.append(
                "✓ Minimal impact. This change appears manageable within the current sprint."
            )
        
        return recommendations
    
    def calculate_team_velocity(
        self,
        stories_completed: List[int],
        days_elapsed: int
    ) -> float:
        """
        Calculate team velocity from historical data.
        
        Args:
            stories_completed: List of story points completed
            days_elapsed: Number of days taken
            
        Returns:
            Velocity (points per day)
        """
        if days_elapsed <= 0:
            raise ValueError("Days elapsed must be greater than 0")
        
        total_points = sum(stories_completed)
        return total_points / days_elapsed

