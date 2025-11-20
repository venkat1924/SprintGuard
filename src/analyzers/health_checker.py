"""
Data Health Check analyzer - Assesses quality and quantity of historical data.
Provides transparency about prediction reliability.
"""
from typing import List, Dict
from dataclasses import dataclass

from src.models.story import Story
from config import (
    HEALTH_CHECK_MIN_STORIES,
    HEALTH_CHECK_MIN_DESCRIPTION_LENGTH,
    HEALTH_CHECK_IDEAL_DESCRIPTION_LENGTH
)


@dataclass
class HealthCheckResult:
    """Result of data health check analysis"""
    overall_grade: str  # A, B, C, D, F
    overall_score: float  # 0-100
    volume_score: float
    completeness_score: float
    quality_score: float
    consistency_score: float
    story_count: int
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            'overall_grade': self.overall_grade,
            'overall_score': round(self.overall_score, 1),
            'volume_score': round(self.volume_score, 1),
            'completeness_score': round(self.completeness_score, 1),
            'quality_score': round(self.quality_score, 1),
            'consistency_score': round(self.consistency_score, 1),
            'story_count': self.story_count,
            'recommendations': self.recommendations
        }


class HealthChecker:
    """
    Analyzes historical story data to assess its quality and suitability
    for predictive analytics.
    """
    
    def __init__(self, stories: List[Story]):
        self.stories = stories
        self.story_count = len(stories)
    
    def assess(self) -> HealthCheckResult:
        """Perform complete health check analysis"""
        volume_score = self._calculate_volume_score()
        completeness_score = self._calculate_completeness_score()
        quality_score = self._calculate_quality_score()
        consistency_score = self._calculate_consistency_score()
        
        # Overall score is weighted average
        overall_score = (
            volume_score * 0.30 +
            completeness_score * 0.25 +
            quality_score * 0.25 +
            consistency_score * 0.20
        )
        
        overall_grade = self._score_to_grade(overall_score)
        recommendations = self._generate_recommendations(
            volume_score, completeness_score, quality_score, consistency_score
        )
        
        return HealthCheckResult(
            overall_grade=overall_grade,
            overall_score=overall_score,
            volume_score=volume_score,
            completeness_score=completeness_score,
            quality_score=quality_score,
            consistency_score=consistency_score,
            story_count=self.story_count,
            recommendations=recommendations
        )
    
    def _calculate_volume_score(self) -> float:
        """
        Score based on number of stories available.
        Minimum threshold: 100 stories for good predictions.
        """
        if self.story_count >= HEALTH_CHECK_MIN_STORIES:
            return 100.0
        else:
            # Linear scale up to minimum
            return (self.story_count / HEALTH_CHECK_MIN_STORIES) * 100
    
    def _calculate_completeness_score(self) -> float:
        """
        Score based on how complete the data is (no missing fields).
        """
        if not self.stories:
            return 0.0
        
        complete_count = 0
        for story in self.stories:
            # Check if all critical fields are present and valid
            has_description = bool(story.description and len(story.description.strip()) > 0)
            has_valid_points = story.estimated_points > 0 and story.actual_points > 0
            has_completion_data = story.days_to_complete > 0
            
            if has_description and has_valid_points and has_completion_data:
                complete_count += 1
        
        return (complete_count / self.story_count) * 100
    
    def _calculate_quality_score(self) -> float:
        """
        Score based on description quality (length, specificity).
        """
        if not self.stories:
            return 0.0
        
        quality_points = 0
        max_possible_points = len(self.stories) * 3  # 3 points per story
        
        for story in self.stories:
            words = story.description.split()
            word_count = len(words)
            
            # Award points for description length
            if word_count >= HEALTH_CHECK_IDEAL_DESCRIPTION_LENGTH:
                quality_points += 2
            elif word_count >= HEALTH_CHECK_MIN_DESCRIPTION_LENGTH:
                quality_points += 1
            
            # Award point for specificity (not too generic)
            generic_phrases = ['fix bug', 'update', 'change', 'modify']
            is_specific = not any(
                phrase in story.description.lower() 
                for phrase in generic_phrases
            ) or word_count > 5
            
            if is_specific:
                quality_points += 1
        
        return (quality_points / max_possible_points) * 100
    
    def _calculate_consistency_score(self) -> float:
        """
        Score based on story point distribution and estimation patterns.
        """
        if not self.stories:
            return 0.0
        
        # Check for Fibonacci-like pattern in story points
        common_points = [1, 2, 3, 5, 8, 13]
        fibonacci_count = sum(
            1 for story in self.stories 
            if story.estimated_points in common_points
        )
        fibonacci_ratio = fibonacci_count / self.story_count
        
        # Check for diversity (not all the same points)
        unique_points = len(set(story.estimated_points for story in self.stories))
        diversity_score = min(unique_points / 5, 1.0)  # Ideal: at least 5 different values
        
        # Check for reasonable estimation (actual vs estimated correlation)
        # Higher correlation = more consistent estimation practice
        estimation_ratios = [
            story.estimation_accuracy() 
            for story in self.stories 
            if story.estimated_points > 0
        ]
        
        if estimation_ratios:
            # Stories consistently within 0.5x to 2x of estimate = good
            reasonable_estimates = sum(
                1 for ratio in estimation_ratios 
                if 0.5 <= ratio <= 2.0
            )
            estimation_quality = reasonable_estimates / len(estimation_ratios)
        else:
            estimation_quality = 0.0
        
        # Weighted combination
        consistency = (
            fibonacci_ratio * 0.3 +
            diversity_score * 0.3 +
            estimation_quality * 0.4
        )
        
        return consistency * 100
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 75:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 40:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(
        self, 
        volume: float, 
        completeness: float, 
        quality: float, 
        consistency: float
    ) -> List[str]:
        """Generate actionable recommendations based on scores"""
        recommendations = []
        
        if volume < 70:
            recommendations.append(
                f"Collect more historical data. You have {self.story_count} stories, "
                f"but {HEALTH_CHECK_MIN_STORIES}+ is recommended for reliable predictions."
            )
        
        if completeness < 80:
            recommendations.append(
                "Some stories are missing critical data (descriptions, story points, or completion info). "
                "Ensure all future stories are fully documented."
            )
        
        if quality < 70:
            recommendations.append(
                "Story descriptions are too brief or generic. Add more detail to descriptions "
                f"(aim for {HEALTH_CHECK_IDEAL_DESCRIPTION_LENGTH}+ words) to improve prediction accuracy."
            )
        
        if consistency < 70:
            recommendations.append(
                "Story point estimates show inconsistent patterns. Consider using standard "
                "Fibonacci sequence (1, 2, 3, 5, 8, 13) and conducting team calibration sessions."
            )
        
        if not recommendations:
            recommendations.append(
                "Excellent data quality! Your historical data is well-suited for accurate predictions."
            )
        
        return recommendations

