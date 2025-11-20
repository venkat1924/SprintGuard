"""
Abstract interface for Risk Assessor implementations.
This allows the ML professor to plug in their own algorithm without modifying other code.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from src.models.story import Story


@dataclass
class RiskResult:
    """
    Result of risk assessment for a user story.
    
    Attributes:
        risk_level: Classification (Low, Medium, High)
        confidence: Confidence score 0-100 (optional, for ML models)
        explanation: Human-readable explanation of the assessment
        similar_stories: List of similar historical stories (optional)
    """
    risk_level: str  # "Low", "Medium", "High"
    confidence: float  # 0-100
    explanation: str
    similar_stories: List[int] = None  # Story IDs
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        result = {
            'risk_level': self.risk_level,
            'confidence': round(self.confidence, 1),
            'explanation': self.explanation
        }
        if self.similar_stories:
            result['similar_stories'] = self.similar_stories
        return result


class RiskAssessorInterface(ABC):
    """
    Abstract base class for risk assessment algorithms.
    
    Any ML implementation must inherit from this class and implement the assess() method.
    This ensures a consistent interface regardless of the underlying algorithm.
    """
    
    def __init__(self, historical_stories: List[Story]):
        """
        Initialize the assessor with historical data.
        
        Args:
            historical_stories: List of past stories for training/comparison
        """
        self.historical_stories = historical_stories
    
    @abstractmethod
    def assess(self, description: str) -> RiskResult:
        """
        Assess the risk level of a new user story.
        
        Args:
            description: Text description of the story to assess
            
        Returns:
            RiskResult object with risk level, confidence, and explanation
        """
        pass
    
    def get_name(self) -> str:
        """Return the name of this risk assessment algorithm"""
        return self.__class__.__name__

