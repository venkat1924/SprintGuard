"""Story data model representing a user story or task"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Story:
    """
    Represents a software development user story with estimation data.
    
    Attributes:
        id: Unique identifier for the story
        description: Text description of the story
        estimated_points: Initial story point estimate (may be None for NeoDataset)
        actual_points: Actual story points after completion (may be None for NeoDataset)
        days_to_complete: Number of days taken to complete (may be None for NeoDataset)
        caused_spillover: Whether this story caused sprint spillover (may be None for NeoDataset)
        risk_level: Historical risk classification (Low/Medium/High/SAFE/RISK)
        epic: Optional epic/category this story belongs to
        reporter: Optional person who created the story
        
    Note: NeoDataset does not provide actual_points, days_to_complete, or caused_spillover.
          These fields will be None when loading from CSV.
    """
    id: int
    description: str
    estimated_points: Optional[int]
    actual_points: Optional[int]
    days_to_complete: Optional[int]
    caused_spillover: Optional[bool]
    risk_level: str
    epic: Optional[str] = None
    reporter: Optional[str] = None
    
    def was_underestimated(self) -> Optional[bool]:
        """
        Returns True if actual effort exceeded estimate by 50%+
        Returns None if actual_points or estimated_points are not available
        """
        if self.actual_points is None or self.estimated_points is None:
            return None
        if self.estimated_points == 0:
            return None
        return self.actual_points > self.estimated_points * 1.5
    
    def estimation_accuracy(self) -> Optional[float]:
        """
        Returns ratio of actual to estimated points (1.0 = perfect)
        Returns None if actual_points or estimated_points are not available
        """
        if self.actual_points is None or self.estimated_points is None:
            return None
        if self.estimated_points == 0:
            return None
        return self.actual_points / self.estimated_points
    
    def to_dict(self) -> dict:
        """Convert story to dictionary representation"""
        return {
            'id': self.id,
            'description': self.description,
            'estimated_points': self.estimated_points,
            'actual_points': self.actual_points,
            'days_to_complete': self.days_to_complete,
            'caused_spillover': self.caused_spillover,
            'risk_level': self.risk_level,
            'epic': self.epic,
            'reporter': self.reporter
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Story':
        """Create Story instance from dictionary"""
        return cls(
            id=data['id'],
            description=data['description'],
            estimated_points=data['estimated_points'],
            actual_points=data['actual_points'],
            days_to_complete=data['days_to_complete'],
            caused_spillover=data['caused_spillover'],
            risk_level=data['risk_level'],
            epic=data.get('epic'),
            reporter=data.get('reporter')
        )

