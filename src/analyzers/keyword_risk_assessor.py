"""
Keyword-based risk assessor implementation (V1).
Simple heuristic-based approach using keyword matching.
"""
from typing import List, Set
import re

from src.models.story import Story
from src.analyzers.risk_assessor_interface import RiskAssessorInterface, RiskResult
from config import HIGH_RISK_KEYWORDS, MEDIUM_RISK_KEYWORDS


class KeywordRiskAssessor(RiskAssessorInterface):
    """
    V1 Risk Assessor using keyword matching heuristics.
    
    This is a simple, interpretable baseline that can be replaced with
    ML-based approaches without changing other code.
    """
    
    def __init__(self, historical_stories: List[Story]):
        super().__init__(historical_stories)
        
        # Learn from historical data: which keywords correlated with high risk outcomes?
        self.learned_high_risk_keywords = self._learn_risk_keywords()
    
    def _learn_risk_keywords(self) -> Set[str]:
        """
        Analyze historical data to identify words that frequently appear
        in stories that caused problems (spillover, underestimation).
        """
        problematic_stories = [
            story for story in self.historical_stories
            if story.caused_spillover or story.was_underestimated()
        ]
        
        # Extract common words from problematic stories
        word_freq = {}
        for story in problematic_stories:
            words = re.findall(r'\b\w+\b', story.description.lower())
            for word in words:
                if len(word) > 3:  # Ignore short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words that appear in at least 3 problematic stories
        learned_keywords = {
            word for word, count in word_freq.items() 
            if count >= 3
        }
        
        return learned_keywords
    
    def assess(self, description: str) -> RiskResult:
        """
        Assess risk based on keyword matching and historical patterns.
        
        Algorithm:
        1. Check for configured high-risk keywords
        2. Check for learned high-risk patterns from historical data
        3. Check for medium-risk keywords
        4. Consider description length and complexity
        5. Return risk classification with explanation
        """
        description_lower = description.lower()
        words = set(re.findall(r'\b\w+\b', description_lower))
        
        # Find matching keywords
        high_risk_matches = [
            kw for kw in HIGH_RISK_KEYWORDS 
            if kw in description_lower
        ]
        
        learned_matches = [
            kw for kw in self.learned_high_risk_keywords 
            if kw in words
        ]
        
        medium_risk_matches = [
            kw for kw in MEDIUM_RISK_KEYWORDS 
            if kw in description_lower
        ]
        
        # Calculate risk score
        risk_score = 0
        explanation_parts = []
        
        # High risk indicators
        if high_risk_matches:
            risk_score += len(high_risk_matches) * 3
            explanation_parts.append(
                f"Contains high-complexity keywords: {', '.join(high_risk_matches[:3])}"
            )
        
        # Learned patterns from historical data
        if learned_matches:
            risk_score += len(learned_matches) * 2
            explanation_parts.append(
                f"Similar to past problematic stories (keywords: {', '.join(list(learned_matches)[:3])})"
            )
        
        # Medium risk indicators
        if medium_risk_matches:
            risk_score += len(medium_risk_matches) * 1
        
        # Description complexity
        word_count = len(description.split())
        if word_count > 20:
            risk_score += 2
            explanation_parts.append("Long, complex description suggests multiple requirements")
        elif word_count < 5:
            risk_score += 1
            explanation_parts.append("Vague or incomplete description (risk of misunderstanding)")
        
        # Find similar historical stories
        similar_stories = self._find_similar_stories(description_lower, words)
        
        # Classify based on score
        if risk_score >= 6:
            risk_level = "High"
            confidence = min(70 + risk_score * 3, 95)
            if not explanation_parts:
                explanation_parts.append("Multiple complexity indicators detected")
        elif risk_score >= 2:
            risk_level = "Medium"
            confidence = 60 + risk_score * 5
            if not explanation_parts:
                explanation_parts.append("Moderate complexity keywords present")
        else:
            risk_level = "Low"
            confidence = 70
            explanation_parts.append("Straightforward story with no high-risk indicators")
        
        # Add historical context
        if similar_stories:
            high_risk_similar = [
                sid for sid in similar_stories
                if any(s.id == sid and s.risk_level == "High" for s in self.historical_stories)
            ]
            if high_risk_similar:
                explanation_parts.append(
                    f"Similar to {len(high_risk_similar)} past high-risk stories"
                )
        
        explanation = ". ".join(explanation_parts) + "."
        
        return RiskResult(
            risk_level=risk_level,
            confidence=confidence,
            explanation=explanation,
            similar_stories=similar_stories[:5] if similar_stories else None
        )
    
    def _find_similar_stories(self, description_lower: str, words: Set[str]) -> List[int]:
        """
        Find historical stories with similar keywords (simple similarity).
        Returns list of story IDs.
        """
        similar = []
        
        for story in self.historical_stories:
            story_words = set(re.findall(r'\b\w+\b', story.description.lower()))
            
            # Calculate simple word overlap
            common_words = words.intersection(story_words)
            
            # Ignore very common words
            common_words = {w for w in common_words if len(w) > 4}
            
            if len(common_words) >= 3:
                similar.append(story.id)
        
        return similar[:10]  # Return top 10 similar stories

