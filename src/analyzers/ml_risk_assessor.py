"""
Machine Learning Risk Assessor
Plug-and-play implementation for deep learning risk prediction.
"""
import os
from typing import List
from src.analyzers.risk_assessor_interface import RiskAssessorInterface, RiskResult
from src.models.story import Story


class MLRiskAssessor(RiskAssessorInterface):
    """
    Machine Learning Risk Assessor - Plug-and-play implementation.
    
    This class provides a standardized interface for integrating ML models
    into SprintGuard. It implements the RiskAssessorInterface, ensuring
    compatibility with the rest of the system.
    
    To implement:
        1. Train your model on the augmented NeoDataset
        2. Save the model to the specified model_path
        3. Implement the model loading logic in __init__
        4. Implement the prediction logic in assess()
    
    Example model types:
        - TF-IDF + Logistic Regression
        - BERT or other transformer models
        - Custom deep learning architectures
    """
    
    def __init__(self, historical_stories: List[Story], model_path: str = None):
        """
        Initialize ML Risk Assessor.
        
        Args:
            historical_stories: List of historical stories (for reference/context)
            model_path: Path to trained model file (e.g., 'models/risk_model.pkl')
        """
        super().__init__(historical_stories)
        self.model = None
        self.model_path = model_path
        
        # TODO: Load trained model when available
        # Example implementation:
        # if model_path and os.path.exists(model_path):
        #     import joblib
        #     self.model = joblib.load(model_path)
        #     print(f"✓ Loaded ML model from {model_path}")
        # else:
        #     print(f"⚠ Model not found at {model_path}")
        
        if model_path and os.path.exists(model_path):
            print(f"✓ Model path configured: {model_path}")
            # Model loading logic goes here
        else:
            print(f"⚠ ML model not yet trained. Using placeholder responses.")
            print(f"  Expected model path: {model_path}")
    
    def assess(self, description: str) -> RiskResult:
        """
        Assess risk using trained ML model.
        
        Args:
            description: User story description text
            
        Returns:
            RiskResult with risk level, confidence, and explanation
        
        TODO: Implement actual prediction logic
        
        Example implementation:
        ```python
        if self.model is None:
            return self._placeholder_response()
        
        # Extract features
        features = self.vectorizer.transform([description])
        
        # Get prediction
        risk_prob = self.model.predict_proba(features)[0]
        risk_label = self.model.predict(features)[0]
        
        # Map to risk levels
        if risk_label == 1:  # RISK
            risk_level = "High" if risk_prob[1] > 0.8 else "Medium"
            confidence = risk_prob[1] * 100
        else:  # SAFE
            risk_level = "Low"
            confidence = risk_prob[0] * 100
        
        return RiskResult(
            risk_level=risk_level,
            confidence=confidence,
            explanation=f"ML model prediction based on {len(self.historical_stories)} historical stories."
        )
        ```
        """
        if self.model is None:
            return self._placeholder_response()
        
        # TODO: Implement actual model prediction
        # This is where your deep learning model inference goes
        raise NotImplementedError(
            "ML model prediction logic to be implemented. "
            "Please train a model on the augmented NeoDataset and implement this method."
        )
    
    def _placeholder_response(self) -> RiskResult:
        """
        Placeholder response when model is not loaded.
        Returns a neutral assessment with clear indication that model is missing.
        """
        return RiskResult(
            risk_level="Medium",
            confidence=50.0,
            explanation=(
                "ML model not yet loaded. Please complete the following steps:\n"
                "1. Run augmentation: python scripts/augment_neodataset.py\n"
                "2. Train your ML model on the augmented dataset\n"
                "3. Save the model and update model_path in app.py\n"
                "4. Implement prediction logic in MLRiskAssessor.assess()"
            )
        )
    
    def get_name(self) -> str:
        """Return the name of this risk assessment algorithm"""
        if self.model is not None:
            return f"MLRiskAssessor (Model: {os.path.basename(self.model_path)})"
        return "MLRiskAssessor (No model loaded)"

