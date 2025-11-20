"""
Machine Learning Risk Assessor
DistilBERT-XGBoost implementation with SHAP explainability.
"""
import os
from typing import List
from src.analyzers.risk_assessor_interface import RiskAssessorInterface, RiskResult
from src.models.story import Story


class MLRiskAssessor(RiskAssessorInterface):
    """
    Hybrid DistilBERT-XGBoost Risk Assessor.
    
    Features:
    - Neuro-symbolic architecture (neural + symbolic features)
    - Real-time prediction with <1s latency
    - SHAP-based explainability
    - Similar story retrieval
    """
    
    def __init__(self, historical_stories: List[Story], model_dir: str = "models"):
        """
        Initialize ML Risk Assessor.
        
        Args:
            historical_stories: List of historical stories
            model_dir: Directory containing model artifacts
        """
        super().__init__(historical_stories)
        self.model_dir = model_dir
        self.predictor = None
        self.retriever = None
        
        # Check if model exists
        model_path = os.path.join(model_dir, 'xgboost_risk_model.json')
        
        if os.path.exists(model_path):
            print(f"✓ Model found at {model_dir}")
            
            # Load predictor and retriever
            try:
                from src.ml.risk_predictor import RiskPredictor
                from src.ml.similarity_retriever import SimilarityRetriever
                
                self.predictor = RiskPredictor.load(model_dir, quantize_bert=True)
                
                # Initialize similarity retriever (reuses predictor's embedder)
                if len(historical_stories) > 0:
                    self.retriever = SimilarityRetriever(
                        historical_stories,
                        embedder=self.predictor.bert_embedder
                    )
                else:
                    print("  ⚠ No historical stories for similarity retrieval")
                    self.retriever = None
                
                print("✓ ML Risk Assessor ready")
                
            except Exception as e:
                print(f"⚠ Error loading model: {e}")
                print("  Using placeholder responses.")
                self.predictor = None
                self.retriever = None
        else:
            print(f"⚠ ML model not found at {model_dir}")
            print("  Please train the model first:")
            print("  1. Run: python scripts/augment_neodataset.py")
            print("  2. Run: python src/ml/train_risk_model.py")
            self.predictor = None
            self.retriever = None
    
    def assess(self, description: str) -> RiskResult:
        """
        Assess risk using DistilBERT-XGBoost model.
        
        Args:
            description: User story description text
            
        Returns:
            RiskResult with risk level, confidence, explanation, and similar stories
        """
        if self.predictor is None:
            return self._placeholder_response()
        
        try:
            # Get prediction with explanation
            risk_level, confidence, explanation = self.predictor.predict(
                description,
                explain=True
            )
            
            # Find similar historical stories
            similar_story_ids = []
            if self.retriever is not None:
                similar = self.retriever.find_similar(description, k=5, min_similarity=0.3)
                similar_story_ids = [s['story_id'] for s in similar[:3]]  # Top 3
            
            return RiskResult(
                risk_level=risk_level,
                confidence=confidence,
                explanation=explanation,
                similar_stories=similar_story_ids
            )
            
        except Exception as e:
            # Graceful fallback on error
            print(f"⚠ Prediction error: {e}")
            return RiskResult(
                risk_level="Medium",
                confidence=50.0,
                explanation=f"Error during prediction: {str(e)}"
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
        if self.predictor is not None:
            return "DistilBERT-XGBoost Risk Assessor"
        return "MLRiskAssessor (No model loaded)"

