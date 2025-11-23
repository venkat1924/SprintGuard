"""
Risk Prediction with Explainability
Combines all components for real-time risk assessment.
"""
import os
import json
import joblib
import numpy as np
import xgboost as xgb
import shap
from typing import Tuple, Dict, Optional

from src.ml.feature_extractors import SymbolicFeatureExtractor
from src.ml.bert_embedder import BertEmbedder
from src.ml.calibration import MulticlassCalibrator
from src.ml.threshold_optimizer import CostSensitiveClassifier


class RiskPredictor:
    """
    End-to-end risk prediction with SHAP explanations.
    
    Pipeline:
    1. Extract symbolic + neural features
    2. XGBoost prediction → calibrated probabilities
    3. Cost-sensitive decision rule → risk level
    4. TreeSHAP → explanation
    """
    
    RISK_LABELS = ['Low', 'Medium', 'High']
    
    def __init__(
        self,
        model_dir: str = "models",
        quantize_bert: bool = True
    ):
        """
        Initialize risk predictor.
        
        Args:
            model_dir: Directory containing model artifacts
            quantize_bert: Use quantized DistilBERT
        """
        self.model_dir = model_dir
        
        print(f"\n[RISK PREDICTOR] Loading model from {model_dir}...")
        print(f"  Quantized BERT: {quantize_bert}")
        
        # Load feature extractors
        print(f"\n[EXTRACTORS] Initializing feature extractors...")
        self.symbolic_extractor = SymbolicFeatureExtractor()
        print(f"  ✓ Symbolic extractor ready ({len(self.symbolic_extractor.get_feature_names())} features)")
        
        self.bert_embedder = BertEmbedder(quantize=quantize_bert, cache_size=1000)
        print(f"  ✓ BERT embedder ready (cache size: 1000)")
        
        # Load scaler
        print(f"\n[SCALER] Loading feature scaler...")
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print(f"  ✓ Loaded scaler from {scaler_path}")
        
        # Load feature names
        print(f"\n[FEATURES] Loading feature names...")
        feature_names_path = os.path.join(model_dir, 'feature_names.json')
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names not found: {feature_names_path}")
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
        print(f"  ✓ Loaded {len(self.feature_names)} feature names")
        
        # Load XGBoost model
        print(f"\n[MODEL] Loading XGBoost model...")
        model_path = os.path.join(model_dir, 'xgboost_risk_model.json')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        print(f"  ✓ Loaded XGBoost model from {model_path}")
        print(f"  ✓ Model has {self.model.num_features()} features")
        
        # Load calibrator (if available)
        calibrator_path = os.path.join(model_dir, 'calibrator.pkl')
        if os.path.exists(calibrator_path):
            self.calibrator = MulticlassCalibrator.load(calibrator_path)
            print(f"  ✓ Loaded probability calibrator")
        else:
            self.calibrator = None
            print(f"  ⚠ No calibrator found (using raw probabilities)")
        
        # Load cost matrix (if available)
        cost_matrix_path = os.path.join(model_dir, 'cost_matrix.json')
        if os.path.exists(cost_matrix_path):
            self.cost_classifier = CostSensitiveClassifier.load(cost_matrix_path)
            print(f"  ✓ Loaded cost-sensitive classifier")
        else:
            self.cost_classifier = CostSensitiveClassifier()
            print(f"  ⚠ Using default cost matrix")
        
        # Initialize SHAP explainer
        print("  Initializing TreeSHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print(f"  ✓ SHAP ready")
        
        # Explanation cache (for repeated queries)
        self._explanation_cache = {}
        
        print("✓ RiskPredictor ready")
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract hybrid features from text.
        
        Args:
            text: User story description
            
        Returns:
            (783,) feature vector
        """
        # Symbolic features
        symbolic = self.symbolic_extractor.extract_features(text)
        
        # BERT embedding
        embedding = self.bert_embedder.embed(text, normalize=True)
        
        # Scale and concatenate
        symbolic_scaled = self.scaler.transform(symbolic.reshape(1, -1))
        features = np.hstack([symbolic_scaled, embedding.reshape(1, -1)])
        
        return features
    
    def predict(
        self,
        text: str,
        explain: bool = True
    ) -> Tuple[str, float, Optional[str]]:
        """
        Predict risk level for a user story.
        
        Args:
            text: User story description
            explain: Whether to generate explanation
            
        Returns:
            Tuple of (risk_level, confidence, explanation)
        """
        import time
        
        # Check cache for explanation
        text_hash = hash(text)
        if text_hash in self._explanation_cache:
            print(f"[PREDICT] Cache hit for text hash {text_hash}")
            cached = self._explanation_cache[text_hash]
            return cached['risk_level'], cached['confidence'], cached['explanation']
        
        print(f"[PREDICT] Processing new prediction...")
        print(f"  Text length: {len(text)} characters")
        
        # Extract features
        feature_start = time.time()
        X = self.extract_features(text)
        feature_time = (time.time() - feature_start) * 1000
        print(f"  ✓ Feature extraction: {feature_time:.1f}ms")
        
        # XGBoost prediction
        predict_start = time.time()
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        raw_proba = self.model.predict(dmatrix)[0]  # (3,)
        predict_time = (time.time() - predict_start) * 1000
        print(f"  ✓ Model inference: {predict_time:.1f}ms")
        
        # Apply calibration
        if self.calibrator is not None:
            proba = self.calibrator.predict_proba(raw_proba.reshape(1, -1))[0]
            print(f"  ✓ Applied probability calibration")
        else:
            proba = raw_proba
        
        # Cost-sensitive decision
        risk_class = self.cost_classifier.predict(proba.reshape(1, -1))[0]
        risk_level = self.RISK_LABELS[risk_class]
        
        # Confidence (probability of predicted class)
        confidence = proba[risk_class] * 100
        
        print(f"  ✓ Prediction: {risk_level} ({confidence:.1f}% confidence)")
        print(f"    Probabilities: Low={proba[0]*100:.1f}%, Medium={proba[1]*100:.1f}%, High={proba[2]*100:.1f}%")
        
        # Generate explanation
        explanation = None
        if explain:
            explain_start = time.time()
            explanation = self._generate_explanation(X, proba, risk_level)
            explain_time = (time.time() - explain_start) * 1000
            print(f"  ✓ Generated explanation: {explain_time:.1f}ms")
        
        # Cache result
        self._explanation_cache[text_hash] = {
            'risk_level': risk_level,
            'confidence': confidence,
            'explanation': explanation
        }
        
        total_time = feature_time + predict_time + (explain_time if explain else 0)
        print(f"  ✓ Total prediction time: {total_time:.1f}ms")
        
        return risk_level, confidence, explanation
    
    def _generate_explanation(
        self,
        X: np.ndarray,
        proba: np.ndarray,
        risk_level: str
    ) -> str:
        """
        Generate human-readable explanation using SHAP.
        
        Args:
            X: Feature vector (1, 783)
            proba: Class probabilities (3,)
            risk_level: Predicted risk level
            
        Returns:
            Natural language explanation
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)  # List of (1, 783) arrays per class
        
        # Get SHAP values for predicted class
        risk_class = self.RISK_LABELS.index(risk_level)
        shap_vector = shap_values[risk_class][0]  # (783,)
        
        # Get top contributing features
        abs_shap = np.abs(shap_vector)
        top_indices = np.argsort(abs_shap)[-5:][::-1]  # Top 5
        
        # Build explanation
        explanation_parts = [f"This story is classified as **{risk_level} Risk**."]
        explanation_parts.append("\n**Key Risk Factors:**")
        
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            feature_value = X[0, idx]
            shap_value = shap_vector[idx]
            
            # Only include if SHAP contribution is significant
            if abs(shap_value) < 0.01:
                continue
            
            # Map to human-readable description
            description = self._feature_to_text(feature_name, feature_value, shap_value)
            if description:
                direction = "↑" if shap_value > 0 else "↓"
                explanation_parts.append(f"- {direction} {description}")
        
        # Add probability distribution
        explanation_parts.append(f"\n**Probability Distribution:**")
        for i, label in enumerate(self.RISK_LABELS):
            explanation_parts.append(f"- {label}: {proba[i]*100:.1f}%")
        
        return "\n".join(explanation_parts)
    
    def _feature_to_text(self, feature_name: str, value: float, shap_value: float) -> Optional[str]:
        """
        Convert feature to human-readable text.
        
        Args:
            feature_name: Name of feature
            value: Feature value
            shap_value: SHAP contribution
            
        Returns:
            Human-readable description or None
        """
        # Symbolic features (interpretable)
        if feature_name == 'flesch_reading_ease':
            if value < 30:
                return "Description is very difficult to read (complexity risk)"
            elif value < 50:
                return "Description is moderately complex"
        
        elif feature_name == 'gunning_fog':
            if value > 16:
                return "Description requires graduate-level reading (high complexity)"
            elif value > 12:
                return "Description is somewhat complex"
        
        elif feature_name == 'weak_modal_density':
            if value > 0.15:
                count = int(value * 10)  # Approximate count
                return f"High ambiguity detected ({count}+ uncertain words: might, could, should)"
        
        elif feature_name == 'has_vague_quantifiers':
            if value > 0.5:
                return "Contains vague terms (fast, easy, user-friendly) - unclear requirements"
        
        elif feature_name == 'passive_voice_ratio':
            if value > 0.20:
                return "High passive voice usage - unclear actor/responsibility"
        
        elif feature_name == 'satd_count':
            if value > 0:
                return f"Contains technical debt markers (TODO, hack, fixme: {int(value)} occurrences)"
        
        elif feature_name == 'security_count':
            if value > 0:
                return f"Security-related task (auth, encryption, etc.: {int(value)} keywords)"
        
        elif feature_name == 'complexity_count':
            if value > 0:
                return f"Integration/legacy complexity detected ({int(value)} keywords)"
        
        elif feature_name == 'word_count':
            if value > 200:
                return "Very lengthy description (information overload risk)"
            elif value < 20:
                return "Very short description (insufficient detail risk)"
        
        elif feature_name.startswith('embedding_'):
            # Neural features (less interpretable)
            return "Semantic complexity detected in text content"
        
        return None
    
    def clear_cache(self):
        """Clear explanation cache."""
        self._explanation_cache.clear()
        self.bert_embedder.clear_cache()
        print("✓ Cache cleared")
    
    @staticmethod
    def load(model_dir: str = "models", quantize_bert: bool = True) -> 'RiskPredictor':
        """
        Load a trained risk predictor.
        
        Args:
            model_dir: Directory containing model artifacts
            quantize_bert: Use quantized BERT
            
        Returns:
            RiskPredictor instance
        """
        return RiskPredictor(model_dir=model_dir, quantize_bert=quantize_bert)

