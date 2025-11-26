"""
ML Configuration - Centralized hyperparameters and constants

Adjust these values to tune model training and inference.
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BertConfig:
    """DistilBERT embedder configuration."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128  # Max tokens (128 sufficient for user stories)
    quantize: bool = True  # INT8 quantization for CPU speedup
    cache_size: int = 5000 # LRU cache for embeddings
    batch_size: int = 32   # Batch size for GPU memory management


@dataclass
class FeatureConfig:
    """Symbolic feature extraction configuration."""
    spacy_model: str = "en_core_web_sm"
    
    # Feature importance weights (symbolic vs embeddings)
    symbolic_weight: float = 2.0  # Higher weight for interpretable features
    embedding_weight: float = 1.0
    
    # Risk lexicons - keywords that indicate potential risk
    satd_keywords: List[str] = field(default_factory=lambda: [
        "hack", "fixme", "todo", "workaround", "temporary",
        "ugly", "hardcoded", "quick fix", "spaghetti"
    ])
    
    security_keywords: List[str] = field(default_factory=lambda: [
        "auth", "token", "jwt", "encrypt", "pii", "gdpr",
        "role", "permission", "injection", "xss", "secret",
        "oauth", "credential", "certificate"
    ])
    
    complexity_keywords: List[str] = field(default_factory=lambda: [
        "legacy", "mainframe", "wrapper", "migration", "api",
        "synchronization", "handshake", "middleware", "integration",
        "refactor", "database"
    ])
    
    weak_modals: List[str] = field(default_factory=lambda: [
        "might", "could", "should", "may", "ought"
    ])
    
    vague_quantifiers: List[str] = field(default_factory=lambda: [
        "fast", "easy", "robust", "user-friendly", "seamless",
        "efficient", "many", "few", "several", "tbd", "appropriate",
        "suitable", "good", "better", "nice"
    ])


@dataclass
class DataConfig:
    """Data loading and splitting configuration."""
    # Default data paths
    train_data_path: str = "data/neodataset_augmented_3class.csv"
    output_dir: str = "models"
    
    # Confidence filtering for training (set to 0 to include all samples)
    confidence_threshold: float = 0.5  # Min risk_confidence to include in training
    
    # 3-class mapping threshold (used by scripts/map_to_3class.py)
    # RISK samples with confidence > this → High, ≤ this → Medium
    # Lower value = more Medium samples, Higher value = more High samples  
    risk_class_threshold: float = 0.99  # Very high to get more Medium samples
    
    # Quick testing - set to None for full dataset
    # Note: needs ~3000+ samples for project-stratified splits to work
    max_samples: int = None  # Use full dataset (set to smaller number only if stratification allows)
    
    # Train/val/test split ratios
    test_size: float = 0.4  # 40% held out for val+test
    val_test_split: float = 0.5  # Split held-out 50/50 → 20% val, 20% test
    
    # Reproducibility
    random_state: int = 42


@dataclass
class XGBoostConfig:
    """XGBoost training hyperparameters."""
    # Core parameters
    objective: str = "multi:softprob"
    num_class: int = 3
    tree_method: str = "hist"  # Fast histogram-based algorithm
    
    # Tree structure
    max_depth: int = 5
    min_child_weight: int = 7
    max_bin: int = 64
    
    # Regularization
    colsample_bytree: float = 0.4
    colsample_bynode: float = 0.6
    subsample: float = 0.7
    reg_alpha: float = 0.5  # L1 regularization
    
    # Learning rate
    eta: float = 0.05  # Learning rate (lower = more robust)
    
    # Training
    num_boost_round: int = 10  # Max boosting rounds (set to 500 for production)
    early_stopping_rounds: int = 5  # Stop if no improvement
    
    # Evaluation
    eval_metric: str = "mlogloss"
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict:
        """Convert to XGBoost params dict."""
        return {
            "objective": self.objective,
            "num_class": self.num_class,
            "tree_method": self.tree_method,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "max_bin": self.max_bin,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bynode": self.colsample_bynode,
            "subsample": self.subsample,
            "reg_alpha": self.reg_alpha,
            "eta": self.eta,
            "eval_metric": self.eval_metric,
            "seed": self.seed,
        }


@dataclass
class CostMatrixConfig:
    """Cost-sensitive classification configuration.
    
    Cost matrix format: cost_matrix[true_class][predicted_class]
    Higher values = worse mistakes
    """
    # Default asymmetric cost matrix
    # Penalizes underestimating risk (predicting Low when actually High)
    cost_matrix: List[List[float]] = field(default_factory=lambda: [
        #  Pred: Low  Med  High
        [0.0, 1.0, 2.0],   # True: Low
        [1.5, 0.0, 1.0],   # True: Medium
        [3.0, 1.5, 0.0],   # True: High (worst to predict Low)
    ])


@dataclass
class InferenceConfig:
    """Runtime inference configuration."""
    model_dir: str = "models"
    quantize_bert: bool = True
    
    # Confidence thresholds for risk levels
    high_risk_threshold: float = 0.6
    medium_risk_threshold: float = 0.4
    
    # SHAP explanation settings
    max_shap_features: int = 10  # Top features to show in explanations


@dataclass 
class MLConfig:
    """Master configuration combining all ML settings."""
    bert: BertConfig = field(default_factory=BertConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    cost_matrix: CostMatrixConfig = field(default_factory=CostMatrixConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# Global default config instance
# Import and modify this in your scripts:
#   from src.ml.config import config
#   config.xgboost.num_boost_round = 1000
config = MLConfig()

