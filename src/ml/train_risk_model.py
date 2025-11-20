"""
Training Pipeline for Hybrid DistilBERT-XGBoost Risk Model
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from typing import Tuple, Dict

from src.ml.feature_extractors import SymbolicFeatureExtractor
from src.ml.bert_embedder import BertEmbedder


class RiskModelTrainer:
    """
    Trains the hybrid DistilBERT-XGBoost risk assessment model.
    
    Pipeline:
    1. Load augmented NeoDataset
    2. Extract symbolic + neural features
    3. Train XGBoost with research-backed hyperparameters
    4. Save model artifacts
    """
    
    def __init__(self, output_dir: str = "models"):
        """
        Initialize trainer.
        
        Args:
            output_dir: Directory to save model artifacts
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize feature extractors
        print("Initializing feature extractors...")
        self.symbolic_extractor = SymbolicFeatureExtractor()
        self.bert_embedder = BertEmbedder(quantize=True, cache_size=5000)
        
        # Will be set during training
        self.scaler = None
        self.model = None
        self.feature_names = None
    
    def load_data(self, csv_path: str, confidence_threshold: float = 0.75) -> pd.DataFrame:
        """
        Load and filter augmented NeoDataset.
        
        Args:
            csv_path: Path to augmented CSV
            confidence_threshold: Minimum risk_confidence to include
            
        Returns:
            Filtered DataFrame
        """
        print(f"\nLoading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"  Total stories: {len(df)}")
        
        # Filter by confidence
        df = df[df['risk_confidence'] > confidence_threshold].copy()
        print(f"  High-confidence stories (>{confidence_threshold}): {len(df)}")
        
        # Combine title + description
        df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        
        # Map risk labels to integers
        label_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df['risk_class'] = df['risk_label'].map(label_map)
        
        print(f"\nClass distribution:")
        print(df['risk_label'].value_counts())
        
        return df
    
    def extract_features(self, texts: list, show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract hybrid features (symbolic + embeddings).
        
        Args:
            texts: List of user story texts
            show_progress: Print progress updates
            
        Returns:
            Tuple of (symbolic_features, embeddings)
        """
        n = len(texts)
        
        if show_progress:
            print(f"\nExtracting features for {n} stories...")
        
        # Extract symbolic features
        if show_progress:
            print("  [1/2] Extracting symbolic features...")
        symbolic_features = np.array([
            self.symbolic_extractor.extract_features(text)
            for text in texts
        ])
        
        # Extract BERT embeddings (batch processing for efficiency)
        if show_progress:
            print("  [2/2] Extracting DistilBERT embeddings...")
        embeddings = self.bert_embedder.embed_batch(texts, normalize=True)
        
        if show_progress:
            print(f"  ✓ Symbolic: {symbolic_features.shape}")
            print(f"  ✓ Embeddings: {embeddings.shape}")
            print(f"  ✓ Cache: {self.bert_embedder.get_cache_info()}")
        
        return symbolic_features, embeddings
    
    def prepare_features(
        self,
        symbolic_features: np.ndarray,
        embeddings: np.ndarray,
        fit_scaler: bool = True
    ) -> np.ndarray:
        """
        Fuse and scale features.
        
        Args:
            symbolic_features: (N, 15) symbolic features
            embeddings: (N, 768) DistilBERT embeddings
            fit_scaler: Whether to fit scaler (True for train, False for val/test)
            
        Returns:
            (N, 783) fused feature matrix
        """
        # Z-score standardization for symbolic features
        if fit_scaler:
            self.scaler = StandardScaler()
            symbolic_scaled = self.scaler.fit_transform(symbolic_features)
        else:
            symbolic_scaled = self.scaler.transform(symbolic_features)
        
        # Concatenate: [Symbolic_15 | Embedding_768]
        X_combined = np.hstack([symbolic_scaled, embeddings])
        
        # Generate feature names
        symbolic_names = self.symbolic_extractor.get_feature_names()
        embedding_names = [f'embedding_{i}' for i in range(768)]
        self.feature_names = symbolic_names + embedding_names
        
        return X_combined
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict = None
    ):
        """
        Train XGBoost model with research-backed hyperparameters.
        
        Args:
            X_train: Training features (N, 783)
            y_train: Training labels (N,)
            X_val: Validation features (M, 783)
            y_val: Validation labels (M,)
            params: Optional XGBoost parameters (uses defaults if None)
        """
        print("\nTraining XGBoost model...")
        
        # Default parameters (research-backed)
        if params is None:
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'tree_method': 'hist',
                'max_bin': 64,
                'max_depth': 5,
                'min_child_weight': 7,
                'colsample_bytree': 0.4,
                'colsample_bynode': 0.6,
                'subsample': 0.7,
                'reg_alpha': 0.5,
                'eta': 0.05,
                'eval_metric': 'mlogloss',
                'seed': 42
            }
        
        print(f"  Hyperparameters: {params}")
        
        # Compute sample weights for class imbalance
        sample_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )
        
        # Compute feature weights (2x for symbolic, 1x for embeddings)
        feature_weights = np.array([2.0] * 15 + [1.0] * 768)
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            weight=sample_weights,
            feature_names=self.feature_names,
            feature_weights=feature_weights
        )
        
        dval = xgb.DMatrix(
            X_val,
            label=y_val,
            feature_names=self.feature_names
        )
        
        # Train model
        evals = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50
        )
        
        print(f"✓ Training complete (best iteration: {self.model.best_iteration})")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        print("\nEvaluating on test set...")
        
        dtest = xgb.DMatrix(X_test, feature_names=self.feature_names)
        y_pred_proba = self.model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy, y_pred_proba
    
    def save_model(self):
        """Save all model artifacts."""
        print(f"\nSaving model artifacts to {self.output_dir}/...")
        
        # XGBoost model
        model_path = os.path.join(self.output_dir, 'xgboost_risk_model.json')
        self.model.save_model(model_path)
        print(f"  ✓ Saved: {model_path}")
        
        # Feature scaler
        scaler_path = os.path.join(self.output_dir, 'feature_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✓ Saved: {scaler_path}")
        
        # Feature names
        feature_names_path = os.path.join(self.output_dir, 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"  ✓ Saved: {feature_names_path}")
        
        # Risk lexicons
        lexicons_path = os.path.join(self.output_dir, 'risk_lexicons.json')
        lexicons = {
            'satd_keywords': SymbolicFeatureExtractor.SATD_KEYWORDS,
            'security_keywords': SymbolicFeatureExtractor.SECURITY_KEYWORDS,
            'complexity_keywords': SymbolicFeatureExtractor.COMPLEXITY_KEYWORDS
        }
        with open(lexicons_path, 'w') as f:
            json.dump(lexicons, f, indent=2)
        print(f"  ✓ Saved: {lexicons_path}")
        
        print("✓ All artifacts saved")


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DistilBERT-XGBoost Risk Model')
    parser.add_argument(
        '--data',
        type=str,
        default='data/neodataset_augmented_high_confidence.csv',
        help='Path to augmented dataset CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Output directory for model artifacts'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.75,
        help='Minimum risk_confidence threshold'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RiskModelTrainer(output_dir=args.output)
    
    # Load data
    df = trainer.load_data(args.data, confidence_threshold=args.confidence)
    
    # Stratified split by project
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df['idproject'],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['idproject'],
        random_state=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Extract features
    X_train_sym, X_train_emb = trainer.extract_features(train_df['full_text'].tolist())
    X_val_sym, X_val_emb = trainer.extract_features(val_df['full_text'].tolist())
    X_test_sym, X_test_emb = trainer.extract_features(test_df['full_text'].tolist())
    
    # Fuse features
    X_train = trainer.prepare_features(X_train_sym, X_train_emb, fit_scaler=True)
    X_val = trainer.prepare_features(X_val_sym, X_val_emb, fit_scaler=False)
    X_test = trainer.prepare_features(X_test_sym, X_test_emb, fit_scaler=False)
    
    # Get labels
    y_train = train_df['risk_class'].values
    y_val = val_df['risk_class'].values
    y_test = test_df['risk_class'].values
    
    # Train
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    trainer.evaluate(X_test, y_test)
    
    # Save
    trainer.save_model()
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)


if __name__ == '__main__':
    main()

