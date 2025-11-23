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
    
    def __init__(self, output_dir: str = "models", tracker=None):
        """
        Initialize trainer.
        
        Args:
            output_dir: Directory to save model artifacts
            tracker: Optional ExperimentTracker for MLflow logging
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
        
        # Experiment tracker
        self.tracker = tracker
    
    def load_data(self, csv_path: str, confidence_threshold: float = 0.75) -> pd.DataFrame:
        """
        Load and filter augmented NeoDataset.
        
        Args:
            csv_path: Path to augmented CSV
            confidence_threshold: Minimum risk_confidence to include
            
        Returns:
            Filtered DataFrame
        """
        print("\n" + "="*70)
        print("[STAGE 4] Loading Training Data")
        print("="*70)
        print(f"Data source: {csv_path}")
        print(f"Confidence threshold: >{confidence_threshold}")
        
        # Check file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"\n[ERROR] Training data not found: {csv_path}\n"
                f"Please run augmentation pipeline first:\n"
                f"  1. python scripts/augment_neodataset.py\n"
                f"  2. python scripts/map_to_3class.py"
            )
        
        print(f"\n[LOAD] Reading CSV file...")
        df = pd.read_csv(csv_path)
        original_count = len(df)
        print(f"  ✓ Loaded {original_count} stories")
        print(f"  ✓ Columns: {len(df.columns)}")
        
        # Validate required columns
        print(f"\n[VALIDATION] Checking required columns...")
        required_cols = ['title', 'description', 'risk_label', 'risk_confidence']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"[ERROR] Missing required columns: {missing_cols}")
        print(f"  ✓ All required columns present")
        
        # Filter by confidence
        print(f"\n[FILTER] Applying confidence threshold...")
        df = df[df['risk_confidence'] > confidence_threshold].copy()
        filtered_count = original_count - len(df)
        print(f"  Removed {filtered_count} low-confidence stories ({filtered_count/original_count*100:.1f}%)")
        print(f"  ✓ Retained {len(df)} high-confidence stories")
        
        if len(df) == 0:
            raise ValueError(f"[ERROR] No stories remain after confidence filtering!")
        
        # Combine title + description
        print(f"\n[TEXT] Creating full_text column...")
        df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        print(f"  ✓ Created full_text for {len(df)} stories")
        
        # Map risk labels to integers
        print(f"\n[LABELS] Mapping risk labels to integer classes...")
        label_map = {'Low': 0, 'Medium': 1, 'High': 2}
        print(f"  Mapping: {label_map}")
        
        # Check current label values
        unique_labels_before = df['risk_label'].unique()
        print(f"  Current risk_label values: {sorted(unique_labels_before)}")
        
        df['risk_class'] = df['risk_label'].map(label_map)
        
        # Validate mapping - check for NaN values
        nan_count = df['risk_class'].isna().sum()
        if nan_count > 0:
            invalid_labels = df[df['risk_class'].isna()]['risk_label'].unique()
            raise ValueError(
                f"\n[ERROR] Label mapping failed!\n"
                f"  {nan_count} stories have invalid risk_label values: {invalid_labels}\n"
                f"  Expected labels: {list(label_map.keys())}\n"
                f"  Did you run: python scripts/map_to_3class.py?"
            )
        print(f"  ✓ Label mapping successful (no NaN values)")
        
        # Validate risk_class values
        unique_classes = sorted(df['risk_class'].unique())
        expected_classes = [0, 1, 2]
        if unique_classes != expected_classes:
            print(f"  ⚠ Warning: Not all classes present")
            print(f"    Found: {unique_classes}")
            print(f"    Expected: {expected_classes}")
        else:
            print(f"  ✓ All 3 classes present: {unique_classes}")
        
        # Class distribution
        print(f"\n[CLASS DISTRIBUTION]")
        class_counts = df['risk_label'].value_counts()
        for label, risk_class in label_map.items():
            count = class_counts.get(label, 0)
            print(f"  {label} (class {risk_class}): {count} ({count/len(df)*100:.1f}%)")
        
        # Check for class imbalance
        min_count = class_counts.min()
        max_count = class_counts.max()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\n[IMBALANCE] Class imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 3:
            print(f"  ⚠ Warning: Significant class imbalance detected")
            print(f"    Using class weights for balanced training")
        else:
            print(f"  ✓ Class distribution is reasonably balanced")
        
        print(f"\n✓ Data loading complete: {len(df)} stories ready for training")
        
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
        import time
        
        n = len(texts)
        
        if show_progress:
            print("\n" + "="*70)
            print("[FEATURES] Extracting Hybrid Features")
            print("="*70)
            print(f"Processing {n} stories...")
        
        # Extract symbolic features
        if show_progress:
            print("\n[1/2] Extracting symbolic features...")
            print("  Features: readability, text stats, risk keywords, linguistic patterns")
            symbolic_start = time.time()
        
        symbolic_features = np.array([
            self.symbolic_extractor.extract_features(text)
            for text in texts
        ])
        
        if show_progress:
            symbolic_time = time.time() - symbolic_start
            print(f"  ✓ Shape: {symbolic_features.shape}")
            print(f"  ✓ Time: {symbolic_time:.1f}s ({symbolic_time/n*1000:.1f}ms per story)")
        
        # Extract BERT embeddings (batch processing for efficiency)
        if show_progress:
            print("\n[2/2] Extracting DistilBERT embeddings...")
            print("  Model: distilbert-base-uncased (quantized)")
            print("  This may take several minutes...")
            embed_start = time.time()
        
        embeddings = self.bert_embedder.embed_batch(texts, normalize=True)
        
        if show_progress:
            embed_time = time.time() - embed_start
            print(f"  ✓ Shape: {embeddings.shape}")
            print(f"  ✓ Time: {embed_time:.1f}s ({embed_time/n*1000:.1f}ms per story)")
            print(f"  ✓ Cache info: {self.bert_embedder.get_cache_info()}")
            
            total_time = symbolic_time + embed_time
            print(f"\n[TIMING] Feature extraction:")
            print(f"  Symbolic: {symbolic_time:.1f}s ({symbolic_time/total_time*100:.1f}%)")
            print(f"  BERT:     {embed_time:.1f}s ({embed_time/total_time*100:.1f}%)")
            print(f"  Total:    {total_time:.1f}s")
        
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
        print("\n[FUSION] Combining and scaling features...")
        
        # Z-score standardization for symbolic features
        if fit_scaler:
            print("  Fitting StandardScaler on symbolic features...")
            self.scaler = StandardScaler()
            symbolic_scaled = self.scaler.fit_transform(symbolic_features)
            print(f"    ✓ Scaler fitted (mean: {self.scaler.mean_[:3]}, std: {self.scaler.scale_[:3]}...)")
        else:
            print("  Applying pre-fitted scaler...")
            symbolic_scaled = self.scaler.transform(symbolic_features)
        
        # Concatenate: [Symbolic_15 | Embedding_768]
        print("  Concatenating symbolic + embedding features...")
        X_combined = np.hstack([symbolic_scaled, embeddings])
        print(f"    ✓ Combined shape: {X_combined.shape}")
        
        # Generate feature names (only once)
        if self.feature_names is None:
            print("  Generating feature names...")
            symbolic_names = self.symbolic_extractor.get_feature_names()
            embedding_names = [f'embedding_{i}' for i in range(768)]
            self.feature_names = symbolic_names + embedding_names
            print(f"    ✓ Total features: {len(self.feature_names)}")
            print(f"      - Symbolic: {len(symbolic_names)}")
            print(f"      - Embeddings: {len(embedding_names)}")
        
        print(f"  ✓ Feature preparation complete")
        
        return X_combined
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict = None,
        num_boost_round: int = 500
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
        print("\n" + "="*70)
        print("[TRAINING] XGBoost Model Training")
        print("="*70)
        
        print(f"\n[DATA] Training set:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique_train, counts_train):
            label_name = ['Low', 'Medium', 'High'][cls]
            print(f"    Class {cls} ({label_name}): {count} ({count/len(y_train)*100:.1f}%)")
        
        print(f"\n[DATA] Validation set:")
        print(f"  X_val shape: {X_val.shape}")
        print(f"  y_val shape: {y_val.shape}")
        unique_val, counts_val = np.unique(y_val, return_counts=True)
        for cls, count in zip(unique_val, counts_val):
            label_name = ['Low', 'Medium', 'High'][cls]
            print(f"    Class {cls} ({label_name}): {count} ({count/len(y_val)*100:.1f}%)")
        
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
        
        print(f"\n[HYPERPARAMETERS]")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Compute sample weights for class imbalance
        print(f"\n[WEIGHTS] Computing sample weights for class balance...")
        sample_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )
        print(f"  ✓ Sample weights computed")
        print(f"    Min weight: {sample_weights.min():.3f}")
        print(f"    Max weight: {sample_weights.max():.3f}")
        print(f"    Mean weight: {sample_weights.mean():.3f}")
        
        # Compute feature weights (2x for symbolic, 1x for embeddings)
        print(f"\n[WEIGHTS] Setting feature importance weights...")
        feature_weights = np.array([2.0] * 15 + [1.0] * 768)
        print(f"  Symbolic features: 2.0x weight (higher importance)")
        print(f"  Embedding features: 1.0x weight (baseline)")
        
        # Create DMatrix objects
        print(f"\n[DMATRIX] Creating XGBoost data matrices...")
        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            weight=sample_weights,
            feature_names=self.feature_names,
            feature_weights=feature_weights
        )
        print(f"  ✓ Training DMatrix: {dtrain.num_row()} rows × {dtrain.num_col()} features")
        
        dval = xgb.DMatrix(
            X_val,
            label=y_val,
            feature_names=self.feature_names
        )
        print(f"  ✓ Validation DMatrix: {dval.num_row()} rows × {dval.num_col()} features")
        
        # Train model
        print(f"\n[XGBOOST] Starting training...")
        print(f"  Max rounds: {num_boost_round}")
        print(f"  Early stopping: {min(50, num_boost_round//10)} rounds")
        print(f"  Progress logged every {min(50, num_boost_round//10)} rounds")
        print("-"*70)
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        early_stopping = min(50, num_boost_round // 10) if num_boost_round > 10 else None
        verbose_eval = min(50, max(1, num_boost_round // 10))
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=verbose_eval
        )
        
        print("-"*70)
        print(f"✓ Training complete!")
        if hasattr(self.model, 'best_iteration'):
            print(f"  Best iteration: {self.model.best_iteration}")
            print(f"  Best score: {self.model.best_score:.6f}")
        else:
            print(f"  Trained for {num_boost_round} rounds (no early stopping)")
        
        # Log to MLflow if tracker is available
        if self.tracker is not None:
            print("\n[TRACKER] Logging training metrics to MLflow...")
            self.tracker.log_params(params)
            self.tracker.log_stage_metrics("training", {
                "num_boost_round": num_boost_round,
                "train_size": len(y_train),
                "val_size": len(y_val),
                "num_features": X_train.shape[1]
            })
            print("  ✓ Training metrics logged")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Tuple of (accuracy, y_pred_proba)
        """
        from sklearn.metrics import (
            classification_report, confusion_matrix, accuracy_score,
            precision_recall_fscore_support
        )
        
        print("\n" + "="*70)
        print("[EVALUATION] Model Performance on Test Set")
        print("="*70)
        
        dtest = xgb.DMatrix(X_test, feature_names=self.feature_names)
        y_pred_proba = self.model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n  Overall Accuracy: {accuracy:.4f}")
        
        print("\n[CLASSIFICATION REPORT]")
        print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
        
        print("\n[CONFUSION MATRIX]")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1, 2], average=None
        )
        
        # Calculate macro-averaged F1
        macro_f1 = f1.mean()
        print(f"\n[MACRO-F1 SCORE]: {macro_f1:.4f}")
        
        # Log to MLflow if tracker is available
        if self.tracker is not None:
            print("\n[TRACKER] Logging evaluation metrics to MLflow...")
            self.tracker.log_stage_metrics("evaluation", {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "f1_low": f1[0],
                "f1_medium": f1[1],
                "f1_high": f1[2],
                "precision_low": precision[0],
                "precision_medium": precision[1],
                "precision_high": precision[2],
                "recall_low": recall[0],
                "recall_medium": recall[1],
                "recall_high": recall[2]
            })
            
            # Generate and log calibration plot
            from src.visualization.publication_plots import generate_calibration_plot
            print("  Generating calibration plot...")
            generate_calibration_plot(
                y_test, y_pred_proba,
                class_names=['Low', 'Medium', 'High'],
                output_path="visualizations/calibration_plot.pdf"
            )
            self.tracker.log_artifact("visualizations/calibration_plot.pdf", "visualizations")
            self.tracker.log_artifact("visualizations/calibration_plot.svg", "visualizations")
            
            print("  ✓ Evaluation metrics logged")
        
        print("\n✓ Evaluation complete")
        
        return accuracy, y_pred_proba
    
    def visualize_embeddings(self, embeddings: np.ndarray, labels: np.ndarray, sample_size: int = 2000):
        """
        Generate t-SNE visualization of DistilBERT embeddings.
        
        Args:
            embeddings: Embedding vectors (n_samples × 768)
            labels: Class labels (n_samples)
            sample_size: Number of samples to visualize
        """
        if self.tracker is None:
            print("[SKIP] No tracker available, skipping embedding visualization")
            return
        
        print("\n[VISUALIZATION] Generating t-SNE embedding plot...")
        
        from src.visualization.publication_plots import generate_tsne_embeddings
        
        generate_tsne_embeddings(
            embeddings=embeddings,
            labels=labels,
            class_names=['Low', 'Medium', 'High'],
            output_path="visualizations/embeddings_tsne.pdf",
            sample_size=sample_size
        )
        
        self.tracker.log_artifact("visualizations/embeddings_tsne.pdf", "visualizations")
        self.tracker.log_artifact("visualizations/embeddings_tsne.svg", "visualizations")
        
        print("  ✓ t-SNE visualization saved and logged")
    
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
    
    print("\n" + "="*70)
    print("SPRINTGUARD ML MODEL TRAINING")
    print("="*70)
    print("Architecture: Hybrid DistilBERT-XGBoost")
    print("="*70)
    
    # Initialize trainer
    print("\n[INIT] Initializing trainer...")
    trainer = RiskModelTrainer(output_dir=args.output)
    
    # Load data
    df = trainer.load_data(args.data, confidence_threshold=args.confidence)
    
    # Stratified split by project
    print("\n" + "="*70)
    print("[SPLIT] Creating Train/Val/Test Splits")
    print("="*70)
    print("Strategy: Stratified by project (prevents data leakage)")
    print("Split ratio: 60% train, 20% val, 20% test")
    
    print("\n[SPLIT 1/2] Creating train and temp sets...")
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df['idproject'],
        random_state=42
    )
    print(f"  Train: {len(train_df)} stories")
    print(f"  Temp (val+test): {len(temp_df)} stories")
    
    print("\n[SPLIT 2/2] Splitting temp into val and test...")
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['idproject'],
        random_state=42
    )
    print(f"  Val:  {len(val_df)} stories")
    print(f"  Test: {len(test_df)} stories")
    
    print(f"\n[SUMMARY] Final split sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Total: {len(df)}")
    
    # Verify stratification worked
    print(f"\n[VALIDATION] Verifying stratification...")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        class_dist = split_df['risk_label'].value_counts(normalize=True) * 100
        print(f"  {split_name} class distribution:")
        for label in ['Low', 'Medium', 'High']:
            pct = class_dist.get(label, 0)
            print(f"    {label}: {pct:.1f}%")
    
    print(f"  ✓ Stratification successful")
    
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

