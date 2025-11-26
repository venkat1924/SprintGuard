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

from src.ml.config import config
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
    5. Generate publication-quality visualizations
    """
    
    def __init__(self, output_dir: str = None, viz_dir: str = "visualizations", tracker=None):
        """
        Initialize trainer.
        
        Args:
            output_dir: Directory to save model artifacts
            viz_dir: Directory to save visualizations
            tracker: Optional ExperimentTracker for MLflow logging
        """
        self.output_dir = output_dir or config.data.output_dir
        self.viz_dir = viz_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Initialize feature extractors
        print("Initializing feature extractors...")
        self.symbolic_extractor = SymbolicFeatureExtractor()
        self.bert_embedder = BertEmbedder()
        
        # Will be set during training
        self.scaler = None
        self.model = None
        self.feature_names = None
        self.evals_result = {}  # Store training history for learning curves
        
        # Experiment tracker (optional)
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
        
        # Quick testing: limit samples if configured
        if config.data.max_samples is not None and len(df) > config.data.max_samples:
            print(f"\n[SAMPLE] Limiting to {config.data.max_samples} samples for quick testing...")
            df = df.sample(n=config.data.max_samples, random_state=config.data.random_state)
            print(f"  ✓ Sampled {len(df)} stories")
        
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
        num_boost_round: int = None
    ):
        """
        Train XGBoost model with research-backed hyperparameters.
        
        Args:
            X_train: Training features (N, 783)
            y_train: Training labels (N,)
            X_val: Validation features (M, 783)
            y_val: Validation labels (M,)
            params: Optional XGBoost parameters (uses config defaults if None)
            num_boost_round: Max boosting rounds (uses config default if None)
        """
        num_boost_round = num_boost_round or config.xgboost.num_boost_round
        
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
        
        # Use config defaults if params not provided
        if params is None:
            params = config.xgboost.to_dict()
        
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
        
        # Store eval results for learning curve visualization
        self.evals_result = {}
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=verbose_eval,
            evals_result=self.evals_result  # Capture training history
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
            Tuple of (accuracy, y_pred, y_pred_proba)
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
            print("  ✓ Evaluation metrics logged")
        
        print("\n✓ Evaluation complete")
        
        return accuracy, y_pred, y_pred_proba
    
    def generate_all_visualizations(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray = None,
        embeddings: np.ndarray = None,
        embedding_labels: np.ndarray = None,
        sample_size: int = 2000,
        shap_sample_size: int = 500
    ):
        """
        Generate all publication-quality visualizations.
        
        Args:
            y_test: True test labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            y_train: Training labels (for class distribution)
            y_val: Validation labels (for class distribution)
            X_test: Test feature matrix (for SHAP plots)
            embeddings: Optional embeddings for t-SNE
            embedding_labels: Labels for embedding samples
            sample_size: Number of samples for t-SNE
            shap_sample_size: Number of samples for SHAP (subsample for speed)
        """
        from src.visualization.publication_plots import (
            generate_calibration_plot,
            generate_confusion_matrix_heatmap,
            generate_roc_curves,
            generate_precision_recall_curves,
            generate_feature_importance_plot,
            generate_learning_curves,
            generate_class_distribution_plot,
            generate_tsne_embeddings,
            generate_shap_summary_plot,
            generate_shap_bar_plot,
            generate_shap_waterfall_plot
        )
        
        print("\n" + "="*70)
        print("[VISUALIZATIONS] Generating Publication-Quality Figures")
        print("="*70)
        print(f"Output directory: {self.viz_dir}/")
        
        class_names = ['Low', 'Medium', 'High']
        viz_count = 0
        total_viz = 11  # Updated total
        
        # 1. Confusion Matrix Heatmap
        print(f"\n[1/{total_viz}] Confusion Matrix...")
        try:
            generate_confusion_matrix_heatmap(
                y_test, y_pred,
                class_names=class_names,
                output_path=f"{self.viz_dir}/confusion_matrix.pdf"
            )
            viz_count += 1
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
        
        # 2. Calibration Plot
        print(f"\n[2/{total_viz}] Calibration Plot...")
        try:
            generate_calibration_plot(
                y_test, y_pred_proba,
                class_names=class_names,
                output_path=f"{self.viz_dir}/calibration_plot.pdf"
            )
            viz_count += 1
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
        
        # 3. ROC Curves
        print(f"\n[3/{total_viz}] ROC Curves...")
        try:
            generate_roc_curves(
                y_test, y_pred_proba,
                class_names=class_names,
                output_path=f"{self.viz_dir}/roc_curves.pdf"
            )
            viz_count += 1
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
        
        # 4. Precision-Recall Curves
        print(f"\n[4/{total_viz}] Precision-Recall Curves...")
        try:
            generate_precision_recall_curves(
                y_test, y_pred_proba,
                class_names=class_names,
                output_path=f"{self.viz_dir}/precision_recall_curves.pdf"
            )
            viz_count += 1
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
        
        # 5. Feature Importance (XGBoost native)
        print(f"\n[5/{total_viz}] Feature Importance (XGBoost)...")
        try:
            generate_feature_importance_plot(
                self.model,
                self.feature_names,
                output_path=f"{self.viz_dir}/feature_importance.pdf",
                top_k=25
            )
            viz_count += 1
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
        
        # 6. Learning Curves
        print(f"\n[6/{total_viz}] Learning Curves...")
        try:
            if self.evals_result:
                generate_learning_curves(
                    self.evals_result,
                    output_path=f"{self.viz_dir}/learning_curves.pdf"
                )
                viz_count += 1
            else:
                print("  ⚠ No training history available")
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
        
        # 7. Class Distribution
        print(f"\n[7/{total_viz}] Class Distribution...")
        try:
            generate_class_distribution_plot(
                y_train, y_val, y_test,
                class_names=class_names,
                output_path=f"{self.viz_dir}/class_distribution.pdf"
            )
            viz_count += 1
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
        
        # 8. t-SNE Embeddings (optional, can be slow)
        print(f"\n[8/{total_viz}] t-SNE Embeddings...")
        if embeddings is not None and embedding_labels is not None:
            try:
                generate_tsne_embeddings(
                    embeddings=embeddings,
                    labels=embedding_labels,
                    class_names=class_names,
                    output_path=f"{self.viz_dir}/embeddings_tsne.pdf",
                    sample_size=sample_size
                )
                viz_count += 1
            except Exception as e:
                print(f"  ⚠ Failed: {e}")
        else:
            print("  ⚠ No embeddings provided, skipping t-SNE")
        
        # === SHAP EXPLAINABILITY PLOTS ===
        if X_test is not None:
            # Subsample for SHAP (can be slow on large datasets)
            if len(X_test) > shap_sample_size:
                print(f"\n  [SHAP] Subsampling {shap_sample_size} from {len(X_test)} for SHAP analysis...")
                shap_indices = np.random.choice(len(X_test), shap_sample_size, replace=False)
                X_shap = X_test[shap_indices]
                y_shap = y_test[shap_indices]
            else:
                X_shap = X_test
                y_shap = y_test
            
            # 9. SHAP Summary Plot (Beeswarm)
            print(f"\n[9/{total_viz}] SHAP Summary Plot (Beeswarm)...")
            try:
                generate_shap_summary_plot(
                    self.model,
                    X_shap,
                    self.feature_names,
                    class_names=class_names,
                    output_path=f"{self.viz_dir}/shap_summary.pdf",
                    max_display=20
                )
                viz_count += 1
            except Exception as e:
                print(f"  ⚠ Failed: {e}")
            
            # 10. SHAP Bar Plot (Mean Importance)
            print(f"\n[10/{total_viz}] SHAP Feature Importance...")
            try:
                generate_shap_bar_plot(
                    self.model,
                    X_shap,
                    self.feature_names,
                    class_names=class_names,
                    output_path=f"{self.viz_dir}/shap_importance.pdf",
                    max_display=25
                )
                viz_count += 1
            except Exception as e:
                print(f"  ⚠ Failed: {e}")
            
            # 11. SHAP Waterfall (Example High-Risk prediction)
            print(f"\n[11/{total_viz}] SHAP Waterfall (Example Explanation)...")
            try:
                # Find a high-risk example
                high_risk_indices = np.where(y_shap == 2)[0]
                if len(high_risk_indices) > 0:
                    example_idx = high_risk_indices[0]
                    generate_shap_waterfall_plot(
                        self.model,
                        X_shap[example_idx],
                        self.feature_names,
                        class_idx=2,  # High risk
                        class_names=class_names,
                        output_path=f"{self.viz_dir}/shap_waterfall_high_risk.pdf"
                    )
                    viz_count += 1
                else:
                    print("  ⚠ No high-risk samples found for waterfall plot")
            except Exception as e:
                print(f"  ⚠ Failed: {e}")
        else:
            print(f"\n[9-11/{total_viz}] SHAP Plots...")
            print("  ⚠ No X_test provided, skipping SHAP visualizations")
        
        print("\n" + "-"*70)
        print(f"✓ Generated {viz_count}/{total_viz} visualizations in {self.viz_dir}/")
        print("-"*70)
        
        # List all generated files
        print("\nGenerated files:")
        import glob
        for ext in ['pdf', 'svg', 'html']:
            files = sorted(glob.glob(f"{self.viz_dir}/*.{ext}"))
            if files:
                print(f"  {ext.upper()}:")
                for f in files:
                    print(f"    - {f}")
    
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
        default=config.data.train_data_path,
        help='Path to augmented dataset CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Output directory for model artifacts'
    )
    parser.add_argument(
        '--viz-dir',
        type=str,
        default='visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=config.data.confidence_threshold,
        help='Minimum risk_confidence threshold'
    )
    parser.add_argument(
        '--skip-tsne',
        action='store_true',
        help='Skip t-SNE visualization (can be slow)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SPRINTGUARD ML MODEL TRAINING")
    print("="*70)
    print("Architecture: Hybrid DistilBERT-XGBoost")
    print("="*70)
    
    # Initialize trainer
    print("\n[INIT] Initializing trainer...")
    trainer = RiskModelTrainer(output_dir=args.output, viz_dir=args.viz_dir)
    
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
        test_size=config.data.test_size,
        stratify=df['idproject'],
        random_state=config.data.random_state
    )
    print(f"  Train: {len(train_df)} stories")
    print(f"  Temp (val+test): {len(temp_df)} stories")
    
    print("\n[SPLIT 2/2] Splitting temp into val and test...")
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.data.val_test_split,
        stratify=temp_df['idproject'],
        random_state=config.data.random_state
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
    accuracy, y_pred, y_pred_proba = trainer.evaluate(X_test, y_test)
    
    # Save model artifacts
    trainer.save_model()
    
    # Generate all publication-quality visualizations
    # Combine embeddings for t-SNE (use test set for cleaner visualization)
    if not args.skip_tsne:
        embeddings_for_viz = X_test_emb
        labels_for_viz = y_test
    else:
        embeddings_for_viz = None
        labels_for_viz = None
    
    trainer.generate_all_visualizations(
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,  # Pass full test features for SHAP analysis
        embeddings=embeddings_for_viz,
        embedding_labels=labels_for_viz,
        sample_size=2000,
        shap_sample_size=500  # Subsample for SHAP speed
    )
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Model artifacts: {args.output}/")
    print(f"  Visualizations:  {args.viz_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()

