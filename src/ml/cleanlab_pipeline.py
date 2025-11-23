"""
Cleanlab Noise Remediation Pipeline
Detects and filters label errors from Snorkel output
Source: Augmenting_NeoDataset.txt Section 4.2, Section 6
"""
from cleanlab.filter import find_label_issues
from cleanlab.dataset import overall_label_health_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
import numpy as np


class CleanlabPipeline:
    """
    Confident Learning for label noise detection
    Source: Augmenting_NeoDataset.txt Section 4.2, Section 6
    
    Theory: Estimates joint distribution of noisy labels and true labels
    Method: Uses classifier confidence to detect label-data mismatches
    """
    
    def __init__(self, df):
        self.df = df
        self.label_issues = None
        self.pred_probs = None
    
    def train_preliminary_model(self):
        """
        Train lightweight classifier on Snorkel labels
        
        Purpose: Get out-of-sample predicted probabilities
        These probabilities reveal where labels disagree with semantic context
        
        Source: Augmenting_NeoDataset.txt Section 6, Step 2
        """
        print("\n" + "="*70)
        print("[STAGE 3] Training Preliminary Classifier for Label Noise Detection")
        print("="*70)
        
        # Prepare data
        print("\n[DATA] Preparing data for preliminary model...")
        X_text = self.df['full_text']
        y = self.df['risk_label_binary']
        
        print(f"  Total stories: {len(X_text)}")
        print(f"  Label distribution:")
        label_counts = y.value_counts()
        for label, count in label_counts.items():
            label_name = 'SAFE' if label == 0 else 'RISK'
            print(f"    {label_name} ({label}): {count} ({count/len(y)*100:.1f}%)")
        
        # TF-IDF vectorization
        print("\n[VECTORIZE] Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=1000, max_df=0.8, min_df=2)
        X = vectorizer.fit_transform(X_text)
        print(f"  ✓ TF-IDF matrix shape: {X.shape}")
        print(f"  ✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"  ✓ Sparsity: {(1.0 - X.nnz / (X.shape[0] * X.shape[1]))*100:.1f}%")
        
        # Train simple classifier
        print("\n[TRAIN] Training Logistic Regression classifier...")
        clf = LogisticRegression(max_iter=500, random_state=42)
        print(f"  Model: LogisticRegression (max_iter=500)")
        
        # Get out-of-sample predicted probabilities via cross-validation
        # This prevents overfitting and gives honest probability estimates
        print("\n[CROSS-VAL] Computing 5-fold cross-validated predictions...")
        print("  This may take a few minutes...")
        import time
        start_time = time.time()
        
        pred_probs = cross_val_predict(
            clf, X, y, 
            cv=5, 
            method='predict_proba',
            n_jobs=-1
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Cross-validation complete in {elapsed:.1f}s")
        print(f"  ✓ Generated predicted probabilities: {pred_probs.shape}")
        
        # Log prediction statistics
        print("\n[PREDICTIONS] Probability statistics:")
        safe_probs = pred_probs[:, 0]
        risk_probs = pred_probs[:, 1]
        print(f"  P(SAFE) - mean: {safe_probs.mean():.3f}, std: {safe_probs.std():.3f}")
        print(f"  P(RISK) - mean: {risk_probs.mean():.3f}, std: {risk_probs.std():.3f}")
        
        print(f"\n✓ Preliminary model training complete")
        return pred_probs
    
    def find_label_issues(self, pred_probs):
        """
        Detect label errors using Cleanlab
        
        Logic: If classifier is 99% confident story is SAFE, but Snorkel
        labeled it RISK, this is likely a label error.
        
        Source: Augmenting_NeoDataset.txt Section 6, Step 3
        """
        print("\n" + "="*70)
        print("[CLEANLAB] Detecting Label Issues")
        print("="*70)
        
        y = self.df['risk_label_binary'].values
        
        print(f"\n[DETECT] Running Confident Learning algorithm...")
        print(f"  Input: {len(y)} labels + predicted probabilities")
        
        # Find label issues (returns boolean mask)
        label_issues_mask = find_label_issues(
            labels=y,
            pred_probs=pred_probs
        )
        
        self.label_issues = label_issues_mask
        
        num_issues = label_issues_mask.sum()
        num_clean = len(self.df) - num_issues
        
        print(f"\n[RESULTS] Label quality assessment:")
        print(f"  Total stories: {len(self.df)}")
        print(f"  Clean labels: {num_clean} ({num_clean/len(self.df)*100:.1f}%)")
        print(f"  Issues detected: {num_issues} ({num_issues/len(self.df)*100:.1f}%)")
        
        # Breakdown by original label
        issues_by_label = {}
        for label in [0, 1]:
            label_mask = (y == label).values if hasattr(y, 'values') else (y == label)
            label_issues = (label_issues_mask & label_mask).sum()
            label_total = label_mask.sum()
            issues_by_label[label] = (label_issues, label_total)
            label_name = 'SAFE' if label == 0 else 'RISK'
            if label_total > 0:
                print(f"  {label_name} label issues: {label_issues}/{label_total} ({label_issues/label_total*100:.1f}%)")
        
        print(f"\n✓ Label issue detection complete")
        
        return label_issues_mask
    
    def calculate_label_health(self, pred_probs):
        """
        Calculate overall label health score
        
        Score: 0-1 (higher is better)
        Interpretation:
        - >0.8: Excellent label quality
        - 0.6-0.8: Good quality
        - 0.4-0.6: Moderate quality
        - <0.4: Poor quality (refine LFs)
        
        Source: Augmenting_NeoDataset.txt Section 6, Step 5
        """
        print("\n=== Calculating Label Health Score ===")
        
        y = self.df['risk_label_binary'].values
        
        health_score = overall_label_health_score(
            labels=y,
            pred_probs=pred_probs
        )
        
        print(f"Overall Label Health Score: {health_score:.3f}")
        print(f"(Range: 0-1, higher is better)")
        
        # Interpretation
        if health_score >= 0.8:
            print("→ Excellent label quality!")
        elif health_score >= 0.6:
            print("→ Good label quality")
        elif health_score >= 0.4:
            print("→ Moderate label quality")
        else:
            print("→ Consider refining labeling functions")
        
        return health_score
    
    def filter_dataset(self):
        """
        Remove label issues from dataset
        
        Strategy: Pruning (not re-weighting)
        Rationale: We have 20K+ examples, can afford to discard ambiguous ones
        
        Source: Augmenting_NeoDataset.txt Section 6, Step 4
        """
        print("\n" + "="*70)
        print("[FILTER] Removing Noisy Labels")
        print("="*70)
        
        if self.label_issues is None:
            raise ValueError("Must run find_label_issues first")
        
        # Keep only clean examples
        print("\n[PRUNE] Filtering dataset...")
        df_clean = self.df[~self.label_issues].copy()
        
        print(f"\n[STATISTICS] Filtering results:")
        print(f"  Original dataset: {len(self.df)} stories")
        print(f"  Issues removed: {self.label_issues.sum()} stories")
        print(f"  Clean dataset: {len(df_clean)} stories")
        print(f"  Retention rate: {len(df_clean)/len(self.df)*100:.1f}%")
        
        # Compare label distributions before/after
        print(f"\n[LABEL DISTRIBUTION] Before vs After filtering:")
        before_counts = self.df['risk_label'].value_counts()
        after_counts = df_clean['risk_label'].value_counts()
        
        for label in ['SAFE', 'RISK']:
            before = before_counts.get(label, 0)
            after = after_counts.get(label, 0)
            removed = before - after
            print(f"  {label}:")
            print(f"    Before: {before} ({before/len(self.df)*100:.1f}%)")
            print(f"    After:  {after} ({after/len(df_clean)*100:.1f}%)")
            print(f"    Removed: {removed} ({removed/before*100:.1f}% of {label} labels)")
        
        # Validate output
        print("\n[VALIDATION] Validating filtered dataset...")
        required_cols = ['risk_label', 'risk_confidence', 'full_text', 'story_points']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        if missing_cols:
            print(f"  ✗ ERROR: Missing columns after filtering: {missing_cols}")
            raise ValueError(f"Filtering failed: missing columns {missing_cols}")
        print(f"  ✓ All required columns present")
        
        print(f"\n✓ Dataset filtering complete")
        
        return df_clean
    
    def run_full_pipeline(self):
        """Execute complete Cleanlab pipeline"""
        self.pred_probs = self.train_preliminary_model()
        self.find_label_issues(self.pred_probs)
        health_score = self.calculate_label_health(self.pred_probs)
        df_clean = self.filter_dataset()
        
        return df_clean, health_score
    
    def log_cleanlab_diagnostics(self, tracker, pred_probs):
        """
        Log Cleanlab diagnostics to experiment tracker.
        
        Args:
            tracker: ExperimentTracker instance
            pred_probs: Predicted probabilities from preliminary model
        """
        from cleanlab.dataset import health_summary
        
        print("\n[TRACKER] Logging Cleanlab diagnostics to MLflow...")
        
        # 1. Calculate overall label health
        y = self.df['risk_label_binary'].values
        health = health_summary(y, pred_probs, verbose=False)
        
        # 2. Log health metrics
        tracker.log_stage_metrics("cleanlab", {
            "overall_health_score": health['overall_label_health_score'],
            "num_issues_detected": self.label_issues.sum() if self.label_issues is not None else 0,
            "pct_data_pruned": (self.label_issues.sum() / len(self.df) * 100) if self.label_issues is not None else 0
        })
        
        # 3. Log per-class statistics
        if self.label_issues is not None:
            for label_name, label_val in [('SAFE', 0), ('RISK', 1)]:
                mask = y == label_val
                issues_in_class = self.label_issues[mask].sum()
                total_in_class = mask.sum()
                
                tracker.log_stage_metrics("cleanlab", {
                    f"issues_{label_name}": issues_in_class,
                    f"pct_pruned_{label_name}": (issues_in_class / total_in_class * 100) if total_in_class > 0 else 0
                })
        
        # 4. Log retention stats
        if self.label_issues is not None:
            retention_rate = (len(self.df) - self.label_issues.sum()) / len(self.df)
            tracker.log_stage_metrics("cleanlab", {
                "retention_rate": retention_rate * 100,
                "original_size": len(self.df),
                "filtered_size": len(self.df) - self.label_issues.sum()
            })
        
        print("  ✓ Cleanlab diagnostics logged to MLflow")

