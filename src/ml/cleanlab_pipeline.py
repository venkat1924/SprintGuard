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
    
    def train_preliminary_model(self):
        """
        Train lightweight classifier on Snorkel labels
        
        Purpose: Get out-of-sample predicted probabilities
        These probabilities reveal where labels disagree with semantic context
        
        Source: Augmenting_NeoDataset.txt Section 6, Step 2
        """
        print("\n=== Training Preliminary Classifier ===")
        
        # Prepare data
        X_text = self.df['full_text']
        y = self.df['risk_label_binary']
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=1000, max_df=0.8, min_df=2)
        X = vectorizer.fit_transform(X_text)
        
        # Train simple classifier
        clf = LogisticRegression(max_iter=500, random_state=42)
        
        # Get out-of-sample predicted probabilities via cross-validation
        # This prevents overfitting and gives honest probability estimates
        print("Computing cross-validated predictions...")
        pred_probs = cross_val_predict(
            clf, X, y, 
            cv=5, 
            method='predict_proba',
            n_jobs=-1
        )
        
        print(f"✓ Generated predicted probabilities for {len(pred_probs)} stories")
        return pred_probs
    
    def find_label_issues(self, pred_probs):
        """
        Detect label errors using Cleanlab
        
        Logic: If classifier is 99% confident story is SAFE, but Snorkel
        labeled it RISK, this is likely a label error.
        
        Source: Augmenting_NeoDataset.txt Section 6, Step 3
        """
        print("\n=== Detecting Label Issues with Cleanlab ===")
        
        y = self.df['risk_label_binary'].values
        
        # Find label issues
        label_issues_mask = find_label_issues(
            labels=y,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence'
        )
        
        self.label_issues = label_issues_mask
        
        num_issues = label_issues_mask.sum()
        print(f"✓ Detected {num_issues} label issues ({num_issues/len(self.df):.2%})")
        
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
        if self.label_issues is None:
            raise ValueError("Must run find_label_issues first")
        
        # Keep only clean examples
        df_clean = self.df[~self.label_issues].copy()
        
        print(f"\n✓ Filtered dataset:")
        print(f"  Original: {len(self.df)} stories")
        print(f"  Issues removed: {self.label_issues.sum()}")
        print(f"  Clean: {len(df_clean)} stories")
        
        return df_clean
    
    def run_full_pipeline(self):
        """Execute complete Cleanlab pipeline"""
        pred_probs = self.train_preliminary_model()
        self.find_label_issues(pred_probs)
        health_score = self.calculate_label_health(pred_probs)
        df_clean = self.filter_dataset()
        
        return df_clean, health_score

