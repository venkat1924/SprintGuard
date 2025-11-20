"""
Snorkel Weak Supervision Pipeline
Combines labeling functions to generate probabilistic training labels
Source: Augmenting_NeoDataset.txt Section 4.1
"""
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
import pandas as pd
import numpy as np


class WeakSupervisionPipeline:
    """
    Implements Snorkel-based weak supervision
    Source: Augmenting_NeoDataset.txt Section 4.1
    
    Mathematical basis: Data Programming paradigm
    - Multiple noisy labeling functions (LFs) vote on each example
    - Generative model learns LF accuracies and correlations
    - Outputs probabilistic labels without ground truth
    """
    
    def __init__(self, df, labeling_functions):
        self.df = df
        self.lfs = labeling_functions
        self.L_matrix = None
        self.label_model = None
    
    def apply_labeling_functions(self):
        """
        Apply all LFs to generate label matrix
        L_matrix[i,j] = vote of LF j on example i
        Values: -1 (ABSTAIN), 0 (SAFE), 1 (RISK)
        """
        print("\n=== Applying Labeling Functions ===")
        applier = PandasLFApplier(lfs=self.lfs)
        self.L_matrix = applier.apply(df=self.df)
        
        print(f"Label matrix shape: {self.L_matrix.shape}")
        print(f"({len(self.df)} stories × {len(self.lfs)} LFs)")
        
        return self.L_matrix
    
    def analyze_lf_performance(self):
        """
        Generate LF statistics
        Key metrics:
        - Coverage: % of examples each LF labels (not ABSTAIN)
        - Overlaps: How many LFs agree on each example
        - Conflicts: How many LFs disagree on same example
        """
        print("\n=== Labeling Function Analysis ===")
        analysis = LFAnalysis(L=self.L_matrix, lfs=self.lfs).lf_summary()
        
        print(analysis)
        
        # Additional metrics
        coverage = (self.L_matrix != -1).any(axis=1).mean()
        print(f"\nOverall Coverage: {coverage:.2%}")
        print(f"(% of stories labeled by at least one LF)")
        
        # Check for label agreement
        num_votes = (self.L_matrix != -1).sum(axis=1)
        print(f"\nAverage LF votes per story: {num_votes.mean():.1f}")
        
        return analysis
    
    def train_label_model(self, n_epochs=500):
        """
        Train Snorkel's generative model
        
        The model learns:
        1. Accuracy of each LF (without knowing true labels)
        2. Correlations between LFs
        3. Weights for combining LF votes
        
        Source: Augmenting_NeoDataset.txt Section 4.1
        """
        print("\n=== Training Snorkel Label Model ===")
        self.label_model = LabelModel(cardinality=2, verbose=True)
        
        self.label_model.fit(
            L_train=self.L_matrix,
            n_epochs=n_epochs,
            log_freq=50,
            seed=42
        )
        
        print("✓ Label model trained")
        return self.label_model
    
    def generate_probabilistic_labels(self):
        """
        Generate final probabilistic labels
        
        Returns:
            DataFrame with new columns:
            - risk_label_binary: 0 (SAFE) or 1 (RISK)
            - risk_label: 'SAFE' or 'RISK'
            - risk_prob_safe: P(SAFE)
            - risk_prob_risk: P(RISK)
            - risk_confidence: max(P(SAFE), P(RISK))
        """
        print("\n=== Generating Probabilistic Labels ===")
        
        # Hard labels (0 or 1)
        labels_hard = self.label_model.predict(L=self.L_matrix)
        
        # Soft labels (probabilities)
        labels_soft = self.label_model.predict_proba(L=self.L_matrix)
        
        # Add to DataFrame
        self.df['risk_label_binary'] = labels_hard
        self.df['risk_label'] = self.df['risk_label_binary'].map({
            0: 'SAFE',
            1: 'RISK'
        })
        
        self.df['risk_prob_safe'] = labels_soft[:, 0]
        self.df['risk_prob_risk'] = labels_soft[:, 1]
        
        # Confidence = max probability
        self.df['risk_confidence'] = labels_soft.max(axis=1)
        
        print(f"\n✓ Generated labels for {len(self.df)} stories")
        print(f"\nLabel distribution:")
        print(self.df['risk_label'].value_counts())
        print(f"\nConfidence statistics:")
        print(self.df['risk_confidence'].describe())
        
        return self.df
    
    def run_full_pipeline(self):
        """Execute complete weak supervision pipeline"""
        self.apply_labeling_functions()
        self.analyze_lf_performance()
        self.train_label_model()
        labeled_df = self.generate_probabilistic_labels()
        return labeled_df

