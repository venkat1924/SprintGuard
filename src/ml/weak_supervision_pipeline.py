"""
Snorkel Weak Supervision Pipeline
Combines labeling functions to generate probabilistic training labels
Source: Augmenting_NeoDataset.txt Section 4.1
"""
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


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
        print("\n" + "="*70)
        print("[STAGE 2] Applying Labeling Functions")
        print("="*70)
        print(f"Number of labeling functions: {len(self.lfs)}")
        print(f"Number of stories: {len(self.df)}")
        
        print("\n[APPLY LFS] Running labeling functions...")
        applier = PandasLFApplier(lfs=self.lfs)
        self.L_matrix = applier.apply(df=self.df)
        
        print(f"\n[LABEL MATRIX] Generated label matrix")
        print(f"  Shape: {self.L_matrix.shape} ({len(self.df)} stories × {len(self.lfs)} LFs)")
        print(f"  Total votes: {self.L_matrix.size}")
        
        # Count vote types
        total_votes = self.L_matrix.size
        abstains = (self.L_matrix == -1).sum()
        safe_votes = (self.L_matrix == 0).sum()
        risk_votes = (self.L_matrix == 1).sum()
        
        print(f"\n[VOTE DISTRIBUTION]")
        print(f"  ABSTAIN: {abstains} ({abstains/total_votes*100:.1f}%)")
        print(f"  SAFE:    {safe_votes} ({safe_votes/total_votes*100:.1f}%)")
        print(f"  RISK:    {risk_votes} ({risk_votes/total_votes*100:.1f}%)")
        
        print(f"\n✓ Labeling functions applied successfully")
        
        return self.L_matrix
    
    def analyze_lf_performance(self):
        """
        Generate LF statistics
        Key metrics:
        - Coverage: % of examples each LF labels (not ABSTAIN)
        - Overlaps: How many LFs agree on each example
        - Conflicts: How many LFs disagree on same example
        """
        print("\n" + "="*70)
        print("[ANALYSIS] Labeling Function Performance")
        print("="*70)
        
        analysis = LFAnalysis(L=self.L_matrix, lfs=self.lfs).lf_summary()
        
        print("\n[LF SUMMARY]")
        print(analysis)
        
        # Additional metrics
        coverage = (self.L_matrix != -1).any(axis=1).mean()
        print(f"\n[COVERAGE METRICS]")
        print(f"  Overall Coverage: {coverage:.2%}")
        print(f"  (% of stories labeled by at least one LF)")
        
        # Stories with no labels
        no_label_count = (~(self.L_matrix != -1).any(axis=1)).sum()
        print(f"  Stories with no labels: {no_label_count} ({no_label_count/len(self.df)*100:.1f}%)")
        
        # Check for label agreement
        num_votes = (self.L_matrix != -1).sum(axis=1)
        print(f"\n[AGREEMENT METRICS]")
        print(f"  Average LF votes per story: {num_votes.mean():.1f}")
        print(f"  Min votes: {num_votes.min()}")
        print(f"  Max votes: {num_votes.max()}")
        
        # Detect conflicts (LFs disagree on same story)
        safe_votes = (self.L_matrix == 0).sum(axis=1)
        risk_votes = (self.L_matrix == 1).sum(axis=1)
        conflicts = ((safe_votes > 0) & (risk_votes > 0)).sum()
        print(f"  Stories with conflicting labels: {conflicts} ({conflicts/len(self.df)*100:.1f}%)")
        
        print(f"\n✓ LF analysis complete")
        
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
        print("\n" + "="*70)
        print("[LABELS] Generating Probabilistic Labels")
        print("="*70)
        
        # Hard labels (0 or 1, or -1 for ABSTAIN)
        print("\n[PREDICT] Computing hard labels...")
        labels_hard = self.label_model.predict(L=self.L_matrix)
        print(f"  ✓ Generated {len(labels_hard)} hard labels")
        
        # Check for -1 (ABSTAIN) or NaN in hard labels
        abstain_count = (labels_hard == -1).sum()
        nan_hard_count = pd.isna(labels_hard).sum()
        if abstain_count > 0 or nan_hard_count > 0:
            total_uncertain = abstain_count + nan_hard_count
            print(f"  ⚠ Warning: {total_uncertain} stories have uncertain labels")
            print(f"    (-1/ABSTAIN: {abstain_count}, NaN: {nan_hard_count})")
            print(f"    These will be marked as SAFE (class 0) with confidence 0.5")
        
        # Soft labels (probabilities)
        print("\n[PREDICT] Computing probabilistic labels...")
        labels_soft = self.label_model.predict_proba(L=self.L_matrix)
        print(f"  ✓ Generated probability distributions")
        print(f"  Shape: {labels_soft.shape}")
        
        # Handle -1 (ABSTAIN) and NaN values before adding to DataFrame
        # For stories with no LF votes: assign SAFE (0) with 50% confidence
        labels_hard_series = pd.Series(labels_hard)
        labels_hard_series = labels_hard_series.replace(-1, 0)  # Convert ABSTAIN to SAFE
        labels_hard_filled = labels_hard_series.fillna(0).astype(int).values
        labels_soft_filled = pd.DataFrame(labels_soft).fillna(0.5).values
        
        # Add to DataFrame
        print("\n[DATAFRAME] Adding label columns...")
        self.df['risk_label_binary'] = labels_hard_filled
        
        # Debug: Check for unexpected values
        unique_binary = self.df['risk_label_binary'].unique()
        print(f"  Debug: Unique risk_label_binary values: {sorted(unique_binary)}")
        nan_binary_count = self.df['risk_label_binary'].isna().sum()
        if nan_binary_count > 0:
            print(f"  ⚠ Warning: Found {nan_binary_count} NaN values in risk_label_binary after fillna!")
        
        # Map with explicit NaN handling
        self.df['risk_label'] = self.df['risk_label_binary'].fillna(0).astype(int).map({
            0: 'SAFE',
            1: 'RISK'
        })
        
        # Fill any remaining NaN values (shouldn't happen, but just in case)
        nan_label_count = self.df['risk_label'].isna().sum()
        if nan_label_count > 0:
            print(f"  ⚠ Warning: Found {nan_label_count} NaN in risk_label after mapping, filling with 'SAFE'")
            self.df['risk_label'] = self.df['risk_label'].fillna('SAFE')
        
        self.df['risk_prob_safe'] = labels_soft_filled[:, 0]
        self.df['risk_prob_risk'] = labels_soft_filled[:, 1]
        
        # Confidence = max probability
        self.df['risk_confidence'] = labels_soft_filled.max(axis=1)
        
        print(f"  ✓ Added 5 new columns:")
        print(f"    - risk_label_binary (0/1)")
        print(f"    - risk_label (SAFE/RISK)")
        print(f"    - risk_prob_safe")
        print(f"    - risk_prob_risk")
        print(f"    - risk_confidence")
        
        # Label distribution
        print(f"\n[LABEL DISTRIBUTION]")
        label_counts = self.df['risk_label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Confidence statistics
        print(f"\n[CONFIDENCE STATISTICS]")
        print(f"  Mean:   {self.df['risk_confidence'].mean():.3f}")
        print(f"  Median: {self.df['risk_confidence'].median():.3f}")
        print(f"  Min:    {self.df['risk_confidence'].min():.3f}")
        print(f"  Max:    {self.df['risk_confidence'].max():.3f}")
        
        # High confidence stories
        high_conf_count = (self.df['risk_confidence'] > 0.75).sum()
        print(f"  High confidence (>0.75): {high_conf_count} ({high_conf_count/len(self.df)*100:.1f}%)")
        
        # Validate output
        print("\n[VALIDATION] Validating output schema...")
        required_cols = ['risk_label', 'risk_label_binary', 'risk_confidence', 
                        'risk_prob_safe', 'risk_prob_risk']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            print(f"  ✗ ERROR: Missing columns: {missing_cols}")
            raise ValueError(f"Label generation failed: missing columns {missing_cols}")
        
        # Validate risk_label values
        unique_labels = self.df['risk_label'].unique()
        if not set(unique_labels).issubset({'SAFE', 'RISK'}):
            print(f"  ✗ ERROR: Invalid risk_label values: {unique_labels}")
            raise ValueError(f"Invalid risk_label values: {unique_labels}")
        
        # Validate confidence range
        if self.df['risk_confidence'].min() < 0 or self.df['risk_confidence'].max() > 1:
            print(f"  ✗ ERROR: risk_confidence out of range [0, 1]")
            raise ValueError("risk_confidence values out of valid range")
        
        print(f"  ✓ Output validation passed")
        
        print(f"\n✓ Label generation complete: {len(self.df)} stories labeled")
        
        return self.df
    
    def run_full_pipeline(self):
        """Execute complete weak supervision pipeline"""
        self.apply_labeling_functions()
        self.analyze_lf_performance()
        self.train_label_model()
        labeled_df = self.generate_probabilistic_labels()
        return labeled_df
    
    def log_lf_diagnostics(self, tracker):
        """
        Log Snorkel diagnostics to experiment tracker and generate visualizations.
        
        Args:
            tracker: ExperimentTracker instance
        """
        from src.visualization.publication_plots import generate_lf_correlation_heatmap
        
        print("\n[TRACKER] Logging Snorkel diagnostics to MLflow...")
        
        # Get LF names
        lf_names = [lf.name for lf in self.lfs]
        
        # 1. Log coverage metrics (% non-abstain per LF)
        coverage = (self.L_matrix != -1).mean(axis=0)
        coverage_metrics = {f"lf_coverage/{lf_name}": cov for lf_name, cov in zip(lf_names, coverage)}
        tracker.log_stage_metrics("snorkel", coverage_metrics)
        
        # 2. Log overall statistics
        overall_coverage = (self.L_matrix != -1).any(axis=1).mean()
        num_votes_per_story = (self.L_matrix != -1).sum(axis=1).mean()
        
        safe_votes = (self.L_matrix == 0).sum(axis=1)
        risk_votes = (self.L_matrix == 1).sum(axis=1)
        conflicts = ((safe_votes > 0) & (risk_votes > 0)).sum()
        
        tracker.log_stage_metrics("snorkel", {
            "overall_coverage": overall_coverage,
            "avg_votes_per_story": num_votes_per_story,
            "num_conflicts": conflicts,
            "pct_conflicts": conflicts / len(self.df) * 100,
            "num_lfs": len(self.lfs),
            "num_stories": len(self.df)
        })
        
        # 3. Log label distribution
        if 'risk_label' in self.df.columns:
            label_counts = self.df['risk_label'].value_counts()
            for label, count in label_counts.items():
                tracker.log_stage_metrics("snorkel", {
                    f"label_count/{label}": count,
                    f"label_pct/{label}": count / len(self.df) * 100
                })
            
            # Confidence statistics
            tracker.log_stage_metrics("snorkel", {
                "confidence_mean": self.df['risk_confidence'].mean(),
                "confidence_median": self.df['risk_confidence'].median(),
                "confidence_min": self.df['risk_confidence'].min(),
                "confidence_max": self.df['risk_confidence'].max(),
                "high_confidence_count": (self.df['risk_confidence'] > 0.75).sum()
            })
        
        # 4. Generate and save LF correlation heatmap (publication figure)
        print("  Generating LF correlation heatmap...")
        generate_lf_correlation_heatmap(
            L_matrix=self.L_matrix,
            lf_names=lf_names,
            output_path="visualizations/lf_correlation_heatmap.pdf"
        )
        
        # Log the figure to MLflow
        tracker.log_artifact("visualizations/lf_correlation_heatmap.pdf", "visualizations")
        tracker.log_artifact("visualizations/lf_correlation_heatmap.svg", "visualizations")
        
        print("  ✓ Snorkel diagnostics logged to MLflow")

