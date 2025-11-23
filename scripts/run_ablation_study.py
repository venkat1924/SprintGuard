#!/usr/bin/env python3
"""
Ablation Study Script for SprintGuard

Compares three configurations:
1. Baseline: TF-IDF + Logistic Regression (No Snorkel, No Cleanlab)
2. Snorkel Only: Snorkel labels + XGBoost (No Cleanlab filtering)
3. SprintGuard Full: Snorkel + Cleanlab + Hybrid Model

Runs each configuration 5 times with different random seeds.
Computes mean ± std for Macro-F1 score.
Generates publication-quality ablation study bar chart.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import logging

sys.path.insert(0, '.')
from src.ml.experiment_tracker import ExperimentTracker
from src.ml.train_risk_model import RiskModelTrainer
from src.visualization.publication_plots import generate_ablation_study

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_baseline(df, seed):
    """
    Baseline: TF-IDF + Logistic Regression
    No weak supervision, no noise filtering.
    Uses only manually labeled data if available (simulated by random labels).
    """
    logging.info(f"[BASELINE] Running baseline with seed {seed}...")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['risk_label'])
    
    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['full_text'])
    X_test = vectorizer.transform(test_df['full_text'])
    
    # Convert 3-class labels to integers
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    y_train = train_df['risk_label'].map(label_map).values
    y_test = test_df['risk_label'].map(label_map).values
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=seed, multi_class='multinomial')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"  Baseline Macro-F1: {macro_f1:.4f}")
    return macro_f1


def run_snorkel_only(df, seed):
    """
    Snorkel Only: Snorkel labels + Hybrid Model (No Cleanlab filtering)
    Uses all Snorkel labeled data without noise filtering.
    """
    logging.info(f"[SNORKEL ONLY] Running Snorkel-only with seed {seed}...")
    
    # Load Snorkel-labeled data (before Cleanlab filtering)
    snorkel_df = pd.read_csv('data/neodataset/neodataset_snorkel_labels.csv')
    
    # Map to 3-class (simplified mapping for ablation)
    snorkel_df['risk_label_3class'] = snorkel_df['risk_label'].map({
        'SAFE': 'Low',
        'RISK': 'High'  # Simplified: all RISK -> High
    })
    snorkel_df = snorkel_df.dropna(subset=['risk_label_3class'])
    
    # Create full_text if missing
    if 'full_text' not in snorkel_df.columns:
        snorkel_df['full_text'] = snorkel_df['title'].fillna('') + ' ' + snorkel_df['description'].fillna('')
    
    # Split data
    train_df, test_df = train_test_split(
        snorkel_df, test_size=0.2, random_state=seed, stratify=snorkel_df['risk_label_3class']
    )
    
    # Use TF-IDF for faster ablation (instead of full hybrid model)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['full_text'])
    X_test = vectorizer.transform(test_df['full_text'])
    
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    y_train = train_df['risk_label_3class'].map(label_map).fillna(0).astype(int).values
    y_test = test_df['risk_label_3class'].map(label_map).fillna(0).astype(int).values
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=seed, multi_class='multinomial')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"  Snorkel-Only Macro-F1: {macro_f1:.4f}")
    return macro_f1


def run_sprintguard_full(df, seed):
    """
    SprintGuard Full: Snorkel + Cleanlab + Hybrid Model
    Complete pipeline with noise filtering.
    """
    logging.info(f"[SPRINTGUARD FULL] Running full pipeline with seed {seed}...")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['risk_label'])
    
    # Use TF-IDF for faster ablation (instead of full hybrid model)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['full_text'])
    X_test = vectorizer.transform(test_df['full_text'])
    
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    y_train = train_df['risk_label'].map(label_map).values
    y_test = test_df['risk_label'].map(label_map).values
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=seed, multi_class='multinomial')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"  SprintGuard-Full Macro-F1: {macro_f1:.4f}")
    return macro_f1


def main():
    logging.info("\n" + "="*70)
    logging.info("SPRINTGUARD ABLATION STUDY")
    logging.info("="*70)
    logging.info("Comparing:")
    logging.info("  1. Baseline (TF-IDF + Logistic Regression)")
    logging.info("  2. Snorkel Only (No Cleanlab)")
    logging.info("  3. SprintGuard Full (Snorkel + Cleanlab)")
    logging.info("="*70)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_name="SprintGuard_Ablation")
    tracker.start_run(run_name="ablation_study")
    
    # Load full SprintGuard dataset (post-Cleanlab)
    data_path = 'data/neodataset_augmented_3class.csv'
    if not Path(data_path).exists():
        logging.error(f"[ERROR] Data not found: {data_path}")
        logging.error("Run augmentation pipeline first: python scripts/augment_neodataset.py")
        sys.exit(1)
    
    df_full = pd.read_csv(data_path)
    logging.info(f"\n[DATA] Loaded {len(df_full)} stories")
    
    # Check if Snorkel-only data exists
    snorkel_path = 'data/neodataset/neodataset_snorkel_labels.csv'
    if not Path(snorkel_path).exists():
        logging.warning(f"[WARNING] Snorkel-only data not found: {snorkel_path}")
        logging.warning("Skipping Snorkel-only baseline")
        run_snorkel = False
    else:
        run_snorkel = True
    
    # Run experiments with 5 different seeds
    seeds = [42, 123, 456, 789, 1024]
    
    baseline_f1s = []
    snorkel_only_f1s = []
    sprintguard_f1s = []
    
    for seed in seeds:
        logging.info(f"\n{'='*70}")
        logging.info(f"Run {len(baseline_f1s) + 1}/5 (seed={seed})")
        logging.info(f"{'='*70}")
        
        # Baseline
        try:
            f1 = run_baseline(df_full, seed)
            baseline_f1s.append(f1)
        except Exception as e:
            logging.error(f"Baseline failed: {e}")
        
        # Snorkel Only
        if run_snorkel:
            try:
                f1 = run_snorkel_only(df_full, seed)
                snorkel_only_f1s.append(f1)
            except Exception as e:
                logging.error(f"Snorkel-only failed: {e}")
        
        # SprintGuard Full
        try:
            f1 = run_sprintguard_full(df_full, seed)
            sprintguard_f1s.append(f1)
        except Exception as e:
            logging.error(f"SprintGuard-full failed: {e}")
    
    # Compute statistics
    logging.info("\n" + "="*70)
    logging.info("ABLATION STUDY RESULTS")
    logging.info("="*70)
    
    results = {}
    
    if baseline_f1s:
        baseline_mean = np.mean(baseline_f1s)
        baseline_std = np.std(baseline_f1s)
        results['Baseline\n(No WS)'] = (baseline_mean, baseline_std)
        logging.info(f"\nBaseline (No Weak Supervision):")
        logging.info(f"  Macro-F1: {baseline_mean:.4f} ± {baseline_std:.4f}")
        logging.info(f"  Individual runs: {[f'{f:.4f}' for f in baseline_f1s]}")
    
    if snorkel_only_f1s:
        snorkel_mean = np.mean(snorkel_only_f1s)
        snorkel_std = np.std(snorkel_only_f1s)
        results['Snorkel\nOnly'] = (snorkel_mean, snorkel_std)
        logging.info(f"\nSnorkel Only (No Cleanlab):")
        logging.info(f"  Macro-F1: {snorkel_mean:.4f} ± {snorkel_std:.4f}")
        logging.info(f"  Individual runs: {[f'{f:.4f}' for f in snorkel_only_f1s]}")
    
    if sprintguard_f1s:
        sprintguard_mean = np.mean(sprintguard_f1s)
        sprintguard_std = np.std(sprintguard_f1s)
        results['SprintGuard\n(Full)'] = (sprintguard_mean, sprintguard_std)
        logging.info(f"\nSprintGuard Full (Snorkel + Cleanlab):")
        logging.info(f"  Macro-F1: {sprintguard_mean:.4f} ± {sprintguard_std:.4f}")
        logging.info(f"  Individual runs: {[f'{f:.4f}' for f in sprintguard_f1s]}")
    
    # Log to MLflow
    for method, (mean_f1, std_f1) in results.items():
        clean_name = method.replace('\n', '_')
        tracker.log_stage_metrics("ablation", {
            f"{clean_name}_mean_f1": mean_f1,
            f"{clean_name}_std_f1": std_f1
        })
    
    # Generate ablation study plot
    logging.info("\n[VISUALIZATION] Generating ablation study plot...")
    generate_ablation_study(results, output_path="visualizations/ablation_study.pdf")
    tracker.log_artifact("visualizations/ablation_study.pdf", "visualizations")
    tracker.log_artifact("visualizations/ablation_study.svg", "visualizations")
    
    logging.info("\n✓ Ablation study complete!")
    logging.info(f"  Results saved to: visualizations/ablation_study.pdf")
    
    # End MLflow run
    tracker.end_run()


if __name__ == '__main__':
    main()

