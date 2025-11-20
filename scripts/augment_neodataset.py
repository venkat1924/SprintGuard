#!/usr/bin/env python3
"""
Main NeoDataset Augmentation Script
Orchestrates the complete weak supervision pipeline

Usage:
    python scripts/augment_neodataset.py
"""
import sys
sys.path.insert(0, '.')

from src.ml.neodataset_loader import load_neodataset, preprocess_neodataset
from src.ml.labeling_functions import ALL_LABELING_FUNCTIONS
from src.ml.weak_supervision_pipeline import WeakSupervisionPipeline
from src.ml.cleanlab_pipeline import CleanlabPipeline
import pandas as pd


def main():
    print('='*60)
    print('NeoDataset Augmentation Pipeline')
    print('Research-Backed Weak Supervision for Risk Labeling')
    print('='*60)
    
    # Step 1: Load and preprocess
    print('\n[1/4] Loading NeoDataset...')
    df = load_neodataset()
    df = preprocess_neodataset(df)
    
    # Step 2: Weak Supervision (Snorkel)
    print('\n[2/4] Running Weak Supervision (Snorkel)...')
    print(f'Applying {len(ALL_LABELING_FUNCTIONS)} research-backed labeling functions...')
    
    ws_pipeline = WeakSupervisionPipeline(df, ALL_LABELING_FUNCTIONS)
    df_labeled = ws_pipeline.run_full_pipeline()
    
    # Save intermediate result
    df_labeled.to_csv('data/neodataset/neodataset_snorkel_labels.csv', index=False)
    print('\n✓ Saved Snorkel labels to data/neodataset/neodataset_snorkel_labels.csv')
    
    # Step 3: Noise Remediation (Cleanlab)
    print('\n[3/4] Running Noise Remediation (Cleanlab)...')
    cleanlab_pipeline = CleanlabPipeline(df_labeled)
    df_clean, health_score = cleanlab_pipeline.run_full_pipeline()
    
    # Step 4: Save final augmented dataset
    print('\n[4/4] Saving final augmented dataset...')
    
    # Full augmented dataset
    output_path = 'data/neodataset_augmented.csv'
    df_clean.to_csv(output_path, index=False)
    print(f'✓ Saved to {output_path}')
    
    # High-confidence subset (for training)
    high_conf_df = df_clean[df_clean['risk_confidence'] > 0.75]
    high_conf_path = 'data/neodataset_augmented_high_confidence.csv'
    high_conf_df.to_csv(high_conf_path, index=False)
    print(f'✓ Saved high-confidence subset ({len(high_conf_df)} stories) to {high_conf_path}')
    
    # Summary statistics
    print('\n' + '='*60)
    print('AUGMENTATION COMPLETE')
    print('='*60)
    print(f'Total stories processed: {len(df)}')
    print(f'Stories with clean labels: {len(df_clean)}')
    print(f'High-confidence stories: {len(high_conf_df)}')
    print(f'Label health score: {health_score:.3f}')
    print(f'\nFinal label distribution:')
    print(df_clean['risk_label'].value_counts())
    print(f'\nLabel distribution (%):')
    print(df_clean['risk_label'].value_counts(normalize=True) * 100)
    print('\n✓ Ready for ML model training!')
    print('\nNext steps:')
    print('1. Review data/neodataset_augmented.csv')
    print('2. (Optional) Create manual validation set')
    print('3. Train ML model on high-confidence subset')


if __name__ == '__main__':
    main()
