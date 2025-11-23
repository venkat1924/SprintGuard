#!/usr/bin/env python3
"""
Main NeoDataset Augmentation Script
Orchestrates the complete weak supervision pipeline

Usage:
    python scripts/augment_neodataset.py
"""
import sys
import time
import subprocess
from pathlib import Path
sys.path.insert(0, '.')

from src.ml.neodataset_loader import load_neodataset, preprocess_neodataset
from src.ml.labeling_functions import ALL_LABELING_FUNCTIONS
from src.ml.weak_supervision_pipeline import WeakSupervisionPipeline
from src.ml.cleanlab_pipeline import CleanlabPipeline
from src.ml.experiment_tracker import ExperimentTracker
import pandas as pd


def main():
    print('='*70)
    print('NeoDataset Augmentation Pipeline')
    print('Research-Backed Weak Supervision for Risk Labeling')
    print('='*70)
    print('This pipeline includes 4 stages:')
    print('  Stage 1: Download & Preprocess NeoDataset')
    print('  Stage 2: Weak Supervision (Snorkel)')
    print('  Stage 3: Noise Filtering (Cleanlab)')
    print('  Stage 4: Label Mapping (Binary → 3-Class)')
    print('='*70)
    
    overall_start = time.time()
    
    # Initialize experiment tracker
    print('\n[MLFLOW] Initializing experiment tracker...')
    tracker = ExperimentTracker(experiment_name="SprintGuard_Augmentation")
    tracker.start_run(run_name=f"augmentation_{int(time.time())}")
    print('  ✓ MLflow tracking enabled')
    
    # Step 1: Load and preprocess
    print('\n[STAGE 1/4] Loading NeoDataset...')
    stage1_start = time.time()
    df = load_neodataset()
    df = preprocess_neodataset(df)
    stage1_time = time.time() - stage1_start
    print(f'✓ Stage 1 complete in {stage1_time:.1f}s')
    
    # Log Stage 1 metrics
    tracker.log_stage_count('raw', len(df))
    tracker.log_stage_metrics('stage1_preprocess', {
        'num_stories': len(df),
        'duration_seconds': stage1_time
    })
    
    # Validate Stage 1 → Stage 2
    print('\n[VALIDATION] Stage 1 → Stage 2 compatibility check...')
    required_cols = ['full_text', 'word_count', 'story_points']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f'  ✗ ERROR: Missing required columns: {missing_cols}')
        sys.exit(1)
    print(f'  ✓ Required columns present: {required_cols}')
    
    # Step 2: Weak Supervision (Snorkel)
    print('\n[STAGE 2/4] Running Weak Supervision (Snorkel)...')
    print(f'Applying {len(ALL_LABELING_FUNCTIONS)} research-backed labeling functions...')
    stage2_start = time.time()
    
    ws_pipeline = WeakSupervisionPipeline(df, ALL_LABELING_FUNCTIONS)
    df_labeled = ws_pipeline.run_full_pipeline()
    
    # Save intermediate result
    intermediate_path = 'data/neodataset/neodataset_snorkel_labels.csv'
    Path(intermediate_path).parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(intermediate_path, index=False)
    stage2_time = time.time() - stage2_start
    print(f'\n✓ Stage 2 complete in {stage2_time:.1f}s')
    print(f'✓ Saved Snorkel labels to {intermediate_path}')
    
    # Log Stage 2 metrics and generate visualizations
    tracker.log_stage_count('snorkel', len(df_labeled))
    tracker.log_stage_metrics('stage2_snorkel', {
        'duration_seconds': stage2_time
    })
    ws_pipeline.log_lf_diagnostics(tracker)
    
    # Validate Stage 2 → Stage 3
    print('\n[VALIDATION] Stage 2 → Stage 3 compatibility check...')
    required_cols = ['risk_label', 'risk_confidence', 'risk_label_binary']
    missing_cols = [col for col in required_cols if col not in df_labeled.columns]
    if missing_cols:
        print(f'  ✗ ERROR: Missing required columns: {missing_cols}')
        sys.exit(1)
    unique_labels = df_labeled['risk_label'].unique()
    if not set(unique_labels).issubset({'SAFE', 'RISK'}):
        print(f'  ✗ ERROR: Invalid risk_label values: {unique_labels}')
        sys.exit(1)
    print(f'  ✓ risk_label values valid: {unique_labels}')
    print(f'  ✓ risk_confidence range: [{df_labeled["risk_confidence"].min():.3f}, {df_labeled["risk_confidence"].max():.3f}]')
    
    # Step 3: Noise Remediation (Cleanlab)
    print('\n[STAGE 3/4] Running Noise Remediation (Cleanlab)...')
    stage3_start = time.time()
    cleanlab_pipeline = CleanlabPipeline(df_labeled)
    df_clean, health_score = cleanlab_pipeline.run_full_pipeline()
    stage3_time = time.time() - stage3_start
    print(f'\n✓ Stage 3 complete in {stage3_time:.1f}s')
    
    # Log Stage 3 metrics
    pred_probs = cleanlab_pipeline.pred_probs if hasattr(cleanlab_pipeline, 'pred_probs') else None
    if pred_probs is not None:
        cleanlab_pipeline.log_cleanlab_diagnostics(tracker, pred_probs)
    tracker.log_stage_count('cleanlab', len(df_clean))
    tracker.log_stage_metrics('stage3_cleanlab', {
        'duration_seconds': stage3_time,
        'health_score': health_score
    })
    
    # Save binary label versions (Stage 3 output)
    print('\n[SAVE] Saving binary-labeled datasets...')
    
    # Full augmented dataset (binary labels)
    output_path = 'data/neodataset_augmented.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f'  ✓ Saved to {output_path}')
    
    # High-confidence subset (binary labels)
    high_conf_df = df_clean[df_clean['risk_confidence'] > 0.75]
    high_conf_path = 'data/neodataset_augmented_high_confidence.csv'
    high_conf_df.to_csv(high_conf_path, index=False)
    print(f'  ✓ Saved high-confidence subset ({len(high_conf_df)} stories) to {high_conf_path}')
    
    # Step 4: Label Mapping (Binary → 3-Class)
    print('\n[STAGE 4/4] Mapping Binary Labels to 3-Class...')
    stage4_start = time.time()
    
    print('  Calling map_to_3class.py script...')
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/map_to_3class.py'],
            check=True,
            capture_output=False,
            text=True
        )
        stage4_time = time.time() - stage4_start
        print(f'\n✓ Stage 4 complete in {stage4_time:.1f}s')
    except subprocess.CalledProcessError as e:
        print(f'\n✗ ERROR: Label mapping failed')
        print(f'  You can run it manually: python scripts/map_to_3class.py')
        stage4_time = 0
    
    # Summary statistics
    overall_time = time.time() - overall_start
    print('\n' + '='*70)
    print('AUGMENTATION PIPELINE COMPLETE')
    print('='*70)
    
    print('\n[TIMING]')
    print(f'  Stage 1 (Download & Preprocess): {stage1_time:.1f}s')
    print(f'  Stage 2 (Weak Supervision):      {stage2_time:.1f}s')
    print(f'  Stage 3 (Noise Filtering):       {stage3_time:.1f}s')
    print(f'  Stage 4 (Label Mapping):         {stage4_time:.1f}s')
    print(f'  Total time:                      {overall_time:.1f}s ({overall_time/60:.1f}m)')
    
    print('\n[STATISTICS]')
    print(f'  Total stories processed: {len(df)}')
    print(f'  Stories with clean labels: {len(df_clean)}')
    print(f'  High-confidence stories: {len(high_conf_df)}')
    print(f'  Label health score: {health_score:.3f}')
    
    print(f'\n[BINARY LABELS] Distribution (SAFE/RISK):')
    binary_counts = df_clean['risk_label'].value_counts()
    for label, count in binary_counts.items():
        print(f'  {label}: {count} ({count/len(df_clean)*100:.1f}%)')
    
    # Check if 3-class files exist
    final_count = len(high_conf_df)
    if Path('data/neodataset_augmented_3class.csv').exists():
        df_3class = pd.read_csv('data/neodataset_augmented_3class.csv')
        print(f'\n[3-CLASS LABELS] Distribution (Low/Medium/High):')
        class3_counts = df_3class['risk_label'].value_counts()
        for label in ['Low', 'Medium', 'High']:
            count = class3_counts.get(label, 0)
            print(f'  {label}: {count} ({count/len(df_3class)*100:.1f}%)')
        
        # Use high-confidence 3-class as final training set
        if Path('data/neodataset_augmented_3class_high_confidence.csv').exists():
            df_3class_hc = pd.read_csv('data/neodataset_augmented_3class_high_confidence.csv')
            final_count = len(df_3class_hc)
    
    # Log final stage count and overall metrics
    tracker.log_stage_count('final', final_count)
    tracker.log_stage_metrics('overall', {
        'total_duration_seconds': overall_time,
        'total_duration_minutes': overall_time / 60,
        'stage1_duration': stage1_time,
        'stage2_duration': stage2_time,
        'stage3_duration': stage3_time,
        'stage4_duration': stage4_time
    })
    
    # Generate Sankey diagram
    print('\n[VISUALIZATION] Generating pipeline flow diagram...')
    from src.visualization.publication_plots import generate_sankey_diagram
    try:
        generate_sankey_diagram(
            stage_counts=tracker.stage_counts,
            output_path="visualizations/sankey_pipeline_flow.pdf"
        )
        tracker.log_artifact("visualizations/sankey_pipeline_flow.pdf", "visualizations")
        print('  ✓ Sankey diagram saved')
    except Exception as e:
        print(f'  ⚠ Could not generate Sankey diagram: {e}')
    
    # End MLflow run
    tracker.end_run()
    print('\n[MLFLOW] Experiment tracking complete')
    print('  View results: mlflow ui')
    
    print('\n[OUTPUT FILES]')
    print('  Binary labels (for API/reference):')
    print(f'    - data/neodataset_augmented.csv')
    print(f'    - data/neodataset_augmented_high_confidence.csv')
    print('  3-Class labels (for training):')
    print(f'    - data/neodataset_augmented_3class.csv')
    print(f'    - data/neodataset_augmented_3class_high_confidence.csv')
    
    print('\n✓ Pipeline ready for ML model training!')
    print('\nNext steps:')
    print('  1. Review generated datasets')
    print('  2. (Optional) Run: python scripts/verify_pipeline.py')
    print('  3. Train model: python src/ml/train_risk_model.py --data data/neodataset_augmented_3class.csv')
    print('  4. Start API: python app.py')


if __name__ == '__main__':
    main()
