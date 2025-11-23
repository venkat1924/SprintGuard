#!/usr/bin/env python3
"""
Pipeline Validation Utility
Checks data compatibility between all pipeline stages

Usage:
    python scripts/verify_pipeline.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, '.')


def check_file_exists(path, description):
    """Check if a file exists and return True/False with logging."""
    if Path(path).exists():
        print(f'  ✓ {description}: {path}')
        return True
    else:
        print(f'  ✗ {description}: {path} [NOT FOUND]')
        return False


def validate_stage1_output():
    """Validate Stage 1 output (preprocessed NeoDataset)."""
    print('\n[STAGE 1] Validating preprocessed data...')
    print('Expected: In-memory DataFrame after neodataset_loader.preprocess_neodataset()')
    print('Cannot validate Stage 1 directly (in-memory only)')
    print('  ℹ Stage 1 validation occurs in Stage 2')
    return True


def validate_stage2_output():
    """Validate Stage 2 output (Snorkel labels)."""
    print('\n[STAGE 2] Validating Snorkel labeled data...')
    
    path = 'data/neodataset/neodataset_snorkel_labels.csv'
    if not check_file_exists(path, 'Snorkel output'):
        print('  ⚠ Run: python scripts/augment_neodataset.py')
        return False
    
    try:
        df = pd.read_csv(path)
        print(f'  ✓ Loaded {len(df)} stories')
        
        # Check required columns from Stage 1
        stage1_cols = ['full_text', 'word_count', 'story_points', 'title', 'description']
        missing = [col for col in stage1_cols if col not in df.columns]
        if missing:
            print(f'  ✗ Missing Stage 1 columns: {missing}')
            return False
        print(f'  ✓ Stage 1 columns present')
        
        # Check Stage 2 specific columns
        stage2_cols = ['risk_label', 'risk_label_binary', 'risk_confidence', 'risk_prob_safe', 'risk_prob_risk']
        missing = [col for col in stage2_cols if col not in df.columns]
        if missing:
            print(f'  ✗ Missing Stage 2 columns: {missing}')
            return False
        print(f'  ✓ Stage 2 columns present')
        
        # Validate risk_label values
        unique_labels = df['risk_label'].unique()
        if not set(unique_labels).issubset({'SAFE', 'RISK'}):
            print(f'  ✗ Invalid risk_label values: {unique_labels}')
            return False
        print(f'  ✓ risk_label values: {unique_labels}')
        
        # Validate risk_confidence range
        if df['risk_confidence'].min() < 0 or df['risk_confidence'].max() > 1:
            print(f'  ✗ risk_confidence out of range [0, 1]')
            return False
        print(f'  ✓ risk_confidence range: [{df["risk_confidence"].min():.3f}, {df["risk_confidence"].max():.3f}]')
        
        print(f'  ✓ Stage 2 output valid')
        return True
        
    except Exception as e:
        print(f'  ✗ Validation failed: {e}')
        return False


def validate_stage3_output():
    """Validate Stage 3 output (Cleanlab filtered data)."""
    print('\n[STAGE 3] Validating Cleanlab filtered data...')
    
    paths = [
        'data/neodataset_augmented.csv',
        'data/neodataset_augmented_high_confidence.csv'
    ]
    
    all_valid = True
    for path in paths:
        if not check_file_exists(path, Path(path).stem):
            print(f'  ⚠ Run: python scripts/augment_neodataset.py')
            all_valid = False
            continue
        
        try:
            df = pd.read_csv(path)
            print(f'    ✓ Loaded {len(df)} stories')
            
            # Check all previous stage columns
            required_cols = ['full_text', 'story_points', 'risk_label', 'risk_confidence']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f'    ✗ Missing required columns: {missing}')
                all_valid = False
                continue
            
            # Validate risk_label values (still binary at this stage)
            unique_labels = df['risk_label'].unique()
            if not set(unique_labels).issubset({'SAFE', 'RISK'}):
                print(f'    ✗ Invalid risk_label values: {unique_labels}')
                all_valid = False
                continue
            
            print(f'    ✓ Schema valid')
            
        except Exception as e:
            print(f'    ✗ Validation failed: {e}')
            all_valid = False
    
    if all_valid:
        print(f'  ✓ Stage 3 output valid')
    return all_valid


def validate_mapping_output():
    """Validate label mapping output."""
    print('\n[MAPPING] Validating 3-class mapped data...')
    
    paths = [
        'data/neodataset_augmented_3class.csv',
        'data/neodataset_augmented_3class_high_confidence.csv'
    ]
    
    all_valid = True
    for path in paths:
        if not check_file_exists(path, Path(path).stem):
            print(f'  ⚠ Run: python scripts/map_to_3class.py')
            all_valid = False
            continue
        
        try:
            df = pd.read_csv(path)
            print(f'    ✓ Loaded {len(df)} stories')
            
            # Check risk_label values (should be 3-class now)
            unique_labels = df['risk_label'].unique()
            expected = {'Low', 'Medium', 'High'}
            if not set(unique_labels).issubset(expected):
                print(f'    ✗ Invalid risk_label values: {unique_labels}')
                print(f'    Expected: {expected}')
                all_valid = False
                continue
            
            # Check distribution
            counts = df['risk_label'].value_counts()
            print(f'    ✓ Label distribution:')
            for label in ['Low', 'Medium', 'High']:
                count = counts.get(label, 0)
                print(f'      {label}: {count} ({count/len(df)*100:.1f}%)')
            
        except Exception as e:
            print(f'    ✗ Validation failed: {e}')
            all_valid = False
    
    if all_valid:
        print(f'  ✓ Mapping output valid')
    return all_valid


def validate_stage4_output():
    """Validate Stage 4 output (trained model artifacts)."""
    print('\n[STAGE 4] Validating trained model artifacts...')
    
    model_files = [
        'models/xgboost_risk_model.json',
        'models/feature_scaler.pkl',
        'models/feature_names.json',
        'models/risk_lexicons.json'
    ]
    
    all_exist = True
    for path in model_files:
        if not check_file_exists(path, Path(path).name):
            all_exist = False
    
    if not all_exist:
        print(f'  ⚠ Run: python src/ml/train_risk_model.py --data data/neodataset_augmented_3class.csv')
        return False
    
    # Try to load model
    try:
        import json
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        print(f'  ✓ Model has {len(feature_names)} features')
        
        # Check feature composition
        symbolic_count = sum(1 for name in feature_names if not name.startswith('embedding_'))
        embedding_count = sum(1 for name in feature_names if name.startswith('embedding_'))
        print(f'    - Symbolic features: {symbolic_count}')
        print(f'    - Embedding features: {embedding_count}')
        
        print(f'  ✓ Stage 4 output valid')
        return True
        
    except Exception as e:
        print(f'  ✗ Model validation failed: {e}')
        return False


def validate_stage5_ready():
    """Validate Stage 5 readiness (Flask API can start)."""
    print('\n[STAGE 5] Validating API readiness...')
    
    # Check data file
    data_ready = check_file_exists('data/neodataset_augmented.csv', 'API data source')
    
    # Check model files
    model_ready = Path('models/xgboost_risk_model.json').exists()
    if model_ready:
        print(f'  ✓ Model artifacts present')
    else:
        print(f'  ✗ Model artifacts missing')
    
    if data_ready and model_ready:
        print(f'  ✓ Stage 5 ready to start')
        print(f'  Run: python app.py')
        return True
    else:
        print(f'  ✗ Stage 5 not ready')
        return False


def main():
    print('='*70)
    print('SprintGuard Pipeline Verification')
    print('='*70)
    print('Checking data compatibility between all stages...')
    
    results = {}
    
    # Validate each stage
    results['Stage 1'] = validate_stage1_output()
    results['Stage 2'] = validate_stage2_output()
    results['Stage 3'] = validate_stage3_output()
    results['Mapping'] = validate_mapping_output()
    results['Stage 4'] = validate_stage4_output()
    results['Stage 5'] = validate_stage5_ready()
    
    # Summary
    print('\n' + '='*70)
    print('VERIFICATION SUMMARY')
    print('='*70)
    
    for stage, valid in results.items():
        status = '✓' if valid else '✗'
        print(f'{status} {stage}: {"VALID" if valid else "INVALID/MISSING"}')
    
    all_valid = all(results.values())
    
    if all_valid:
        print('\n✓ All stages validated successfully!')
        print('Pipeline is ready to use.')
    else:
        print('\n✗ Some stages failed validation.')
        print('Please run the missing steps.')
    
    print('\nPipeline execution order:')
    print('  1. python scripts/augment_neodataset.py')
    print('  2. python scripts/map_to_3class.py')
    print('  3. python src/ml/train_risk_model.py --data data/neodataset_augmented_3class.csv')
    print('  4. python app.py')
    
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()

