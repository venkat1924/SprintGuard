#!/usr/bin/env python3
"""
Post-Augmentation Label Mapping Script
Converts binary labels (SAFE/RISK) to 3-class labels (Low/Medium/High)

Usage:
    python scripts/map_to_3class.py [--input INPUT_CSV] [--output OUTPUT_CSV]
"""
import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, '.')


def validate_input_data(df):
    """
    Validate that input data has required columns and correct format.
    
    Args:
        df: Input DataFrame
        
    Raises:
        ValueError: If validation fails
    """
    print('\n[VALIDATION] Validating input data schema...')
    
    required_columns = ['risk_label', 'risk_confidence']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"[ERROR] Missing required columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Check risk_label values
    unique_labels = df['risk_label'].unique()
    expected_labels = {'SAFE', 'RISK'}
    
    if not set(unique_labels).issubset(expected_labels):
        raise ValueError(
            f"[ERROR] Unexpected risk_label values: {unique_labels}\n"
            f"Expected only: {expected_labels}"
        )
    
    # Check risk_confidence range
    min_conf = df['risk_confidence'].min()
    max_conf = df['risk_confidence'].max()
    
    if min_conf < 0 or max_conf > 1:
        raise ValueError(
            f"[ERROR] risk_confidence out of range [0, 1]: min={min_conf}, max={max_conf}"
        )
    
    print(f'  ✓ Schema validation passed')
    print(f'  ✓ Found {len(df)} stories')
    print(f'  ✓ risk_label values: {unique_labels}')
    print(f'  ✓ risk_confidence range: [{min_conf:.3f}, {max_conf:.3f}]')


def map_to_3class(df, confidence_threshold=0.85):
    """
    Map binary labels to 3-class labels based on confidence.
    
    Mapping strategy:
    - SAFE → Low (regardless of confidence)
    - RISK + high confidence (>threshold) → High
    - RISK + low confidence (≤threshold) → Medium
    
    Args:
        df: Input DataFrame with risk_label and risk_confidence
        confidence_threshold: Threshold for High vs Medium RISK
        
    Returns:
        DataFrame with updated risk_label column
    """
    print(f'\n[MAPPING] Converting binary labels to 3-class labels...')
    print(f'  Strategy:')
    print(f'    - SAFE → Low')
    print(f'    - RISK (confidence > {confidence_threshold}) → High')
    print(f'    - RISK (confidence ≤ {confidence_threshold}) → Medium')
    
    df_mapped = df.copy()
    
    # Count before mapping
    label_counts_before = df['risk_label'].value_counts()
    print(f'\n  Labels before mapping:')
    for label, count in label_counts_before.items():
        print(f'    {label}: {count} ({count/len(df)*100:.1f}%)')
    
    def map_label(row):
        if row['risk_label'] == 'SAFE':
            return 'Low'
        elif row['risk_label'] == 'RISK':
            if row['risk_confidence'] > confidence_threshold:
                return 'High'
            else:
                return 'Medium'
        else:
            # Should never reach here if validation passed
            return 'Medium'
    
    df_mapped['risk_label'] = df_mapped.apply(map_label, axis=1)
    
    # Count after mapping
    label_counts_after = df_mapped['risk_label'].value_counts()
    print(f'\n  Labels after mapping:')
    for label in ['Low', 'Medium', 'High']:
        count = label_counts_after.get(label, 0)
        print(f'    {label}: {count} ({count/len(df_mapped)*100:.1f}%)')
    
    print(f'\n  ✓ Mapping complete')
    
    return df_mapped


def validate_output_data(df):
    """
    Validate that output data has correct 3-class labels.
    
    Args:
        df: Output DataFrame
        
    Raises:
        ValueError: If validation fails
    """
    print('\n[VALIDATION] Validating output data...')
    
    unique_labels = df['risk_label'].unique()
    expected_labels = {'Low', 'Medium', 'High'}
    
    if not set(unique_labels).issubset(expected_labels):
        unexpected = set(unique_labels) - expected_labels
        raise ValueError(
            f"[ERROR] Unexpected risk_label values after mapping: {unexpected}\n"
            f"Expected only: {expected_labels}"
        )
    
    # Check that all three classes are present
    if set(unique_labels) != expected_labels:
        missing = expected_labels - set(unique_labels)
        print(f'  ⚠ Warning: Missing label classes: {missing}')
    
    print(f'  ✓ Output validation passed')
    print(f'  ✓ risk_label values: {sorted(unique_labels)}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert binary labels (SAFE/RISK) to 3-class labels (Low/Medium/High)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/neodataset_augmented.csv',
        help='Input CSV file with binary labels (default: data/neodataset_augmented.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/neodataset_augmented_3class.csv',
        help='Output CSV file with 3-class labels (default: data/neodataset_augmented_3class.csv)'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.85,
        help='Confidence threshold for High vs Medium RISK (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    print('='*70)
    print('Binary to 3-Class Label Mapping')
    print('='*70)
    print(f'Input:  {args.input}')
    print(f'Output: {args.output}')
    print(f'Confidence threshold: {args.confidence_threshold}')
    
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f'\n[ERROR] Input file not found: {args.input}')
        print('Please run augmentation pipeline first:')
        print('  python scripts/augment_neodataset.py')
        sys.exit(1)
    
    # Load data
    print(f'\n[LOADING] Reading {args.input}...')
    try:
        df = pd.read_csv(args.input)
        print(f'  ✓ Loaded {len(df)} stories')
        print(f'  ✓ Columns: {len(df.columns)}')
    except Exception as e:
        print(f'[ERROR] Failed to load input file: {e}')
        sys.exit(1)
    
    # Validate input
    try:
        validate_input_data(df)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    
    # Map labels
    try:
        df_mapped = map_to_3class(df, confidence_threshold=args.confidence_threshold)
    except Exception as e:
        print(f'[ERROR] Label mapping failed: {e}')
        sys.exit(1)
    
    # Validate output
    try:
        validate_output_data(df_mapped)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    
    # Save output
    print(f'\n[SAVING] Writing to {args.output}...')
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_mapped.to_csv(args.output, index=False)
        print(f'  ✓ Saved {len(df_mapped)} stories')
        print(f'  ✓ File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB')
    except Exception as e:
        print(f'[ERROR] Failed to save output file: {e}')
        sys.exit(1)
    
    # Also create high-confidence subset
    high_conf_input = args.input.replace('.csv', '_high_confidence.csv')
    high_conf_output = args.output.replace('.csv', '_high_confidence.csv')
    
    if Path(high_conf_input).exists():
        print(f'\n[BONUS] Processing high-confidence subset...')
        print(f'  Input:  {high_conf_input}')
        print(f'  Output: {high_conf_output}')
        
        try:
            df_high = pd.read_csv(high_conf_input)
            print(f'  ✓ Loaded {len(df_high)} high-confidence stories')
            
            validate_input_data(df_high)
            df_high_mapped = map_to_3class(df_high, confidence_threshold=args.confidence_threshold)
            validate_output_data(df_high_mapped)
            
            df_high_mapped.to_csv(high_conf_output, index=False)
            print(f'  ✓ Saved to {high_conf_output}')
        except Exception as e:
            print(f'  ⚠ Warning: Failed to process high-confidence subset: {e}')
    
    print('\n' + '='*70)
    print('✓ MAPPING COMPLETE')
    print('='*70)
    print(f'Output files ready:')
    print(f'  - {args.output}')
    if Path(high_conf_output).exists():
        print(f'  - {high_conf_output}')
    print('\nNext steps:')
    print('  1. Verify the mapped labels look correct')
    print('  2. Train ML model: python src/ml/train_risk_model.py --data', args.output)


if __name__ == '__main__':
    main()

