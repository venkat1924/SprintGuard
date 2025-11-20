#!/usr/bin/env python3
"""
Explore NeoDataset Script
Downloads and performs EDA on NeoDataset
"""
import sys
sys.path.insert(0, '.')

from src.ml.neodataset_loader import load_neodataset, explore_dataset, preprocess_neodataset


def main():
    print('='*60)
    print('NeoDataset Exploration')
    print('='*60)
    
    # Load
    print('\nDownloading NeoDataset from HuggingFace...')
    df = load_neodataset()
    
    # Explore
    explore_dataset(df)
    
    # Preprocess
    print('\nPreprocessing dataset...')
    df_clean = preprocess_neodataset(df)
    
    # Additional statistics
    print('\n=== Text Statistics ===')
    print(f'Average description length: {df_clean["word_count"].mean():.1f} words')
    print(f'Median description length: {df_clean["word_count"].median():.1f} words')
    print(f'\nStories with lists: {df_clean["has_list"].sum()} ({df_clean["has_list"].mean()*100:.1f}%)')
    print(f'Stories with code blocks: {df_clean["has_code_block"].sum()} ({df_clean["has_code_block"].mean()*100:.1f}%)')
    
    # Save preprocessed version
    output_path = 'data/neodataset/neodataset_preprocessed.csv'
    df_clean.to_csv(output_path, index=False)
    print(f'\n✓ Saved preprocessed dataset to {output_path}')
    print(f'\nReady to run augmentation pipeline!')


if __name__ == '__main__':
    main()
