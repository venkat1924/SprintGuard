"""
NeoDataset Loader and Preprocessor
Downloads and prepares NeoDataset from HuggingFace for weak supervision
"""
from datasets import load_dataset
import pandas as pd


def load_neodataset(cache_dir='./data/neodataset'):
    """
    Load NeoDataset from HuggingFace
    
    Returns:
        pd.DataFrame with columns: title, description, weight (story_points),
        project_id, state, created, etc.
    """
    print("Downloading NeoDataset from HuggingFace...")
    dataset = load_dataset("giseldo/neodataset", cache_dir=cache_dir)
    df = dataset['issues'].to_pandas()
    
    print(f"✓ Loaded {len(df)} stories")
    return df


def explore_dataset(df):
    """Generate EDA report"""
    print("\n=== NeoDataset Exploration ===")
    print(f"Total stories: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nStory points distribution:")
    print(df['weight'].value_counts().sort_index())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nSample story:")
    print(df[['title', 'description', 'weight']].iloc[0])


def preprocess_neodataset(df):
    """
    Clean and prepare dataset for labeling
    """
    # Filter valid stories
    df = df[df['weight'].notna()].copy()
    df = df[df['description'].notna()].copy()
    df = df[df['title'].notna()].copy()
    
    # Rename for consistency
    df = df.rename(columns={'weight': 'story_points'})
    
    # Combine title + description for text analysis
    df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    
    # Add text statistics
    df['word_count'] = df['description'].str.split().str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    df['char_count'] = df['description'].str.len()
    
    # Check for list structures (bullets)
    df['has_list'] = df['description'].str.contains(r'[-*•]\s', regex=True, na=False)
    df['list_item_count'] = df['description'].str.count(r'[-*•]\s', na=0)
    
    # Check for code blocks
    df['has_code_block'] = df['description'].str.contains('```', regex=False, na=False)
    
    print(f"\n✓ Preprocessed {len(df)} valid stories")
    return df

