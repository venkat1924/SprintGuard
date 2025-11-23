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
    print("\n" + "="*70)
    print("[STAGE 1] Loading NeoDataset from HuggingFace")
    print("="*70)
    print(f"Cache directory: {cache_dir}")
    print("Dataset: giseldo/neodataset")
    
    try:
        print("\n[DOWNLOAD] Fetching dataset from HuggingFace Hub...")
        dataset = load_dataset("giseldo/neodataset", cache_dir=cache_dir)
        print("  ✓ Download complete")
        
        print("\n[CONVERT] Converting to pandas DataFrame...")
        # Try 'train' first (newer datasets), fall back to 'issues' (older)
        if 'train' in dataset:
            df = dataset['train'].to_pandas()
        elif 'issues' in dataset:
            df = dataset['issues'].to_pandas()
        else:
            raise KeyError(f"Dataset has unexpected structure. Available keys: {list(dataset.keys())}")
        print(f"  ✓ Converted {len(df)} stories")
        
        # Log dataset statistics
        print("\n[DATASET INFO] Raw dataset statistics:")
        print(f"  Total stories: {len(df)}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Column names: {df.columns.tolist()}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Log missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n[DATASET INFO] Missing values detected:")
            for col, count in missing[missing > 0].items():
                print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"\n[DATASET INFO] No missing values detected")
        
        print(f"\n✓ Dataset loaded successfully")
        return df
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load NeoDataset: {str(e)}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        raise


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
    print("\n" + "="*70)
    print("[PREPROCESSING] Cleaning and feature engineering")
    print("="*70)
    
    original_count = len(df)
    print(f"Starting with {original_count} stories")
    
    # Filter valid stories
    print("\n[FILTER] Removing stories with missing data...")
    before_filter = len(df)
    
    # Check which column name is used for story points
    sp_col = None
    if 'storypoints' in df.columns:
        sp_col = 'storypoints'
    elif 'weight' in df.columns:
        sp_col = 'weight'
    else:
        print(f"  ⚠ Warning: No story points column found. Available columns: {df.columns.tolist()}")
        print(f"  Continuing without story points filtering...")
    
    if sp_col:
        df = df[df[sp_col].notna()].copy()
        print(f"  After removing missing {sp_col}: {len(df)} stories ({before_filter - len(df)} removed)")
        before_filter = len(df)
    
    df = df[df['description'].notna()].copy()
    print(f"  After removing missing description: {len(df)} stories ({before_filter - len(df)} removed)")
    before_filter = len(df)
    
    df = df[df['title'].notna()].copy()
    print(f"  After removing missing title: {len(df)} stories ({before_filter - len(df)} removed)")
    
    valid_count = len(df)
    filtered_count = original_count - valid_count
    print(f"  ✓ Filtered out {filtered_count} invalid stories ({filtered_count/original_count*100:.1f}%)")
    print(f"  ✓ {valid_count} valid stories remaining")
    
    # Rename for consistency
    print("\n[RENAME] Renaming columns...")
    if sp_col and sp_col != 'story_points':
        df = df.rename(columns={sp_col: 'story_points'})
        print(f"  ✓ Renamed '{sp_col}' → 'story_points'")
    else:
        print(f"  ℹ No renaming needed for story_points column")
    
    # Combine title + description for text analysis
    print("\n[FEATURE] Creating computed columns...")
    df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    print(f"  ✓ Created 'full_text' (title + description)")
    
    # Add text statistics
    df['word_count'] = df['description'].str.split().str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    df['char_count'] = df['description'].str.len()
    print(f"  ✓ Created text statistics (word_count, title_word_count, char_count)")
    
    # Log statistics
    print(f"    - word_count: mean={df['word_count'].mean():.1f}, median={df['word_count'].median():.0f}")
    print(f"    - char_count: mean={df['char_count'].mean():.1f}, median={df['char_count'].median():.0f}")
    
    # Check for list structures (bullets)
    df['has_list'] = df['description'].str.contains(r'[-*•]\s', regex=True, na=False)
    df['list_item_count'] = df['description'].str.count(r'[-*•]\s').fillna(0).astype(int)
    list_count = df['has_list'].sum()
    print(f"  ✓ Detected lists in {list_count} stories ({list_count/len(df)*100:.1f}%)")
    
    # Check for code blocks
    df['has_code_block'] = df['description'].str.contains('```', regex=False, na=False)
    code_count = df['has_code_block'].sum()
    print(f"  ✓ Detected code blocks in {code_count} stories ({code_count/len(df)*100:.1f}%)")
    
    # Validate output schema
    print("\n[VALIDATION] Validating output schema...")
    expected_cols = ['title', 'description', 'story_points', 'full_text', 'word_count', 
                     'char_count', 'has_list', 'list_item_count', 'has_code_block']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"  ✗ ERROR: Missing expected columns: {missing_cols}")
        raise ValueError(f"Preprocessing failed: missing columns {missing_cols}")
    print(f"  ✓ All expected columns present: {expected_cols}")
    
    print(f"\n✓ Preprocessing complete: {len(df)} stories ready for labeling")
    print(f"✓ Output shape: {df.shape}")
    
    return df

