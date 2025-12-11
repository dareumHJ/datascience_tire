"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np


def load_data(train_path, test_path):
    """
    Load train and test datasets
    
    Args:
        train_path: Path to train CSV file
        test_path: Path to test CSV file
        
    Returns:
        train_df, test_df, train_ids, test_ids, y
    """
    print("\n[STEP 1] LOADING DATA")
    print("-" * 70)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   Train: {train_df.shape}")
    print(f"   Test:  {test_df.shape}")
    
    # Extract target (Class column in train.csv)
    if 'Class' in train_df.columns:
        # Convert 'Good'/'NG' to 0/1
        y_raw = train_df['Class'].values
        y = np.where(y_raw == 'Good', 0, 1)  # Good=0, NG=1
        train_df = train_df.drop(columns=['Class'])
        print(f"   Converted Class: 'Good'→0, 'NG'→1")
    elif 'target' in train_df.columns:
        y = train_df['target'].values
        train_df = train_df.drop(columns=['target'])
    else:
        raise ValueError("Target column not found (expected 'Class' or 'target')")
    
    # Generate train IDs if not present
    if 'ID' in train_df.columns:
        train_ids = train_df['ID'].values
        train_df = train_df.drop(columns=['ID'])
    else:
        train_ids = np.array([f'TRAIN_{i}' for i in range(len(train_df))])
        print("   ⚠️  No ID column in train.csv, generated IDs")
    
    # Extract test IDs
    if 'ID' in test_df.columns:
        test_ids = test_df['ID'].values
        test_df = test_df.drop(columns=['ID'])
    else:
        test_ids = np.array([f'TEST_{i}' for i in range(len(test_df))])
        print("   ⚠️  No ID column in test.csv, generated IDs")
    
    print(f"   Target distribution: Good={np.sum(y==0)} / Bad={np.sum(y==1)}")
    
    return train_df, test_df, train_ids, test_ids, y


def preprocess_data(train_df, test_df):
    """
    Basic preprocessing: handle missing values and align columns
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        
    Returns:
        Preprocessed train_df, test_df
    """
    # Fill missing values
    train_df = train_df.fillna(-999)
    test_df = test_df.fillna(-999)
    
    # Align columns
    common_cols = train_df.columns.intersection(test_df.columns)
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]
    
    return train_df, test_df