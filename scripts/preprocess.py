import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

def load_and_preprocess():
    """Load raw data and apply preprocessing"""
    
    df = pd.read_csv('data/raw/iris.csv')
    print(f"Loaded data shape: {df.shape}")
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print(f"Found missing values:\n{df.isnull().sum()}")
        df = df.dropna()
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Preprocessed data shape: {X_scaled.shape}")
    
    # Save processed data
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    X_scaled.to_csv('data/processed/features.csv', index=False)
    y.to_csv('data/processed/target.csv', index=False)
    
    print("✓ Data preprocessing completed")
    return X_scaled, y

if __name__ == "__main__":
    load_and_preprocess()
