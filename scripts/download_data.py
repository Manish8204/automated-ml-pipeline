import os
import pandas as pd
from pathlib import Path

# Create data directory if it doesn't exist
Path("data/raw").mkdir(parents=True, exist_ok=True)

print("Downloading dataset...")

# Using Iris dataset as default for demo
# Can be replaced with any Kaggle dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save raw data
df.to_csv('data/raw/iris.csv', index=False)
print(f"✓ Dataset saved to data/raw/iris.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
