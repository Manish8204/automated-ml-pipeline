import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from pathlib import Path

def train_models():
    """Train multiple ML models and save the best one"""
    
    print("Loading preprocessed data...")
    X = pd.read_csv('data/processed/features.csv')
    y = pd.read_csv('data/processed/target.csv').values.ravel()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}\n")
    
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=200, random_state=42),
        'svm': SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        results[name] = score
        print(f"  Accuracy: {score:.4f}\n")
    
    # Save best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    
    Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, f'models/{best_model_name}.pkl')
    
    print(f"✓ Best model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")
    print(f"✓ Model saved to models/{best_model_name}.pkl")

if __name__ == "__main__":
    train_models()
