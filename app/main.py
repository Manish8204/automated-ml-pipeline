from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(
    title="ML Pipeline API",
    description="Automated ML Pipeline for Iris Classification",
    version="1.0.0"
)

# Load model
try:
    model_path = Path('models/random_forest.pkl')
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        model = None
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

class PredictionRequest(BaseModel):
    """Input features for prediction"""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    """Prediction output"""
    prediction: int
    confidence: float

@app.get("/")
def read_root():
    return {
        "message": "Welcome to ML Pipeline API",
        "docs": "/docs",
        "status": "Model loaded" if model is not None else "Model not found"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make predictions using the trained model"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Prepare input
        features = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get confidence (probability)
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "loaded" if model else "not_loaded"}
