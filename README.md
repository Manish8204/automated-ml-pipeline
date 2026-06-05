# Automated ML Pipeline

A complete end-to-end machine learning pipeline with data preprocessing, model training, and API deployment using FastAPI.

## Features
- Automated data preprocessing and feature engineering
- Multiple ML models for comparison
- REST API for real-time predictions
- Easy model retraining and versioning

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Manish8204/automated-ml-pipeline.git
cd automated-ml-pipeline
pip install -r requirements.txt
```

### Quick Start

1. Download dataset:
```bash
python scripts/download_data.py
```

2. Train model:
```bash
python train.py
```

3. Run API server:
```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for API documentation.

## Project Structure

```
automated-ml-pipeline/
├── data/
├── models/
├── app/
├── scripts/
└── notebooks/
```

## License
MIT
