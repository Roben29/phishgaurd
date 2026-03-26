import joblib

import joblib
from pathlib import Path

def load_model():
    base_path = Path(__file__).resolve().parent.parent  
    model_path = base_path / "output" / "model.pkl"
    scaler_path = base_path / "output" / "scaler.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler