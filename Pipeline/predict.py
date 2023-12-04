from joblib import load
import pandas as pd

def load_models():
    base_model = load('Models/base_model_pipeline.joblib')
    dynamic_model = load('Models/dynamic_model_pipeline.joblib')
    return base_model, dynamic_model

def predict_prices(new_data, base_model, dynamic_model):
    base_predictions = base_model.predict(new_data)
    dynamic_predictions = dynamic_model.predict(new_data)
    return base_predictions, dynamic_predictions
