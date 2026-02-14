import joblib
import os

def load_model():
    model_path = os.path.join("saved_models", "logistic_regression.pkl")
    return joblib.load(model_path)