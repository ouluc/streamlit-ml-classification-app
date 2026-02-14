import joblib

def load_model():
    return joblib.load("saved_models/logistic_regression.pkl")