import joblib

def load_model():
    return joblib.load("saved_models/random_forest.pkl")