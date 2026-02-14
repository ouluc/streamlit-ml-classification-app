import joblib

def load_model():
    model = joblib.load("saved_models/knn.pkl")
    scaler = joblib.load("saved_models/knn_scaler.pkl")
    return model, scaler