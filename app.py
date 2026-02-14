import streamlit as st
import pandas as pd
import os
import importlib
from sklearn.metrics import confusion_matrix, classification_report

from utils.preprocessing import preprocess_data
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix

# ---------------------------------------------------
# Streamlit App Config
# ---------------------------------------------------
st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("ML Classification Model Evaluation")

# ---------------------------------------------------
# Step 1: Dataset Upload
# ---------------------------------------------------
st.header("Step 1/2: Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (Test data only)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

data = pd.read_csv(uploaded_file)
st.success("Dataset uploaded successfully!")
st.write("Preview of uploaded data:")
st.dataframe(data.head())

# ---------------------------------------------------
# Step 2: Model Selection (Auto from models folder)
# ---------------------------------------------------
st.header("Step 2/2: Select Machine Learning Model")

MODELS_DIR = "models"

model_files = [
    f.replace(".py", "")
    for f in os.listdir(MODELS_DIR)
    if f.endswith(".py") and f != "__init__.py"
]

selected_model_name = st.selectbox(
    "Choose a classification model",
    model_files
)

# ---------------------------------------------------
# Step 3: Load Selected Model Dynamically
# ---------------------------------------------------
model_module = importlib.import_module(f"models.{selected_model_name}")

model = model_module.load_model()

# ---------------------------------------------------
# Step 4: Preprocess Data
# ---------------------------------------------------
X_test, y_test = preprocess_data(data)

# ---------------------------------------------------
# Step 5: Predictions
# ---------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------
# Step 6: Evaluation Metrics
# ---------------------------------------------------
st.header("Evaluation Metrics")

metrics = calculate_metrics(y_test, y_pred, model, X_test)

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
col1.metric("Precision", f"{metrics['precision']:.4f}")

col2.metric("Recall", f"{metrics['recall']:.4f}")
col2.metric("F1 Score", f"{metrics['f1']:.4f}")

col3.metric("AUC Score", f"{metrics['auc']:.4f}")
col3.metric("MCC Score", f"{metrics['mcc']:.4f}")

# ---------------------------------------------------
# Step 7: Confusion Matrix / Classification Report
# ---------------------------------------------------
st.header("Model Performance Analysis")

cm = confusion_matrix(y_test, y_pred)

st.subheader("Confusion Matrix")
plot_confusion_matrix(cm)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))