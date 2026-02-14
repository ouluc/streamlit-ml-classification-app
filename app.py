import streamlit as st
import pandas as pd
import os
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from utils.preprocessing import preprocess_data
from utils.metrics import calculate_metrics

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="ML Classification App",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS (Safe for Streamlit Cloud)
# ---------------------------------------------------
st.markdown(
    """
    <style>
    .card {
        background-color: #f9f9f9;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# App Header
# ---------------------------------------------------
st.title("ML Classification Model Evaluation")
st.caption("Upload test data, select a trained model, and evaluate performance")

st.markdown("---")

# ---------------------------------------------------
# STEP 1: Dataset Upload
# ---------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Step 1/1: Upload Test Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload CSV file (test data only)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

data = pd.read_csv(uploaded_file)

st.success("Dataset uploaded successfully!")
st.write("Preview of uploaded data:")
st.dataframe(data.head(), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# STEP 2: Model Selection
# ---------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Step 2/2: Select Classification Model</div>', unsafe_allow_html=True)

MODELS_DIR = "models"

model_files = [
    f.replace(".py", "")
    for f in os.listdir(MODELS_DIR)
    if f.endswith(".py") and f != "__init__.py"
]

selected_model_name = st.selectbox(
    "Choose a trained model",
    model_files
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
model_module = importlib.import_module(f"models.{selected_model_name}")
model = model_module.load_model()

# ---------------------------------------------------
# Preprocess & Predict
# ---------------------------------------------------
X_test, y_test = preprocess_data(data)
y_pred = model.predict(X_test)

# ---------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------
metrics = calculate_metrics(y_test, y_pred, model, X_test)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Evaluation Metrics</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
m1.metric("Precision", f"{metrics['precision']:.4f}")

m2.metric("Recall", f"{metrics['recall']:.4f}")
m2.metric("F1 Score", f"{metrics['f1']:.4f}")

m3.metric("AUC Score", f"{metrics['auc']:.4f}")
m3.metric("MCC Score", f"{metrics['mcc']:.4f}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Performance Analysis
# ---------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Performance Analysis</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.2])

# Confusion Matrix
with left:
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    labels = ["Bad", "Good"]

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

# Classification Report
with right:
    st.subheader("Classification Report")

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    df_report = pd.DataFrame(report_dict).transpose().round(3)
    st.dataframe(df_report, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)