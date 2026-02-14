# Streamlit ML Classification App

This project demonstrates a Streamlit-based web application for evaluating
multiple machine learning classification models.

## Features (Planned)
- CSV upload for test data
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix / classification report

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- XGBoost

| **ML Model Name** | **Observation about Model Performance** |
|------------------|------------------------------------------|
| **Logistic Regression** | Served as a strong baseline model with stable and interpretable results. Provided decent accuracy but was limited in capturing complex non-linear patterns compared to ensemble models. |
| **Decision Tree** | Achieved good training performance but showed signs of overfitting. The model was sensitive to data variations and tree depth, resulting in reduced generalization on unseen data. |
| **k-Nearest Neighbors (kNN)** | Performance depended heavily on the choice of *k*. Worked reasonably well for smaller datasets but showed scalability issues and sensitivity to feature scaling and noise. |
| **Naive Bayes (Gaussian)** | Fast and computationally efficient. Performed reasonably well when feature independence assumptions held, but overall accuracy was lower due to strong simplifying assumptions. |
| **Random Forest (Ensemble)** | Significantly improved performance over individual models. Reduced overfitting by aggregating multiple decision trees and achieved higher accuracy and more stable predictions. |
| **XGBoost (Ensemble)** | Delivered the best overall performance. Effectively captured complex feature interactions using gradient boosting and achieved superior accuracy and recall with proper tuning. |