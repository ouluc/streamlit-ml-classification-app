# Streamlit Machine Learning Classification App

This project demonstrates a Streamlit-based web application for evaluating multiple machine learning classification models.

## Problem Statement

- **Objective**: Binary classification for predicting the quality of red wine.
- **Positive Class (1)**: Wine with quality score **≥ 6** (Good Quality Wine).
- **Negative Class (0)**: Wine with quality score **< 6** (Not So Good Quality Wine).

---

## Dataset Details

- **Dataset Name**: UCI *Wine Quality – Red Wine* Dataset
- **Source**: UCI Machine Learning Repository
- **Total Instances**: 1,599 samples
- **Feature Size**: 12  
  - **Input Features**: 11  
  - **Target Variable**: 1 (Binary quality label)

**Models Used:** Below six classification models are implemented:
1. Logistic Regression 
2. Decision Tree Classifier 
3. K-Nearest Neighbor Classifier 
4. Naive Bayes Classifier - Gaussian or Multinomial 
5. Ensemble Model - Random Forest 
6. Ensemble Model - XGBoost 

## Comparison Table with the evaluation metrics calculated for all the 6 models:

| **ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1 Score** | **MCC** |
|------------------|-------------|---------|---------------|------------|--------------|---------|
| **Logistic Regression** | 0.8938 | 0.8804 | 0.6957 | 0.3721 | 0.4848 | 0.4580 |
| **Decision Tree Classifier** | 0.6844 | 0.8361 | 0.2958 | 0.9767 | 0.4541 | 0.4227 |
| **k-Nearest Neighbors (kNN)** | 0.6250 | 0.7838 | 0.2609 | 0.9767 | 0.4118 | 0.3732 |
| **Naive Bayes (Gaussian)** | 0.6531 | 0.8520 | 0.2703 | 0.9302 | 0.4188 | 0.3696 |
| **Random Forest (Ensemble)** | 0.6500 | 0.9314 | 0.2745 | 0.9767 | 0.4286 | 0.3933 |
| **XGBoost (Ensemble)** | 0.6312 | 0.9354 | 0.2642 | 0.9767 | 0.4158 | 0.3781 |

##  Observations on the performance of each model

| **ML Model Name** | **Observation about Model Performance** |
|------------------|------------------------------------------|
| **Logistic Regression** | Achieved the highest accuracy and best balance between precision and recall among all models. It shows strong overall classification ability with comparatively fewer false positives and false negatives, making it the most stable and reliable baseline model. |
| **Decision Tree Classifier** | Demonstrates extremely high recall, correctly identifying most positive cases, but suffers from very low precision and accuracy. This indicates overfitting and a high number of false positives. |
| **k-Nearest Neighbors (kNN)** | Shows high recall but poor precision and accuracy, suggesting that it classifies most samples as positive. It does not generalize well on this dataset. |
| **Naive Bayes (Gaussian)** | Performs moderately with high recall but low precision. Many false positives despite acceptable AUC performance. |
| **Random Forest (Ensemble)** | Provides excellent AUC score, indicating strong ranking capability. However, low precision and accuracy suggest class imbalance bias. |
| **XGBoost (Ensemble)** | Achieves the highest AUC among all models, reflecting superior probability estimation and discrimination power. However, low precision and accuracy indicate class imbalance. |

## Overview of Steamlit app
- Upload test data (CSV file). You may download the sample data file from folder "sample_data".
- Select ML model of your choice from the drop down. Above listed all 6 models will appear.
- Refer to evaluation matric, Confusion matrix and Classification report for above selected model.
