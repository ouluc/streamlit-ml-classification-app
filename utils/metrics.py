from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

def calculate_metrics(y_true, y_pred, model, X_test):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "auc": roc_auc_score(y_true, model.predict_proba(X_test), multi_class="ovr"),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }