# src/model_evaluation.py
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from utils import load_params, logger
import os

def main():
    params = load_params()['model_evaluation']['metrics']
    
    model = pickle.load(open("models/model.pkl", "rb"))
    test_df = pd.read_csv("data/features/test_bow.csv")

    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics_dict = {}
    if "accuracy" in params:
        metrics_dict["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
    if "precision" in params:
        metrics_dict["precision"] = round(precision_score(y_test, y_pred), 4)
    if "recall" in params:
        metrics_dict["recall"] = round(recall_score(y_test, y_pred), 4)
    if "auc" in params:
        metrics_dict["auc"] = round(roc_auc_score(y_test, y_proba), 4)

    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    logger.info(f"Evaluation complete: {metrics_dict}")

if __name__ == "__main__":
    main()