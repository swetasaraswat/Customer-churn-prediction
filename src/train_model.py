"""
train_model.py
---------------
Standalone script version of the modeling pipeline in
notebooks/churn_analysis.ipynb — trains and evaluates all three
classifiers and prints a comparison table. Useful for CI, quick
re-runs, or environments without Jupyter.

Run:
    python src/generate_data.py      # if data/customer_churn.csv doesn't exist yet
    python src/train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

RANDOM_STATE = 42


def load_and_prepare(path="data/customer_churn.csv"):
    df = pd.read_csv(path)
    model_df = df.drop(columns=["customerID"]).copy()
    model_df["Churn"] = (model_df["Churn"] == "Yes").astype(int)

    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        model_df[col] = (model_df[col] == "Yes").astype(int)
    model_df["gender"] = (model_df["gender"] == "Male").astype(int)

    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    model_df = pd.get_dummies(model_df, columns=multi_cat_cols, drop_first=True)

    X = model_df.drop(columns=["Churn"])
    y = model_df["Churn"]
    return X, y


def main():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    sample_weight_gb = np.where(
        y_train == 1, (y_train == 0).sum() / (y_train == 1).sum(), 1.0
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    rows = []
    for name, model in models.items():
        if name == "Gradient Boosting":
            model.fit(X_train_scaled, y_train, sample_weight=sample_weight_gb)
        else:
            model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 3),
            "Precision": round(precision_score(y_test, preds), 3),
            "Recall": round(recall_score(y_test, preds), 3),
            "F1": round(f1_score(y_test, preds), 3),
            "ROC-AUC": round(roc_auc_score(y_test, probs), 3),
        })

    results_df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    print("\nModel comparison (sorted by F1):\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
