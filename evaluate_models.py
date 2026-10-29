
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(BASE_DIR / "insurance.csv")

# Create classification target
median_charge = df["charges"].median()
df["high_charges"] = (df["charges"] >= median_charge).astype(int)

X = df.drop(["charges", "high_charges"], axis=1)
y = df["high_charges"]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Use the same test split for every model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = [
    ("Logistic Regression", "classification_model.pkl",
     "classification_columns.pkl", None),
    ("KNN", "knn_model.pkl", "knn_columns.pkl", "knn_scaler.pkl"),
    ("ANN", "ann_model.pkl", "ann_columns.pkl", "ann_scaler.pkl"),
    ("SVM", "svm_model.pkl", "svm_columns.pkl", "svm_scaler.pkl"),
    ("Decision Tree", "decision_tree_model.pkl",
     "decision_tree_columns.pkl", None),
    ("Naive Bayes", "naive_bayes_model.pkl",
     "naive_bayes_columns.pkl", None),
    ("Random Forest", "random_forest_model.pkl",
     "random_forest_columns.pkl", None),
]

results = []

for name, model_file, columns_file, scaler_file in models:
    try:
        model = joblib.load(BASE_DIR / model_file)
        columns = joblib.load(BASE_DIR / columns_file)

        # Match the features and their training order
        X_test_model = X_test.reindex(columns=columns, fill_value=0)

        # Apply scaling only to models that need it
        if scaler_file:
            scaler = joblib.load(BASE_DIR / scaler_file)
            X_test_model = scaler.transform(X_test_model)

        predictions = model.predict(X_test_model)

        metrics = {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
            "Precision": round(
                precision_score(y_test, predictions, zero_division=0) * 100, 2
            ),
            "Recall": round(
                recall_score(y_test, predictions, zero_division=0) * 100, 2
            ),
            "F1-Score": round(
                f1_score(y_test, predictions, zero_division=0) * 100, 2
            ),
        }

        results.append(metrics)

        print(f"\n{'=' * 45}")
        print(name)
        print("=" * 45)
        print("Accuracy:", metrics["Accuracy"], "%")
        print("Precision:", metrics["Precision"], "%")
        print("Recall:", metrics["Recall"], "%")
        print("F1-Score:", metrics["F1-Score"], "%")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            target_names=["Normal Cost", "High Cost"],
            zero_division=0
        ))

    except Exception as error:
        print(f"\nERROR in {name}: {error}")

if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Accuracy", ascending=False
    )

    results_df.to_csv(
        BASE_DIR / "model_performance.csv",
        index=False
    )

    print("\n\nFINAL MODEL COMPARISON")
    print(results_df.to_string(index=False))
    print("\nResults saved to model_performance.csv")
else:
    print("\nNo models were evaluated successfully.")
