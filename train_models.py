import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("insurance.csv")

# Prepare data
X = df.drop("charges", axis=1)
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Classification target: high or low charges
y_class = (df["charges"] > df["charges"].median()).astype(int)

# Regression target: actual charges
y_reg = df["charges"]

# Split classification data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# Train classification model
classification_model = LogisticRegression(max_iter=1000)
classification_model.fit(X_train, y_train)

print("Classification Accuracy:",
      round(accuracy_score(y_test, classification_model.predict(X_test)) * 100, 2), "%")

# Save classification model and its input columns
joblib.dump(classification_model, "classification_model.pkl")
joblib.dump(X.columns.tolist(), "classification_columns.pkl")

# Split regression data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Train regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

print("Regression MAE:",
      round(mean_absolute_error(y_test, regression_model.predict(X_test)), 2))

print("Regression R2 Score:",
      round(regression_model.score(X_test, y_test), 4))

# Save regression model and its input columns
joblib.dump(regression_model, "regression_model.pkl")
joblib.dump(X.columns.tolist(), "regression_columns.pkl")

print("\nBoth models trained and saved successfully!")