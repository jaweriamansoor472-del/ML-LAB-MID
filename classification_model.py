import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load data
data = pd.read_csv("insurance.csv")

# 2. Convert text values into numbers
data["sex"] = data["sex"].map({"female": 0, "male": 1})
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
data["region"] = data["region"].map({
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
})

# 3. Create high-cost category
threshold = data["charges"].median()
data["high_cost"] = (data["charges"] >= threshold).astype(int)

# 4. Select input columns and target
X = data[["age", "sex", "bmi", "children", "smoker", "region"]]
y = data["high_cost"]

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# 8. Save the model
joblib.dump(model, "classification_model.joblib")

print("Classification model saved!")