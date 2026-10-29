
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Load the dataset
data = pd.read_csv("insurance.csv")

# Convert text into numbers
data["sex"] = data["sex"].map({"female": 0, "male": 1})
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
data["region"] = data["region"].map({
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
})

# Select input columns and target
X = data[["age", "sex", "bmi", "children", "smoker", "region"]]
y = data["charges"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Check model performance
predictions = model.predict(X_test)
print("R2 Score:", r2_score(y_test, predictions))

# Save the trained model
joblib.dump(model, "regression_model.joblib")

print("Regression model saved!")
