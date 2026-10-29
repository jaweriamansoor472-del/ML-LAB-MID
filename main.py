
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(title="InsureAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# Existing regression and logistic regression models
classification_model = joblib.load(BASE_DIR / "classification_model.pkl")
regression_model = joblib.load(BASE_DIR / "regression_model.pkl")

classification_columns = joblib.load(BASE_DIR / "classification_columns.pkl")
regression_columns = joblib.load(BASE_DIR / "regression_columns.pkl")

# KNN
knn_model = joblib.load(BASE_DIR / "knn_model.pkl")
knn_scaler = joblib.load(BASE_DIR / "knn_scaler.pkl")
knn_columns = joblib.load(BASE_DIR / "knn_columns.pkl")

# ANN
ann_model = joblib.load(BASE_DIR / "ann_model.pkl")
ann_scaler = joblib.load(BASE_DIR / "ann_scaler.pkl")
ann_columns = joblib.load(BASE_DIR / "ann_columns.pkl")

# SVM
svm_model = joblib.load(BASE_DIR / "svm_model.pkl")
svm_scaler = joblib.load(BASE_DIR / "svm_scaler.pkl")
svm_columns = joblib.load(BASE_DIR / "svm_columns.pkl")

# Decision Tree
decision_tree_model = joblib.load(BASE_DIR / "decision_tree_model.pkl")
decision_tree_columns = joblib.load(
    BASE_DIR / "decision_tree_columns.pkl"
)

# Naive Bayes
naive_bayes_model = joblib.load(BASE_DIR / "naive_bayes_model.pkl")
naive_bayes_columns = joblib.load(BASE_DIR / "naive_bayes_columns.pkl")

# Random Forest
random_forest_model = joblib.load(BASE_DIR / "random_forest_model.pkl")
random_forest_columns = joblib.load(
    BASE_DIR / "random_forest_columns.pkl"
)


class Customer(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


def prepare_data(customer, columns):
    data = pd.DataFrame([customer])
    data = pd.get_dummies(data, dtype=int)
    data = data.reindex(columns=columns, fill_value=0)
    return data


def predict_classification_model(
    customer,
    model,
    columns,
    scaler=None
):
    data = prepare_data(customer.model_dump(), columns)

    if scaler is not None:
        data = scaler.transform(data)

    prediction = int(model.predict(data)[0])

    result = "High Cost" if prediction == 1 else "Normal Cost"

    response = {
        "prediction": prediction,
        "result": result
    }

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(data)[0][1])
        response["probability"] = round(probability * 100, 1)

    return response


@app.get("/")
def homepage():
    return FileResponse(BASE_DIR / "index.html")


# Existing regression endpoint
@app.post("/predict/regression")
def predict_regression(customer: Customer):
    data = prepare_data(customer.model_dump(), regression_columns)
    prediction = float(regression_model.predict(data)[0])

    return {"prediction": round(max(0, prediction), 2)}


# Existing logistic regression endpoint
@app.post("/predict/classification")
def predict_classification(customer: Customer):
    return predict_classification_model(
        customer,
        classification_model,
        classification_columns
    )


# KNN endpoint
@app.post("/predict/knn")
def predict_knn(customer: Customer):
    return predict_classification_model(
        customer, knn_model, knn_columns, knn_scaler
    )


# ANN endpoint
@app.post("/predict/ann")
def predict_ann(customer: Customer):
    return predict_classification_model(
        customer, ann_model, ann_columns, ann_scaler
    )


# SVM endpoint
@app.post("/predict/svm")
def predict_svm(customer: Customer):
    return predict_classification_model(
        customer, svm_model, svm_columns, svm_scaler
    )


# Decision Tree endpoint
@app.post("/predict/decision-tree")
def predict_decision_tree(customer: Customer):
    return predict_classification_model(
        customer, decision_tree_model, decision_tree_columns
    )


# Naive Bayes endpoint
@app.post("/predict/naive-bayes")
def predict_naive_bayes(customer: Customer):
    return predict_classification_model(
        customer, naive_bayes_model, naive_bayes_columns
    )


# Random Forest endpoint
@app.post("/predict/random-forest")
def predict_random_forest(customer: Customer):
    return predict_classification_model(
        customer, random_forest_model, random_forest_columns
    )

@app.get("/model-performance")
def get_model_performance():
    results_path = BASE_DIR / "model_performance.csv"

    if not results_path.exists():
        return {"error": "Model performance results not found"}

    results = pd.read_csv(results_path)
    return results.to_dict(orient="records")
