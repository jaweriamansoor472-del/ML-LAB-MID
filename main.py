
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

classification_model = joblib.load(BASE_DIR / "classification_model.pkl")
regression_model = joblib.load(BASE_DIR / "regression_model.pkl")

classification_columns = joblib.load(BASE_DIR / "classification_columns.pkl")
regression_columns = joblib.load(BASE_DIR / "regression_columns.pkl")


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


@app.get("/")
def homepage():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/predict/regression")
def predict_regression(customer: Customer):
    data = prepare_data(customer.model_dump(), regression_columns)
    prediction = float(regression_model.predict(data)[0])

    return {"prediction": round(max(0, prediction), 2)}


@app.post("/predict/classification")
def predict_classification(customer: Customer):
    data = prepare_data(customer.model_dump(), classification_columns)
    prediction = int(classification_model.predict(data)[0])
    probability = float(classification_model.predict_proba(data)[0][1])

    return {
        "prediction": prediction,
        "result": "High Cost" if prediction == 1 else "Normal Cost",
        "probability": round(probability * 100, 1)
    }
