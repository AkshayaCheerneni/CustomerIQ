import os
from io import BytesIO
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = FastAPI(
    title="CustomerIQ API",
    description="Customer churn, CLV prediction and segmentation API",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

churn_model = None
clv_model = None
cluster_scaler = None
kmeans_model = None

uploaded_data = None

numerical_features = [
    "age",
    "tenure_months",
    "monthly_spend",
    "total_spend",
    "login_frequency",
    "support_tickets",
    "product_usage",
    "discount_used",
    "average_monthly_spend",
    "usage_per_month",
    "support_ticket_rate"
]

categorical_features = [
    "gender",
    "location",
    "subscription_type",
    "payment_method",
    "age_group",
    "tenure_group"
]

predictor_features = numerical_features + categorical_features

clustering_features = [
    "tenure_months",
    "monthly_spend",
    "login_frequency",
    "support_tickets",
    "product_usage",
    "customer_lifetime_value"
]


class CustomerInput(BaseModel):
    age: float
    tenure_months: float
    monthly_spend: float
    total_spend: float
    login_frequency: float
    support_tickets: float
    product_usage: float
    discount_used: float
    average_monthly_spend: float
    usage_per_month: float
    support_ticket_rate: float
    gender: str
    location: str
    subscription_type: str
    payment_method: str
    age_group: str
    tenure_group: str
    customer_lifetime_value: float


def load_churn_model():
    global churn_model

    if churn_model is None:
        churn_model = joblib.load(
            os.path.join(MODEL_DIR, "churn_model.joblib")
        )

    return churn_model


def load_clv_model():
    global clv_model

    if clv_model is None:
        clv_model = joblib.load(
            os.path.join(MODEL_DIR, "clv_model.joblib")
        )

    return clv_model


def load_cluster_models():
    global cluster_scaler
    global kmeans_model

    if cluster_scaler is None:
        cluster_scaler = joblib.load(
            os.path.join(MODEL_DIR, "cluster_scaler.joblib")
        )

    if kmeans_model is None:
        kmeans_model = joblib.load(
            os.path.join(MODEL_DIR, "kmeans_model.joblib")
        )

    return cluster_scaler, kmeans_model


@app.get("/")
def home():
    return {
        "message": "CustomerIQ API is running"
    }


@app.post("/upload")
async def upload_data(file: Annotated[UploadFile, File(...)]):
    global uploaded_data

    contents = await file.read()

    try:
        uploaded_data = pd.read_csv(BytesIO(contents))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a valid CSV file."
        )

    return {
        "message": "File uploaded successfully",
        "rows": len(uploaded_data),
        "columns": list(uploaded_data.columns)
    }


@app.post("/predict/churn")
def predict_churn(customer: CustomerInput):
    model = load_churn_model()
    data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(
        data[predictor_features]
    )[0]

    return {
        "churn_prediction": int(prediction),
        "churn_status": "Churn" if prediction == 1 else "Retained"
    }


@app.post("/predict/clv")
def predict_clv(customer: CustomerInput):
    model = load_clv_model()
    data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(
        data[predictor_features]
    )[0]

    return {
        "predicted_clv": float(prediction)
    }


@app.post("/segment")
def segment_customer(customer: CustomerInput):
    scaler, model = load_cluster_models()
    data = pd.DataFrame([customer.model_dump()])

    cluster_data = data[clustering_features]

    scaled_data = scaler.transform(
        cluster_data
    )

    cluster = model.predict(
        scaled_data
    )[0]

    return {
        "cluster": int(cluster)
    }


@app.get("/metrics")
def get_metrics():
    return {
        "classification_results": pd.read_csv(
            os.path.join(
                BASE_DIR,
                "Data",
                "Processed",
                "classification_results.csv"
            )
        ).to_dict(orient="records"),
        "regression_results": pd.read_csv(
            os.path.join(
                BASE_DIR,
                "Data",
                "Processed",
                "regression_results.csv"
            )
        ).to_dict(orient="records")
    }


@app.post("/train")
def train_models():
    global churn_model
    global clv_model

    data_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "customerIQ_customer_churn_features.csv"
    )

    data = pd.read_csv(data_path)

    X = data[predictor_features]

    y_churn = data["churn"]

    churn_model = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            StandardScaler(),
                            numerical_features
                        ),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            categorical_features
                        )
                    ]
                )
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    churn_model.fit(X, y_churn)

    y_clv = data["customer_lifetime_value"]

    clv_model = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            StandardScaler(),
                            numerical_features
                        ),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            categorical_features
                        )
                    ]
                )
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    clv_model.fit(X, y_clv)

    joblib.dump(
        churn_model,
        os.path.join(MODEL_DIR, "churn_model.joblib")
    )

    joblib.dump(
        clv_model,
        os.path.join(MODEL_DIR, "clv_model.joblib")
    )

    return {
        "message": "Models trained successfully"
    }
