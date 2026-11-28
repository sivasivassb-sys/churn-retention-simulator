import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import CustomerFeatures, SimulateRequest
from src.predictor import ChurnRetentionEngine

SAMPLE_CUSTOMERS = [
    {
        "id": "high_risk_fiber",
        "label": "High-Risk New Fiber User",
        "summary": "Month-to-month fiber customer (3 months tenure) paying by electronic check with no support bundle.",
        "profile": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.5,
            "TotalCharges": 268.5,
        },
    },
    {
        "id": "medium_risk_dsl",
        "label": "Medium-Risk DSL User",
        "summary": "Month-to-month DSL customer (14 months tenure) with partial services and mailed check billing.",
        "profile": {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 14,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 58.2,
            "TotalCharges": 814.8,
        },
    },
    {
        "id": "low_risk_loyal",
        "label": "Low-Risk Loyal 2-Year User",
        "summary": "Established 2-year contract customer (54 months tenure) with security, tech support, and auto-pay.",
        "profile": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 54,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 79.4,
            "TotalCharges": 4287.6,
        },
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads ML pipeline and SHAP explainer once at server startup."""
    app.state.engine = ChurnRetentionEngine()
    yield


app = FastAPI(
    title="Customer Churn Predictor & What-If Retention API",
    description="Real-time churn scoring, probability-space SHAP explainability, and retention ROI simulation.",
    version="1.0.0",
    lifespan=lifespan,
)

raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine(request: Request) -> ChurnRetentionEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        request.app.state.engine = ChurnRetentionEngine()
        engine = request.app.state.engine
    return engine


@app.get("/health")
def health_check(request: Request):
    """Returns API status and winning model evaluation metadata."""
    engine = get_engine(request)
    meta = engine.metadata
    return {
        "status": "ok",
        "model_name": meta.get("best_model_name"),
        "recommended_threshold": engine.threshold,
        "test_roc_auc": meta.get("test_evaluation_tuned", {}).get("roc_auc"),
        "test_recall_tuned": meta.get("test_evaluation_tuned", {}).get("recall"),
        "test_net_savings_usd": meta.get("test_evaluation_tuned", {}).get("expected_net_savings_usd"),
    }


@app.get("/sample-customers")
def get_sample_customers():
    """Returns realistic customer presets for instant dashboard testing."""
    return {"customers": SAMPLE_CUSTOMERS}


@app.post("/predict")
def predict_churn(customer: CustomerFeatures, request: Request):
    """Fast churn prediction without SHAP computation."""
    engine = get_engine(request)
    df = pd.DataFrame([customer.model_dump()])
    probability = float(engine.pipeline.predict_proba(df)[0, 1])

    return {
        "churn_probability": round(probability, 4),
        "churn_probability_pct": round(probability * 100, 2),
        "threshold": engine.threshold,
        "risk_tier": engine._risk_tier(probability),
        "flagged_for_retention": bool(probability >= engine.threshold),
    }


@app.post("/explain")
def explain_churn(
    customer: CustomerFeatures,
    request: Request,
    top_k: int = Query(default=8, ge=1, le=25),
):
    """Returns churn prediction plus grouped probability-space SHAP drivers."""
    engine = get_engine(request)
    return engine.predict_and_explain(customer.model_dump(), top_k=top_k)


@app.post("/simulate")
def simulate_retention(payload: SimulateRequest, request: Request):
    """Compares baseline vs. modified customer profile and computes 12-month ROI."""
    engine = get_engine(request)
    changes = payload.changes.model_dump(exclude_none=True)

    return engine.simulate_retention(
        customer=payload.customer.model_dump(),
        changes=changes,
        intervention_cost_usd=payload.intervention_cost_usd,
        horizon_months=payload.horizon_months,
        top_k=payload.top_k,
    )