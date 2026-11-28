import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src.preprocessing import CATEGORICAL_COLS, load_and_split_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
BACKGROUND_PATH = BASE_DIR / "models" / "shap_background.npy"

DISPLAY_NAMES = {
    "tenure": "Tenure (Months)",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
    "total_active_services": "Active Add-On Services",
    "avg_monthly_spend": "Avg Monthly Spend ($)",
    "monthly_charge_diff": "Recent Bill Change ($)",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "PaperlessBilling": "Paperless Billing",
    "has_support_bundle": "Support Bundle (Security/Tech)",
    "is_new_customer": "New Customer (<= 6 Months)",
    "is_auto_pay": "Auto-Pay Enabled",
    "gender": "Gender",
    "InternetService": "Internet Service",
    "Contract": "Contract Type",
    "PaymentMethod": "Payment Method",
}


def build_shap_background(sample_size: int = 100, random_state: int = 42) -> np.ndarray:
    """Saves a compact transformed training sample for fast probability-space SHAP."""
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]

    X_train, _, _, _ = load_and_split_data()
    bg_raw = X_train.sample(n=min(sample_size, len(X_train)), random_state=random_state)
    bg_transformed = np.asarray(preprocessor.transform(bg_raw), dtype=np.float64)

    BACKGROUND_PATH.parent.mkdir(exist_ok=True)
    np.save(BACKGROUND_PATH, bg_transformed)
    return bg_transformed


class ChurnRetentionEngine:
    """Unified ML inference, SHAP explainability, and What-If simulation engine."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
        background_path: Path = BACKGROUND_PATH,
    ):
        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.threshold = float(self.metadata["recommended_threshold"])
        self.feature_names = list(
            self.preprocessor.named_steps["column_transformer"].get_feature_names_out()
        )

        if background_path.exists():
            self.background = np.load(background_path)
        else:
            self.background = build_shap_background()

        self.explainer = self._init_explainer()

    def _init_explainer(self):
        """Initializes TreeExplainer in probability space with safe fallback."""
        try:
            return shap.TreeExplainer(
                self.classifier,
                data=self.background,
                model_output="probability",
            )
        except Exception:
            # Model-agnostic fallback if a non-tree model or version mismatch occurs
            return shap.Explainer(
                lambda x: self.classifier.predict_proba(x)[:, 1],
                self.background,
            )

    @staticmethod
    def _extract_positive_shap(shap_vals: Any, expected_val: Any) -> tuple[np.ndarray, float]:
        """Normalizes SHAP output shapes across XGBoost, LightGBM, RF, and SHAP versions."""
        if isinstance(shap_vals, shap.Explanation):
            base = shap_vals.base_values
            vals = shap_vals.values
        else:
            base = expected_val
            vals = shap_vals

        if isinstance(vals, list):
            vals = vals[1]

        vals_arr = np.asarray(vals, dtype=np.float64)
        if vals_arr.ndim == 3:
            vals_arr = vals_arr[0, :, 1]
        elif vals_arr.ndim == 2:
            vals_arr = vals_arr[0]

        base_arr = np.asarray(base, dtype=np.float64).ravel()
        base_scalar = float(base_arr[1] if len(base_arr) == 2 else base_arr[0])

        return vals_arr, base_scalar

    def _risk_tier(self, probability: float) -> str:
        if probability >= 0.60:
            return "High"
        if probability >= self.threshold:
            return "Medium"
        return "Low"

    def _group_shap_values(
        self,
        raw_shap: np.ndarray,
        engineered_row: pd.Series,
    ) -> list[dict]:
        """Aggregates one-hot encoded SHAP values back to parent business features."""
        grouped: dict[str, float] = {}

        for feat_name, shap_val in zip(self.feature_names, raw_shap):
            parent = feat_name
            for cat_col in CATEGORICAL_COLS:
                if feat_name.startswith(f"{cat_col}_"):
                    parent = cat_col
                    break
            grouped[parent] = grouped.get(parent, 0.0) + float(shap_val)

        drivers = []
        for feature, impact in grouped.items():
            raw_val = engineered_row[feature]
            if isinstance(raw_val, (np.integer, int)):
                val_out: Any = int(raw_val)
            elif isinstance(raw_val, (np.floating, float)):
                val_out = round(float(raw_val), 2)
            else:
                val_out = str(raw_val)

            drivers.append(
                {
                    "feature": feature,
                    "display_name": DISPLAY_NAMES.get(feature, feature),
                    "value": val_out,
                    "impact_pct": round(impact * 100, 2),
                    "direction": "increases_risk" if impact >= 0 else "reduces_risk",
                }
            )

        drivers.sort(key=lambda d: abs(d["impact_pct"]), reverse=True)
        return drivers

    def predict_and_explain(self, customer: dict, top_k: int = 8) -> dict:
        """Scores a single customer and returns grouped probability-space SHAP drivers."""
        df = pd.DataFrame([customer])

        probability = float(self.pipeline.predict_proba(df)[0, 1])
        engineered_df = self.preprocessor.named_steps["feature_engineering"].transform(df)
        X_proc = np.asarray(self.preprocessor.transform(df), dtype=np.float64)

        if isinstance(self.explainer, shap.TreeExplainer):
            raw_vals = self.explainer.shap_values(X_proc)
            shap_vec, base_prob = self._extract_positive_shap(
                raw_vals, self.explainer.expected_value
            )
        else:
            explanation = self.explainer(X_proc)
            shap_vec, base_prob = self._extract_positive_shap(explanation, None)

        all_drivers = self._group_shap_values(shap_vec, engineered_df.iloc[0])

        return {
            "churn_probability": round(probability, 4),
            "churn_probability_pct": round(probability * 100, 2),
            "base_probability_pct": round(base_prob * 100, 2),
            "threshold": self.threshold,
            "risk_tier": self._risk_tier(probability),
            "flagged_for_retention": bool(probability >= self.threshold),
            "top_drivers": all_drivers[:top_k],
            "all_drivers_sum_check_pct": round(
                base_prob * 100 + sum(d["impact_pct"] for d in all_drivers), 2
            ),
        }

    def simulate_retention(
        self,
        customer: dict,
        changes: dict,
        intervention_cost_usd: float = 100.0,
        horizon_months: int = 12,
        top_k: int = 8,
    ) -> dict:
        """Compares baseline vs. modified customer profile and estimates retention ROI."""
        original = self.predict_and_explain(customer, top_k=top_k)

        modified_customer = {**customer, **changes}
        simulated = self.predict_and_explain(modified_customer, top_k=top_k)

        prob_delta = original["churn_probability"] - simulated["churn_probability"]
        prob_delta_pct = round(prob_delta * 100, 2)

        monthly_revenue = float(modified_customer.get("MonthlyCharges", 0.0))
        annual_customer_value = monthly_revenue * horizon_months
        expected_saved_revenue = max(prob_delta, 0.0) * annual_customer_value
        net_roi_usd = expected_saved_revenue - intervention_cost_usd
        roi_pct = (
            round((net_roi_usd / intervention_cost_usd) * 100, 1)
            if intervention_cost_usd > 0
            else None
        )

        return {
            "original": original,
            "simulated": simulated,
            "applied_changes": changes,
            "risk_reduction_pct": prob_delta_pct,
            "roi_analysis": {
                "horizon_months": horizon_months,
                "annual_customer_value_usd": round(annual_customer_value, 2),
                "intervention_cost_usd": round(intervention_cost_usd, 2),
                "expected_saved_revenue_usd": round(expected_saved_revenue, 2),
                "expected_net_roi_usd": round(net_roi_usd, 2),
                "roi_pct": roi_pct,
                "recommendation": (
                    "Profitable Retention Action"
                    if net_roi_usd > 0
                    else "Not Cost-Effective"
                ),
            },
        }


if __name__ == "__main__":
    print("Building SHAP background dataset...")
    build_shap_background()

    engine = ChurnRetentionEngine()
    _, X_test, _, _ = load_and_split_data()

    # Pick the highest-risk customer in the test set for a realistic demo
    test_probs = engine.pipeline.predict_proba(X_test)[:, 1]
    highest_risk_idx = int(np.argmax(test_probs))
    sample_customer = X_test.iloc[highest_risk_idx].to_dict()

    print("\n=== High-Risk Customer Baseline Explanation ===")
    baseline = engine.predict_and_explain(sample_customer, top_k=6)
    print(f"Predicted Churn Risk: {baseline['churn_probability_pct']}% ({baseline['risk_tier']} Risk)")
    print(f"Baseline Average:     {baseline['base_probability_pct']}%")
    print(f"SHAP Additivity Check:{baseline['all_drivers_sum_check_pct']}% (should match Predicted Churn Risk)")
    print("\nTop 6 Drivers:")
    for d in baseline["top_drivers"]:
        sign = "+" if d["impact_pct"] >= 0 else ""
        print(f"  - {d['display_name']} = {d['value']}: {sign}{d['impact_pct']}%")

    print("\n=== What-If Retention Simulation ===")
    discount_price = round(float(sample_customer["MonthlyCharges"]) * 0.90, 2)
    offer = {
        "Contract": "One year",
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": discount_price,
    }
    sim = engine.simulate_retention(
        sample_customer,
        changes=offer,
        intervention_cost_usd=120.0,
    )
    print(f"Before Offer:   {sim['original']['churn_probability_pct']}% ({sim['original']['risk_tier']})")
    print(f"After Offer:    {sim['simulated']['churn_probability_pct']}% ({sim['simulated']['risk_tier']})")
    print(f"Risk Reduction: -{sim['risk_reduction_pct']} percentage points")
    print(f"ROI Analysis:   {json.dumps(sim['roi_analysis'], indent=2)}")