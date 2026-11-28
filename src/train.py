import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor, load_and_split_data

# Business assumptions for retention ROI threshold tuning
CHURN_LOSS = 500.0          # Avg revenue lost when a customer churns ($)
OFFER_COST = 50.0           # Cost of a targeted retention offer ($)
SAVE_RATE = 0.30            # Probability a targeted churner stays after the offer


def get_candidate_models() -> dict:
    """Returns candidate classifiers from simple linear baseline to boosted trees."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
    }


def calculate_net_savings(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes expected net retention savings ($) compared to taking no action."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    expected_saved_revenue = tp * (CHURN_LOSS * SAVE_RATE)
    total_offer_spend = (tp + fp) * OFFER_COST
    return float(expected_saved_revenue - total_offer_spend)


def benchmark_models(X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> pd.DataFrame:
    """Evaluates all candidate models using leak-free 5-fold stratified CV."""
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }

    rows = []
    for name, classifier in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        rows.append(
            {
                "Model": name,
                "CV_ROC_AUC": scores["test_roc_auc"].mean(),
                "CV_ROC_STD": scores["test_roc_auc"].std(),
                "CV_PR_AUC": scores["test_pr_auc"].mean(),
                "CV_F1": scores["test_f1"].mean(),
                "CV_Recall": scores["test_recall"].mean(),
                "CV_Precision": scores["test_precision"].mean(),
            }
        )

    results_df = pd.DataFrame(rows).sort_values("CV_ROC_AUC", ascending=False).reset_index(drop=True)
    return results_df


def tune_threshold_oof(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> dict:
    """Tunes classification threshold strictly on out-of-fold training probabilities."""
    oof_probs = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    thresholds = np.arange(0.15, 0.75, 0.01)
    best_f1_thresh = 0.50
    best_f1 = -1.0
    best_biz_thresh = 0.50
    best_biz_savings = -np.inf

    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)
        savings = calculate_net_savings(y_train.to_numpy(), preds)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = float(round(t, 2))
        if savings > best_biz_savings:
            best_biz_savings = savings
            best_biz_thresh = float(round(t, 2))

    return {
        "f1_optimal_threshold": best_f1_thresh,
        "oof_best_f1": round(float(best_f1), 4),
        "business_optimal_threshold": best_biz_thresh,
        "oof_best_net_savings": round(float(best_biz_savings), 2),
    }


def evaluate_at_threshold(y_true: pd.Series, probs: np.ndarray, threshold: float) -> dict:
    """Computes full classification metrics and business ROI at a specific threshold."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    return {
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "expected_net_savings_usd": round(calculate_net_savings(y_true.to_numpy(), preds), 2),
    }


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=== Step 1: 5-Fold Stratified Cross-Validation Benchmark ===")
    benchmark_df = benchmark_models(X_train, y_train, cv)
    print(benchmark_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Select winning model based on CV ROC-AUC
    best_model_name = benchmark_df.iloc[0]["Model"]
    best_classifier = get_candidate_models()[best_model_name]
    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", best_classifier),
        ]
    )
    print(f"\nWinner selected by 5-Fold CV ROC-AUC: {best_model_name}")

    print("\n=== Step 2: Out-of-Fold Threshold Tuning (Zero Test Leakage) ===")
    threshold_info = tune_threshold_oof(best_pipeline, X_train, y_train, cv)
    tuned_threshold = threshold_info["business_optimal_threshold"]
    print(f"F1-Optimal Threshold (OOF):       {threshold_info['f1_optimal_threshold']:.2f} (F1: {threshold_info['oof_best_f1']:.4f})")
    print(f"Business-Optimal Threshold (OOF): {tuned_threshold:.2f} (OOF Net Savings: ${threshold_info['oof_best_net_savings']:,.2f})")

    print("\n=== Step 3: Final Evaluation on Locked-Away Test Set (1,409 customers) ===")
    best_pipeline.fit(X_train, y_train)
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]

    default_metrics = evaluate_at_threshold(y_test, test_probs, threshold=0.50)
    tuned_metrics = evaluate_at_threshold(y_test, test_probs, threshold=tuned_threshold)

    comparison = pd.DataFrame(
        [
            {"Setting": "Default (0.50)", **{k: v for k, v in default_metrics.items() if k != "confusion_matrix"}},
            {"Setting": f"Tuned ({tuned_threshold:.2f})", **{k: v for k, v in tuned_metrics.items() if k != "confusion_matrix"}},
        ]
    )
    print(comparison.to_string(index=False))

    print(f"\nConfusion Matrix @ Default (0.50): {default_metrics['confusion_matrix']}")
    print(f"Confusion Matrix @ Tuned   ({tuned_threshold:.2f}): {tuned_metrics['confusion_matrix']}")

    # Save trained pipeline and metadata for Phases 5 & 6
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "churn_model.joblib"
    metadata_path = models_dir / "model_metadata.json"

    joblib.dump(best_pipeline, model_path)

    feature_names = (
        best_pipeline.named_steps["preprocessor"]
        .named_steps["column_transformer"]
        .get_feature_names_out()
        .tolist()
    )

    metadata = {
        "best_model_name": best_model_name,
        "recommended_threshold": tuned_threshold,
        "threshold_tuning_oof": threshold_info,
        "cv_benchmark": benchmark_df.to_dict(orient="records"),
        "test_evaluation_default": default_metrics,
        "test_evaluation_tuned": tuned_metrics,
        "business_assumptions": {
            "churn_loss_usd": CHURN_LOSS,
            "offer_cost_usd": OFFER_COST,
            "save_rate": SAVE_RATE,
        },
        "engineered_feature_names": feature_names,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved trained pipeline -> {model_path}")
    print(f"Saved model metadata   -> {metadata_path}")


if __name__ == "__main__":
    main()