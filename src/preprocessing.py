import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INTERNET_ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "total_active_services",
    "avg_monthly_spend",
    "monthly_charge_diff",
]

BINARY_COLS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "has_support_bundle",
    "is_new_customer",
    "is_auto_pay",
]

CATEGORICAL_COLS = [
    "gender",
    "InternetService",
    "Contract",
    "PaymentMethod",
]


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cleans redundant service labels and creates domain churn features."""

    def fit(self, X: pd.DataFrame, y=None):
        self.output_columns_ = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Defensive numeric conversion for live API inputs
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)

        # Collapse redundant categories into clean binary states
        for col in INTERNET_ADDON_COLS:
            df[col] = df[col].replace({"No internet service": "No"})
        df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

        # Map binary Yes/No fields to 1/0 integers (prevents redundant OHE pairs)
        for col in YES_NO_COLS:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

        # Engineer domain features
        df["total_active_services"] = df[INTERNET_ADDON_COLS].sum(axis=1).astype(int)
        df["has_support_bundle"] = (
            (df["OnlineSecurity"] == 1) | (df["TechSupport"] == 1)
        ).astype(int)

        safe_tenure = df["tenure"].replace(0, np.nan)
        df["avg_monthly_spend"] = (df["TotalCharges"] / safe_tenure).fillna(df["MonthlyCharges"])
        df["monthly_charge_diff"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
        df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
        df["is_auto_pay"] = (
            df["PaymentMethod"]
            .astype(str)
            .str.contains("automatic", case=False, na=False)
            .astype(int)
        )

        return df[NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns_, dtype=object)


def build_preprocessor() -> Pipeline:
    """Returns a full preprocessing Pipeline ready to pair with any classifier."""
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", TelcoFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )


def load_and_split_data(
    filepath: str = "data/processed/telco_cleaned.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Loads cleaned data and returns a stratified train/test split."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()

    print(f"Train shape: {X_train.shape} | Churn rate: {y_train.mean() * 100:.2f}%")
    print(f"Test shape:  {X_test.shape} | Churn rate: {y_test.mean() * 100:.2f}%")

    # Fit ONLY on training data to prevent data leakage
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.named_steps["column_transformer"].get_feature_names_out()

    print(f"\nProcessed Train matrix: {X_train_processed.shape}")
    print(f"Processed Test matrix:  {X_test_processed.shape}")
    print(f"Missing values (Train / Test): {np.isnan(X_train_processed).sum()} / {np.isnan(X_test_processed).sum()}")
    print(f"Total final features ({len(feature_names)}): {list(feature_names)}")

    # Sanity check: simulate a single raw "What-If" update
    sample = X_test.iloc[[0]].copy()
    sample["Contract"] = "Two year"
    sample["TechSupport"] = "Yes"
    sample_processed = preprocessor.transform(sample)
    print(f"\nSingle-row What-If transform shape: {sample_processed.shape} (Success)")