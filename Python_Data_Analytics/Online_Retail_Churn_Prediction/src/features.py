"""
features.py
-----------
Feature engineering for the churn prediction dataset.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features:
      - Total_Spend: Total_Purchases × Avg_Purchase_Value
      - Engagement_Score: Website_Visits_Last_Month + Referred_Friends
      - High_Support: binary flag for customers with > 2 support tickets
      - Income_Band: quartile-based band from Annual_Income_USD
    """
    df = df.copy()

    if {"Total_Purchases", "Avg_Purchase_Value"}.issubset(df.columns):
        df["Total_Spend"] = df["Total_Purchases"] * df["Avg_Purchase_Value"]

    # Use explicit known engagement columns so the aggregation is predictable
    engagement_cols = [
        c for c in ["Website_Visits_Last_Month", "Referred_Friends"] if c in df.columns
    ]
    if engagement_cols:
        df["Engagement_Score"] = df[engagement_cols].sum(axis=1)

    if "Support_Tickets_Last_6_Months" in df.columns:
        df["High_Support"] = (df["Support_Tickets_Last_6_Months"] > 2).astype(int)

    if "Annual_Income_USD" in df.columns:
        df["Income_Band"] = pd.qcut(
            df["Annual_Income_USD"],
            4,
            labels=["Low", "Mid-Low", "Mid-High", "High"],
        )

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Return a ColumnTransformer that scales numeric and one-hot encodes categorical features."""
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )
    return preprocessor
