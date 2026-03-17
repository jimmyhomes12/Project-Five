"""
train_model.py
--------------
Train Logistic Regression and Random Forest churn prediction models,
evaluate them, and persist the fitted pipelines to disk.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_prep import clean, load_raw
from src.features import add_features, build_preprocessor

MODELS_DIR = Path(__file__).parent.parent / "models"
TARGET_COL = "Churn"


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, engineer features, and split into train/test sets."""
    df = clean(load_raw())
    df = add_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def build_logreg_pipeline(X_train: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X_train)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )


def build_rf_pipeline(X_train: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X_train)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
            )),
        ]
    )


def evaluate(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n{'='*50}")
    print(f"{name}  –  ROC AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))


def save_model(pipeline: Pipeline, filename: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved model → {path}")


def run() -> None:
    X_train, X_test, y_train, y_test = prepare_data()

    # --- Logistic Regression ---
    logreg = build_logreg_pipeline(X_train)
    logreg.fit(X_train, y_train)
    evaluate("Logistic Regression", logreg, X_test, y_test)
    save_model(logreg, "churn_logreg.pkl")

    # --- Random Forest ---
    rf = build_rf_pipeline(X_train)
    rf.fit(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test)
    save_model(rf, "churn_rf.pkl")


if __name__ == "__main__":
    run()
