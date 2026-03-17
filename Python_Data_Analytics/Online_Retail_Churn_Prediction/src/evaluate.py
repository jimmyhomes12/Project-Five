"""
evaluate.py
-----------
Load persisted models and produce evaluation plots:
  - ROC curves for both models
  - Feature importances for the Random Forest model
  - SHAP summary plot for model interpretability
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import RocCurveDisplay, roc_auc_score

from src.data_prep import clean, load_raw
from src.features import add_features
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
TARGET_COL = "Churn"


def load_model(filename: str):
    path = MODELS_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)


def get_test_data() -> tuple[pd.DataFrame, pd.Series]:
    df = clean(load_raw())
    df = add_features(df)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_test, y_test


def plot_roc_curves(logreg, rf, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_proba_lr = logreg.predict_proba(X_test)[:, 1]
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba_lr, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test, y_proba_rf, name="Random Forest", ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random Chance")
    ax.set_title("ROC Curves – Churn Models")
    ax.legend()
    plt.tight_layout()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "roc_curves.png"
    plt.savefig(out, dpi=150)
    print(f"Saved ROC curves → {out}")
    plt.show()


def plot_feature_importances(rf, X_test: pd.DataFrame) -> None:
    """Plot top-20 feature importances from the Random Forest pipeline."""
    model = rf.named_steps["model"]
    preprocessor = rf.named_steps["preprocess"]

    # Reconstruct feature names after ColumnTransformer
    num_names = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_names = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]).tolist()
    feature_names = list(num_names) + cat_names

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top20 = importances.nlargest(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    top20.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top-20 Feature Importances – Random Forest")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "feature_importances.png"
    plt.savefig(out, dpi=150)
    print(f"Saved feature importances → {out}")
    plt.show()


def plot_shap_summary(rf, X_test: pd.DataFrame, feature_names: list[str]) -> None:
    """Generate and save a SHAP summary bar plot for the Random Forest model."""
    rf_model = rf.named_steps["model"]
    preprocessor = rf.named_steps["preprocess"]

    # Use a sample of 200 rows to keep computation tractable
    rng = np.random.RandomState(42)
    n_sample = min(200, len(X_test))
    idx = rng.choice(len(X_test), size=n_sample, replace=False)
    X_sample_trans = preprocessor.transform(X_test.iloc[idx])

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample_trans)

    # shap_values may be (n_samples, n_features, n_classes) or a list
    sv = np.array(shap_values)
    sv_churn = sv[:, :, 1] if sv.ndim == 3 else shap_values[1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv_churn,
        X_sample_trans,
        feature_names=feature_names,
        max_display=15,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance – Top Drivers of Churn Prediction")
    plt.tight_layout()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary → {out}")

    # Print top SHAP features
    shap_imp = pd.Series(np.abs(sv_churn).mean(axis=0), index=feature_names).nlargest(5)
    print("Top 5 SHAP features (mean |SHAP value|):")
    for feat, val in shap_imp.items():
        print(f"  {feat}: {val:.6f}")


def run() -> None:
    logreg = load_model("churn_logreg.pkl")
    rf = load_model("churn_rf.pkl")
    X_test, y_test = get_test_data()

    lr_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"Logistic Regression ROC AUC: {lr_auc:.4f}")
    print(f"Random Forest      ROC AUC: {rf_auc:.4f}")

    plot_roc_curves(logreg, rf, X_test, y_test)

    # Reconstruct feature names for importance and SHAP plots
    preprocessor = rf.named_steps["preprocess"]
    num_names = list(preprocessor.transformers_[0][2])
    cat_encoder = preprocessor.transformers_[1][1]
    cat_names = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]).tolist()
    feature_names = num_names + cat_names

    plot_feature_importances(rf, X_test)
    plot_shap_summary(rf, X_test, feature_names)


if __name__ == "__main__":
    run()
