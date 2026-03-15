"""
data_prep.py
------------
Load and perform initial cleaning on the raw churn dataset.
"""

from pathlib import Path

import pandas as pd


RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "online_retail_churn_raw.csv"
PROCESSED_PATH = (
    Path(__file__).parent.parent / "data" / "processed" / "online_retail_churn_clean.csv"
)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV file and return a DataFrame."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame, reference_date: str | None = None) -> pd.DataFrame:
    """
    Basic cleaning:
      - Drop the CustomerID column (not predictive).
      - Parse Last_Purchase_Date and derive Days_Since_Last_Purchase.
      - Drop rows with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    reference_date : str or None
        ISO-format date string (e.g. "2025-12-31") used as the "today" anchor
        when computing recency.  Defaults to the maximum date in the dataset,
        which is fine for offline experimentation but should be set explicitly
        in production or when comparing models trained at different points in time.
    """
    df = df.copy()

    # Drop identifier column
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Parse date and compute recency feature
    if "Last_Purchase_Date" in df.columns:
        df["Last_Purchase_Date"] = pd.to_datetime(df["Last_Purchase_Date"], errors="coerce")
        ref = (
            pd.Timestamp(reference_date)
            if reference_date is not None
            else df["Last_Purchase_Date"].max()
        )
        df["Days_Since_Last_Purchase"] = (ref - df["Last_Purchase_Date"]).dt.days
        df = df.drop(columns=["Last_Purchase_Date"])

    df = df.dropna()
    return df


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """Save cleaned DataFrame to the processed data directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run() -> pd.DataFrame:
    """Full data preparation pipeline: load → clean → save."""
    df_raw = load_raw()
    df_clean = clean(df_raw)
    save_processed(df_clean)
    print(f"Saved cleaned data to {PROCESSED_PATH}  ({len(df_clean):,} rows)")
    return df_clean


if __name__ == "__main__":
    run()
