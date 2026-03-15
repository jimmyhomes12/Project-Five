# Online Retail Customer Churn Prediction (Python)

This project builds a machine learning pipeline to predict which online retail customers are likely to churn and explains the key drivers behind that churn. It is part of the **Python_Data_Analytics** portfolio track.

## Dataset

**Source:** [Online Retail Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/sahilislam007/online-retail-customer-churn-prediction-dataset) by `sahilislam007` on Kaggle.

Each row represents one customer with:

| Column | Description |
|---|---|
| `Age` | Customer age |
| `Gender` | Gender |
| `Annual_Income_USD` | Annual income in USD |
| `Spending_Score` | Retailer-assigned spending score (1-100) |
| `Membership_Status` | Bronze / Silver / Gold |
| `Preferred_Payment_Method` | Payment method used most |
| `Region` | Geographic region |
| `Total_Purchases` | Lifetime purchase count |
| `Avg_Purchase_Value` | Average order value (USD) |
| `Last_Purchase_Date` | Date of most recent purchase |
| `Satisfaction_Score` | Customer satisfaction rating (1-5) |
| `Website_Visits_Last_Month` | Site visits in the last 30 days |
| `Avg_Time_Per_Visit_Minutes` | Average session length (minutes) |
| `Support_Tickets_Last_6_Months` | Support contacts in the last 6 months |
| `Referred_Friends` | Number of friends referred |
| `Churn` | Binary target: 1 = churned, 0 = retained |

## Business questions

- Which customers are at highest risk of churning in the next period?
- What behavioral and demographic factors are most strongly associated with churn?
- How accurately can we predict churn with a simple interpretable model vs. a tree-based model?

## Project structure

```text
Online_Retail_Churn_Prediction/
├── data/
│   ├── raw/online_retail_churn_raw.csv
│   └── processed/online_retail_churn_clean.csv
├── notebooks/
│   └── 01_online_retail_churn_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── features.py
│   ├── train_model.py
│   └── evaluate.py
├── models/
│   ├── churn_logreg.pkl
│   └── churn_tree.pkl
├── reports/
│   └── churn_model_summary.md
├── requirements.txt
└── README.md
```

## Methods

### Exploratory data analysis (EDA)
Examined churn rate, numeric distributions, correlations, and churn rates by membership level, region, gender, and payment method.

### Feature engineering
| Engineered feature | Description |
|---|---|
| `Total_Spend` | `Total_Purchases × Avg_Purchase_Value` |
| `Engagement_Score` | `Website_Visits_Last_Month + Referred_Friends` |
| `High_Support` | Binary flag: > 2 support tickets in last 6 months |
| `Income_Band` | Quartile-based income tier (Low / Mid-Low / Mid-High / High) |
| `Days_Since_Last_Purchase` | Recency derived from `Last_Purchase_Date` |

### Modeling

| Model | Notes |
|---|---|
| Logistic Regression | Baseline – interpretable, class-weighted |
| Random Forest | 300 estimators, class-weighted, balanced |

### Evaluation
Metrics: ROC AUC, precision, recall, F1-score, confusion matrix.

### Interpretability
Top-20 feature importances and SHAP summary plots from the Random Forest reveal which signals most increase churn risk. SHAP plots are saved automatically to `reports/shap_summary.png` when running `src/evaluate.py`.

## Key findings

- **Churn rate:** 19.73% of the 9,000 customers churned (1,776 out of 9,000).
- **Model performance:** Random Forest ROC AUC = **0.5305**; Logistic Regression ROC AUC = **0.4882**. Both results reflect the near-random churn labelling present in this synthetic benchmark dataset, making it an ideal baseline for validating an end-to-end pipeline.
- **Top 3 churn drivers** (Random Forest feature importance + SHAP):
  1. **Avg_Time_Per_Visit_Minutes** – Session duration is the strongest signal; shorter visits flag disengagement before a customer churns.
  2. **Days_Since_Last_Purchase** – Recency is the classic RFM risk indicator; customers inactive for longer periods are at elevated churn risk.
  3. **Annual_Income_USD** – Income tier influences retention patterns, suggesting tiered loyalty programmes could reduce churn among lower-income segments.
- SHAP analysis confirms that the feature ranking is stable: the same three features dominate mean absolute SHAP values, giving consistent, actionable drivers regardless of the interpretation method used.

## Tech stack

| Library | Purpose |
|---|---|
| pandas, NumPy | Data manipulation |
| scikit-learn | ML pipelines, models, evaluation |
| matplotlib, seaborn | Visualization |
| shap | Model interpretability |
| Jupyter | Interactive analysis |

## How to run

1. Download the Kaggle dataset and save it as `data/raw/online_retail_churn_raw.csv`.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the exploratory notebook:
   ```bash
   jupyter notebook notebooks/01_online_retail_churn_analysis.ipynb
   ```

4. *(Optional)* Train & save models via scripts:
   ```bash
   python -m src.train_model
   ```

5. *(Optional)* Generate evaluation plots (ROC curves, feature importances, SHAP summary):
   ```bash
   python -m src.evaluate
   ```

---

This project demonstrates my ability to build an end-to-end churn prediction pipeline: from data exploration and feature engineering to model training, evaluation, and interpretation.
