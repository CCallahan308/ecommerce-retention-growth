"""
Train on official Kaggle train.csv labels, then predict on remaining members.

Usage:
    $env:PYTHONPATH = "." ; python src/train_predict.py
"""

import logging
import os
import sys
import time
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import DATA_DIR, load_all_data
from src.features import RFMFeatureTransformer, EngagementFeatureTransformer
from src.models import evaluate_model, get_splits, make_pipeline, train_models
from sklearn.pipeline import Pipeline
from scipy.stats import ks_2samp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_train_labels(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load official Kaggle training labels.

    Parameters
    ----------
    filepath : str, optional
        Path to explicit train CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame populated with 'msno' strings and 'is_churn' boolean label status.
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "train.csv")
    df = pd.read_csv(filepath, dtype={"msno": "string", "is_churn": "Int8"})
    logger.info("Loaded %s labeled users from train.csv", f"{len(df):,}")
    logger.info("Churn rate: %.2f%%", df["is_churn"].mean() * 100)
    return df


def build_base_features(
    members: pd.DataFrame,
    transactions: pd.DataFrame,
    user_logs: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build structural row definitions for every active member before pipeline feature joins.

    Parameters
    ----------
    members : pd.DataFrame
        DataFrame of user demographics.
    transactions : pd.DataFrame
        DataFrame of user transaction history.
    user_logs : pd.DataFrame
        DataFrame of user engagement signals.
    cutoff_date : pd.Timestamp
        Global cutoff boundary stringing historical data.

    Returns
    -------
    pd.DataFrame
        Base row definitions specifying which users require ML modeling inference scoring.
    """
    active_msnos = set(transactions["msno"]).union(set(user_logs["msno"]))
    logger.info("Users with activity: %s", f"{len(active_msnos):,}")

    base = members[members["msno"].isin(active_msnos)].copy()
    base["tenure_days"] = (cutoff_date - base["registration_init_time"]).dt.days
    base["tenure_days"] = base["tenure_days"].fillna(0).clip(lower=0)

    base = base.drop(columns=["registration_init_time"])
    return base


def check_feature_drift(train_features: pd.DataFrame, predict_features: pd.DataFrame, threshold: float = 0.05) -> None:
    """
    Checks for feature distribution drift using Kolmogorov-Smirnov test.
    Logs warnings for features that have drifted significantly.

    Parameters
    ----------
    train_features: pd.DataFrame
        Training data representation.
    predict_features: pd.DataFrame
        Evaluation data representation unobserved by prior HPO constraints.
    threshold: float
        Acceptance score variance significance alpha-probability.
    """
    logger.info("Checking for feature distribution drift...")
    numeric_cols = train_features.select_dtypes(include=["number"]).columns
    
    drift_detected = False
    for col in numeric_cols:
        # Ignore target variable or IDs if accidentally included
        if col in ["is_churn", "msno"]:
            continue
            
        train_vals = train_features[col].dropna()
        pred_vals = predict_features[col].dropna()
        
        if len(train_vals) == 0 or len(pred_vals) == 0:
            continue
            
        # Perform KS test
        statistic, p_value = ks_2samp(train_vals, pred_vals)
        
        if p_value < threshold:
            drift_detected = True
            logger.warning(
                f"Drift detected in feature '{col}': "
                f"KS Statistic = {statistic:.4f}, p-value = {p_value:.4e}"
            )
            
    if not drift_detected:
        logger.info("No significant feature drift detected.")
    else:
        logger.warning("Feature drift detected. Monitor model performance carefully.")


def main() -> None:
    t0 = time.perf_counter()
    np.random.seed(42)

    logger.info("Loading datasets...")
    members, transactions, user_logs = load_all_data()
    train_labels = load_train_labels()

    max_date = transactions["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=30)
    logger.info("Feature cutoff date: %s", cutoff)

    feature_pipeline = Pipeline([
        ('rfm', RFMFeatureTransformer(transactions, cutoff)),
        ('eng', EngagementFeatureTransformer(user_logs, cutoff)),
    ])

    base_features = build_base_features(members, transactions, user_logs, cutoff)

    # split labeled vs unlabeled
    train_msno = set(train_labels["msno"])
    labeled_mask = base_features["msno"].isin(train_msno)

    train_base = base_features[labeled_mask].copy()
    predict_base = base_features[~labeled_mask].copy()

    train_df = pd.merge(train_base, train_labels, on="msno", how="inner")

    logger.info("Labeled users with features: %s", f"{len(train_df):,}")
    logger.info("Unlabeled users to predict:  %s", f"{len(predict_base):,}")

    X = train_df.drop(columns=["is_churn"])
    y = train_df[["msno", "is_churn"]]

    X_train, X_test, y_train, y_test = get_splits(X, y)

    prep = make_pipeline()
    lr_model, xgb_model = train_models(X_train, y_train, feature_pipeline, prep)

    print(f"\n{'=' * 60}")
    print("EVALUATION ON HELD-OUT LABELED DATA (20%)")
    print(f"{'=' * 60}")
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "processed"
    )
    os.makedirs(out_dir, exist_ok=True)

    if len(predict_base) > 0:
        print(f"\n{'=' * 60}")
        print("PIPELINE VALIDATION")
        print(f"{'=' * 60}")
        
        X_predict = predict_base.drop(columns=["msno"])
        # predict_proba runs pipeline transformation
        proba = xgb_model.predict_proba(X_predict)[:, 1]

        results = pd.DataFrame(
            {
                "msno": predict_base["msno"].values,
                "churn_probability": proba,
                "predicted_churn": (proba >= 0.5).astype(int),
            }
        )

        pred_path = os.path.join(out_dir, "predictions.csv")
        results.to_csv(pred_path, index=False)

        print(f"\n{'=' * 60}")
        print("PREDICTIONS ON UNLABELED USERS")
        print(f"{'=' * 60}")
        print(f"Total predicted:  {len(results):,}")
        print(
            f"Predicted churn:  {results['predicted_churn'].sum():,}"
            f"  ({results['predicted_churn'].mean():.2%})"
        )
        print(f"Avg probability:  {results['churn_probability'].mean():.4f}")
        print(f"Saved to: {pred_path}")
    else:
        logger.info("No unlabeled users to predict on.")

    # also save train predictions for analysis
    X_train_full = train_df.drop(columns=["is_churn", "msno"])
    train_proba = xgb_model.predict_proba(X_train_full)[:, 1]

    train_results = pd.DataFrame(
        {
            "msno": train_df["msno"].values,
            "is_churn": train_df["is_churn"].values,
            "churn_probability": train_proba,
            "predicted_churn": (train_proba >= 0.5).astype(int),
        }
    )
    train_pred_path = os.path.join(out_dir, "train_predictions.csv")
    train_results.to_csv(train_pred_path, index=False)
    print(f"\nTrain-set predictions saved to: {train_pred_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
