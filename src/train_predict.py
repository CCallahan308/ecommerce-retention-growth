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
from src.features import build_engagement_features, build_rfm_features
from src.models import evaluate_model, get_splits, make_pipeline, train_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_train_labels(filepath: Optional[str] = None) -> pd.DataFrame:
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "train.csv")
    df = pd.read_csv(filepath, dtype={"msno": "string", "is_churn": "Int8"})
    logger.info("Loaded %s labeled users from train.csv", f"{len(df):,}")
    logger.info("Churn rate: %.2f%%", df["is_churn"].mean() * 100)
    return df


def build_all_features(
    members: pd.DataFrame,
    transactions: pd.DataFrame,
    user_logs: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build features for every member who has at least some activity.
    Members with zero transactions AND zero log entries get skipped
    (nothing to learn from).
    """
    logger.info("Building RFM features...")
    rfm = build_rfm_features(transactions, cutoff_date)

    logger.info("Building engagement features...")
    eng = build_engagement_features(user_logs, cutoff_date)

    active_msnos = set(rfm["msno"]).union(set(eng["msno"]))
    logger.info("Users with activity: %s", f"{len(active_msnos):,}")

    base = members[members["msno"].isin(active_msnos)].copy()
    base = pd.merge(base, rfm, on="msno", how="left")
    base = pd.merge(base, eng, on="msno", how="left")

    base["tenure_days"] = (cutoff_date - base["registration_init_time"]).dt.days
    base["tenure_days"] = base["tenure_days"].fillna(0).clip(lower=0)

    base = base.drop(columns=["registration_init_time"])

    # numeric NaNs -> 0, categoricals already handled upstream
    numeric_cols = base.select_dtypes(include=["number"]).columns
    base[numeric_cols] = base[numeric_cols].fillna(0)
    return base


def main() -> None:
    t0 = time.perf_counter()

    logger.info("Loading datasets...")
    members, transactions, user_logs = load_all_data()
    train_labels = load_train_labels()

    max_date = transactions["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=30)
    logger.info("Feature cutoff date: %s", cutoff)

    all_features = build_all_features(members, transactions, user_logs, cutoff)
    logger.info(
        "Feature matrix: %s rows x %s cols",
        f"{len(all_features):,}",
        all_features.shape[1],
    )

    # split labeled vs unlabeled
    train_msno = set(train_labels["msno"])
    labeled_mask = all_features["msno"].isin(train_msno)

    train_features = all_features[labeled_mask].copy()
    predict_features = all_features[~labeled_mask].copy()

    train_df = pd.merge(train_features, train_labels, on="msno", how="inner")

    logger.info("Labeled users with features: %s", f"{len(train_df):,}")
    logger.info("Unlabeled users to predict:  %s", f"{len(predict_features):,}")

    X = train_df.drop(columns=["is_churn"])
    y = train_df[["msno", "is_churn"]]

    X_train, X_test, y_train, y_test = get_splits(X, y)

    prep = make_pipeline(X_train)
    lr_model, xgb_model = train_models(X_train, y_train, prep)

    print(f"\n{'=' * 60}")
    print("EVALUATION ON HELD-OUT LABELED DATA (20%)")
    print(f"{'=' * 60}")
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "processed"
    )
    os.makedirs(out_dir, exist_ok=True)

    if len(predict_features) > 0:
        X_predict = predict_features.drop(columns=["msno"])
        proba = xgb_model.predict_proba(X_predict)[:, 1]

        results = pd.DataFrame(
            {
                "msno": predict_features["msno"].values,
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
