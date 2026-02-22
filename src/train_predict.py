"""
Train on official Kaggle train.csv labels, then predict on remaining members.

Uses the KKBox competition ground-truth churn labels for training instead of
the heuristic ``prep_targets()`` labeler.  After evaluation on a held-out
portion of the labeled data, the best model predicts churn probabilities for
every member *not* in train.csv.

Usage::

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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_train_labels(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load official Kaggle train.csv with ground-truth churn labels."""
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
    Build features for every member who has at least *some* activity.

    Members with zero transactions **and** zero log entries are excluded
    because there is no signal to learn from.
    """
    logger.info("Building RFM features …")
    rfm = build_rfm_features(transactions, cutoff_date)

    logger.info("Building engagement features …")
    eng = build_engagement_features(user_logs, cutoff_date)

    # Users who appear in at least one of the two feature tables
    active_msnos = set(rfm["msno"]).union(set(eng["msno"]))
    logger.info("Users with activity: %s", f"{len(active_msnos):,}")

    base = members[members["msno"].isin(active_msnos)].copy()
    base = pd.merge(base, rfm, on="msno", how="left")
    base = pd.merge(base, eng, on="msno", how="left")

    # Tenure feature
    base["tenure_days"] = (cutoff_date - base["registration_init_time"]).dt.days
    base["tenure_days"] = base["tenure_days"].fillna(0).clip(lower=0)

    base = base.drop(columns=["registration_init_time"])

    # Fill numeric NaNs with 0; leave categoricals alone (already handled by loader)
    numeric_cols = base.select_dtypes(include=["number"]).columns
    base[numeric_cols] = base[numeric_cols].fillna(0)
    return base


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()

    # ── 1. Load raw data ──────────────────────────────────────────────────
    logger.info("Loading datasets …")
    members, transactions, user_logs = load_all_data()
    train_labels = load_train_labels()

    # ── 2. Feature cutoff ─────────────────────────────────────────────────
    max_date = transactions["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=30)
    logger.info("Feature cutoff date: %s", cutoff)

    # ── 3. Build features for every active member ─────────────────────────
    all_features = build_all_features(members, transactions, user_logs, cutoff)
    logger.info("Feature matrix: %s rows × %s cols", f"{len(all_features):,}", all_features.shape[1])

    # ── 4. Separate labeled (train) vs unlabeled (predict) ────────────────
    train_msno = set(train_labels["msno"])
    labeled_mask = all_features["msno"].isin(train_msno)

    train_features = all_features[labeled_mask].copy()
    predict_features = all_features[~labeled_mask].copy()

    # Attach ground-truth labels
    train_df = pd.merge(train_features, train_labels, on="msno", how="inner")

    logger.info("Labeled users with features: %s", f"{len(train_df):,}")
    logger.info("Unlabeled users to predict:  %s", f"{len(predict_features):,}")

    # ── 5. Train / eval split (80/20 stratified) ─────────────────────────
    X = train_df.drop(columns=["is_churn"])
    y = train_df[["msno", "is_churn"]]

    X_train, X_test, y_train, y_test = get_splits(X, y)

    # ── 6. Train models ──────────────────────────────────────────────────
    prep = make_pipeline(X_train)
    lr_model, xgb_model = train_models(X_train, y_train, prep)

    # ── 7. Evaluate on held-out labeled users ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("EVALUATION ON HELD-OUT LABELED DATA (20%)")
    print(f"{'=' * 60}")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")  # noqa: F841
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")  # noqa: F841

    # ── 8. Predict on unlabeled members ──────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    if len(predict_features) > 0:
        X_predict = predict_features.drop(columns=["msno"])
        proba = xgb_model.predict_proba(X_predict)[:, 1]

        results = pd.DataFrame({
            "msno": predict_features["msno"].values,
            "churn_probability": proba,
            "predicted_churn": (proba >= 0.5).astype(int),
        })

        pred_path = os.path.join(out_dir, "predictions.csv")
        results.to_csv(pred_path, index=False)

        print(f"\n{'=' * 60}")
        print("PREDICTIONS ON UNLABELED USERS")
        print(f"{'=' * 60}")
        print(f"Total predicted:  {len(results):,}")
        print(f"Predicted churn:  {results['predicted_churn'].sum():,}"
              f"  ({results['predicted_churn'].mean():.2%})")
        print(f"Avg probability:  {results['churn_probability'].mean():.4f}")
        print(f"Saved to: {pred_path}")
    else:
        logger.info("No unlabeled users to predict on.")

    # ── 9. Also save the full train-set predictions for analysis ─────────
    X_train_full = train_df.drop(columns=["is_churn", "msno"])
    train_proba = xgb_model.predict_proba(X_train_full)[:, 1]

    train_results = pd.DataFrame({
        "msno": train_df["msno"].values,
        "is_churn": train_df["is_churn"].values,
        "churn_probability": train_proba,
        "predicted_churn": (train_proba >= 0.5).astype(int),
    })
    train_pred_path = os.path.join(out_dir, "train_predictions.csv")
    train_results.to_csv(train_pred_path, index=False)
    print(f"\nTrain-set predictions saved to: {train_pred_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
