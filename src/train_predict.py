"""
Train churn models on the labeled cohort, evaluate on a held-out split, then
score the remaining *active* users and persist the model + a metrics card.

Usage:
    $env:PYTHONPATH = "." ; python src/train_predict.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scipy.stats import ks_2samp
from sklearn.pipeline import Pipeline

from src.config import CHURN_WINDOW_DAYS, CV_FOLDS, RANDOM_STATE, TEST_SIZE
from src.data_loader import DATA_DIR, load_all_data
from src.features import (
    EngagementFeatureTransformer,
    RFMFeatureTransformer,
    active_at_cutoff_msnos,
)
from src.models import (
    calibrate_model,
    evaluate_model,
    get_splits,
    make_pipeline,
    train_models,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "reports")


def log_section(title: str) -> None:
    """Log a visually delimited section header."""
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def load_train_labels(filepath: str | None = None) -> pd.DataFrame:
    """
    Load training labels (msno, is_churn).

    Parameters
    ----------
    filepath : str, optional
        Path to an explicit train CSV file; defaults to ``data/raw/train.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'msno' strings and 'is_churn' labels.
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
    Build the base row per active member (demographics + tenure) before the
    RFM/engagement feature joins performed by the sklearn pipeline.

    Parameters
    ----------
    members : pd.DataFrame
        User demographics.
    transactions : pd.DataFrame
        User transaction history.
    user_logs : pd.DataFrame
        User engagement signals.
    cutoff_date : pd.Timestamp
        Cutoff boundary; tenure is measured up to this date.

    Returns
    -------
    pd.DataFrame
        One row per member with any transaction or log activity.
    """
    active_msnos = set(transactions["msno"]).union(set(user_logs["msno"]))
    logger.info("Users with activity: %s", f"{len(active_msnos):,}")

    base = members[members["msno"].isin(active_msnos)].copy()
    base["tenure_days"] = (cutoff_date - base["registration_init_time"]).dt.days
    base["tenure_days"] = base["tenure_days"].fillna(0).clip(lower=0)

    return base.drop(columns=["registration_init_time"])


def check_feature_drift(
    train_features: pd.DataFrame,
    predict_features: pd.DataFrame,
    threshold: float = 0.05,
) -> bool:
    """
    Detect distribution drift between the training and scoring cohorts using a
    two-sample Kolmogorov-Smirnov test per numeric feature.

    Parameters
    ----------
    train_features : pd.DataFrame
        Engineered features for the training cohort.
    predict_features : pd.DataFrame
        Engineered features for the scoring cohort.
    threshold : float
        Significance level; a feature drifts if its KS p-value falls below it.

    Returns
    -------
    bool
        True if any feature drifted significantly.
    """
    logger.info("Checking for feature distribution drift...")
    numeric_cols = train_features.select_dtypes(include=["number"]).columns

    drift_detected = False
    for col in numeric_cols:
        if col in ("is_churn", "msno"):
            continue

        train_vals = train_features[col].dropna()
        pred_vals = predict_features[col].dropna()
        if len(train_vals) == 0 or len(pred_vals) == 0:
            continue

        statistic, p_value = ks_2samp(train_vals, pred_vals)
        if p_value < threshold:
            drift_detected = True
            logger.warning(
                "Drift detected in feature '%s': KS=%.4f, p=%.4e",
                col,
                statistic,
                p_value,
            )

    if drift_detected:
        logger.warning("Feature drift detected; monitor scoring quality.")
    else:
        logger.info("No significant feature drift detected.")
    return drift_detected


def score_cohort(model: Any, base: pd.DataFrame) -> pd.DataFrame:
    """
    Produce churn probabilities and 0.5-threshold predictions for a cohort.

    ``msno`` is kept in ``base`` deliberately: the pipeline's ColumnTransformer
    excludes it from the feature matrix via a regex, so it survives only as a
    join key on the output.

    Parameters
    ----------
    model : Any
        Fitted estimator exposing ``predict_proba``.
    base : pd.DataFrame
        Base feature rows (including 'msno').

    Returns
    -------
    pd.DataFrame
        Columns: msno, churn_probability, predicted_churn.
    """
    proba = model.predict_proba(base)[:, 1]
    return pd.DataFrame(
        {
            "msno": base["msno"].values,
            "churn_probability": proba,
            "predicted_churn": (proba >= 0.5).astype(int),
        }
    )


def write_model_card(path: str, metadata: dict[str, Any]) -> None:
    """Write run metadata + metrics to a JSON model card (committed evidence)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
        f.write("\n")
    logger.info("Wrote model card to %s", path)


def main() -> None:
    t0 = time.perf_counter()

    logger.info("Loading datasets...")
    members, transactions, user_logs = load_all_data()
    train_labels = load_train_labels()

    # Self-describe the data source: synthetic msno are short "U#####" ids, real
    # KKBox msno are long base64 hashes.
    sample_msno = str(members["msno"].iloc[0]) if len(members) else ""
    data_source = (
        "synthetic"
        if sample_msno.startswith("U") and len(sample_msno) <= 8
        else "KKBox v2 (real)"
    )
    logger.info("Data source: %s", data_source)

    max_date = transactions["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    logger.info("Feature cutoff date: %s", cutoff)

    feature_pipeline = Pipeline(
        [
            ("rfm", RFMFeatureTransformer(transactions, cutoff)),
            ("eng", EngagementFeatureTransformer(user_logs, cutoff)),
        ]
    )

    base_features = build_base_features(members, transactions, user_logs, cutoff)

    # Training cohort = labeled users. Scoring cohort = unlabeled users who are
    # ALSO active at the cutoff, i.e. the same population the model was trained
    # on. Scoring long-dormant users (a different feature distribution) yields
    # degenerate near-certain-churn predictions, so we exclude them here.
    train_msno = set(train_labels["msno"])
    active_msnos = active_at_cutoff_msnos(transactions, cutoff)
    labeled_mask = base_features["msno"].isin(train_msno)
    active_mask = base_features["msno"].isin(active_msnos)

    train_base = base_features[labeled_mask].copy()
    predict_base = base_features[~labeled_mask & active_mask].copy()

    train_df = pd.merge(train_base, train_labels, on="msno", how="inner")
    logger.info("Labeled users with features:     %s", f"{len(train_df):,}")
    logger.info("Active unlabeled users to score: %s", f"{len(predict_base):,}")

    X = train_df.drop(columns=["is_churn"])
    y = train_df[["msno", "is_churn"]]
    X_train, X_test, y_train, y_test = get_splits(X, y)

    prep = make_pipeline()
    lr_model, xgb_model = train_models(X_train, y_train, feature_pipeline, prep)

    log_section("EVALUATION ON HELD-OUT TEST SPLIT (20%)")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression (baseline)")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Calibrate probabilities — the ROI simulator multiplies them by lifetime
    # value, so they must mean what they say, not merely rank correctly.
    calibrated_xgb = calibrate_model(xgb_model, X_train, y_train)
    cal_metrics = evaluate_model(
        calibrated_xgb, X_test, y_test, "XGBoost (calibrated)"
    )
    final_model = calibrated_xgb

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if len(predict_base) > 0:
        log_section("SCORING ACTIVE UNLABELED USERS")
        # feature_pipeline was fitted inside train_models (same instance), so
        # transform here materializes the exact features the model consumed.
        check_feature_drift(
            feature_pipeline.transform(train_base),
            feature_pipeline.transform(predict_base),
        )
        results = score_cohort(final_model, predict_base)
        pred_path = os.path.join(PROCESSED_DIR, "predictions.csv")
        results.to_csv(pred_path, index=False)
        logger.info("Scored %s users", f"{len(results):,}")
        logger.info(
            "Predicted churn: %s (%.2f%%)",
            f"{int(results['predicted_churn'].sum()):,}",
            results["predicted_churn"].mean() * 100,
        )
        logger.info("Mean probability: %.4f", results["churn_probability"].mean())
        logger.info("Saved to %s", pred_path)
    else:
        logger.info(
            "No active unlabeled users to score "
            "(every active-at-cutoff user is already labeled)."
        )

    # Save train-set predictions for error analysis.
    train_results = score_cohort(final_model, train_df.drop(columns=["is_churn"]))
    train_results.insert(1, "is_churn", train_df["is_churn"].values)
    train_pred_path = os.path.join(PROCESSED_DIR, "train_predictions.csv")
    train_results.to_csv(train_pred_path, index=False)
    logger.info("Train-set predictions saved to %s", train_pred_path)

    # Persist the shipped model (gitignored binary) + a committed metrics card.
    model_path = os.path.join(PROCESSED_DIR, "model.joblib")
    joblib.dump(final_model, model_path)
    logger.info("Saved fitted model to %s", model_path)

    classifier = xgb_model.named_steps["classifier"]
    tuned = {
        k: classifier.get_params()[k]
        for k in ("max_depth", "learning_rate", "n_estimators", "subsample")
    }
    write_model_card(
        os.path.join(REPORTS_DIR, "metrics.json"),
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": "30-day subscription churn (binary classification)",
            "config": {
                "churn_window_days": CHURN_WINDOW_DAYS,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "cv_folds": CV_FOLDS,
            },
            "data": {
                "source": data_source,
                "feature_cutoff": str(cutoff),
                "n_labeled": len(train_df),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "churn_rate": round(float(train_labels["is_churn"].mean()), 4),
            },
            "best_xgboost_params": {
                "max_depth": int(tuned["max_depth"]),
                "learning_rate": float(tuned["learning_rate"]),
                "n_estimators": int(tuned["n_estimators"]),
                "subsample": float(tuned["subsample"]),
            },
            "test_metrics": {
                "logistic_regression": lr_metrics,
                "xgboost": xgb_metrics,
                "xgboost_calibrated": cal_metrics,
            },
        },
    )

    logger.info("Done in %.1f minutes.", (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
