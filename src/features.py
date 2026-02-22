"""
Feature Engineering Module.

This module constructs historical customer features (RFM and Engagement) and defines
the target variable for predictive modeling, enforcing strict temporal cutoffs to
prevent target leakage. Vectorized pandas operations are prioritized.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_target_labels(transactions: pd.DataFrame, cutoff_date: datetime) -> pd.DataFrame:
    """
    Define churn target based on a temporal cutoff.

    A user is considered churned (1) if their latest membership expiration
    date before the cutoff passes without a subsequent transaction, or if
    they explicitly cancelled.

    Parameters
    ----------
    transactions : pd.DataFrame
        Cleaned transactions dataframe.
    cutoff_date : datetime
        The date separating historical features from future outcomes.

    Returns
    -------
    pd.DataFrame
        DataFrame containing `msno` and `is_churn` (1 or 0).

    """
    logger.info(f"Building target labels with cutoff date: {cutoff_date}")

    # Sort transactions by date
    tx = transactions.sort_values(["msno", "transaction_date"]).copy()

    # Find the last transaction for each user before or on the cutoff date
    tx_hist = tx[tx["transaction_date"] <= cutoff_date].groupby("msno").last().reset_index()

    # Find if there's any transaction after the expiration date + 30 days (grace period)
    # For simplicity in this mock-up, we'll use `is_cancel` from the historical
    # record as the primary churn indicator, or if their expiration is in the past.

    tx_hist["is_churn"] = np.where(
        (tx_hist["is_cancel"] == 1) | (tx_hist["membership_expire_date"] <= cutoff_date),
        1,
        0
    )

    return pd.DataFrame(tx_hist[["msno", "is_churn"]])


def build_rfm_features(transactions: pd.DataFrame, cutoff_date: datetime) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary (RFM) features for each user.

    Parameters
    ----------
    transactions : pd.DataFrame
        Cleaned transactions dataframe.
    cutoff_date : datetime
        The date separating historical features from future outcomes.

    Returns
    -------
    pd.DataFrame
        DataFrame containing `msno` and aggregated RFM features.

    """
    logger.info("Building RFM features.")

    # Filter out future data to prevent leakage
    tx_hist = transactions[transactions["transaction_date"] <= cutoff_date].copy()

    rfm = tx_hist.groupby("msno").agg(
        recency=("transaction_date", lambda x: (cutoff_date - x.max()).days),
        frequency=("msno", "count"),
        monetary_total=("actual_amount_paid", "sum"),
        monetary_avg=("actual_amount_paid", "mean"),
        auto_renew_ratio=("is_auto_renew", "mean"),
    ).reset_index()

    # Fill any NaNs that might occur
    rfm.fillna(0, inplace=True)
    return rfm


def build_engagement_features(user_logs: pd.DataFrame, cutoff_date: datetime) -> pd.DataFrame:
    """
    Calculate usage intensity and trends over the last 30 and 60 days.

    Parameters
    ----------
    user_logs : pd.DataFrame
        Cleaned user logs dataframe.
    cutoff_date : datetime
        The date separating historical features from future outcomes.

    Returns
    -------
    pd.DataFrame
        DataFrame containing `msno` and engagement metrics.

    """
    logger.info("Building engagement features.")

    # Filter logs prior to cutoff
    logs_hist = user_logs[user_logs["date"] <= cutoff_date].copy()

    # Windows
    logs_30d = pd.DataFrame(logs_hist[logs_hist["date"] > (cutoff_date - pd.Timedelta(days=30))])
    logs_60d = pd.DataFrame(logs_hist[logs_hist["date"] > (cutoff_date - pd.Timedelta(days=60))])

    # Aggregations for 30 days
    agg_30d = logs_30d.groupby("msno").agg(
        total_secs_30d=("total_secs", "sum"),
        active_days_30d=("date", "nunique"),
        unique_songs_30d=("num_unq", "sum")
    ).reset_index()

    # Aggregations for 60 days
    agg_60d = logs_60d.groupby("msno").agg(
        total_secs_60d=("total_secs", "sum"),
        active_days_60d=("date", "nunique")
    ).reset_index()

    # Merge windows
    engagement = pd.merge(agg_60d, agg_30d, on="msno", how="left").fillna(0)

    # Calculate trend (Ratio of 30d to 60d average)
    engagement["secs_trend"] = np.where(
        engagement["total_secs_60d"] > 0,
        (engagement["total_secs_30d"] * 2) / engagement["total_secs_60d"],
        0
    )

    return engagement


def engineer_features(members: pd.DataFrame, transactions: pd.DataFrame, user_logs: pd.DataFrame, cutoff_date: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master pipeline to join demographics, RFM, engagement, and targets.

    Parameters
    ----------
    members : pd.DataFrame
        Members data.
    transactions : pd.DataFrame
        Transactions data.
    user_logs : pd.DataFrame
        User logs data.
    cutoff_date : datetime
        The date separating historical features from future outcomes.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        X (features dataframe) and y (target dataframe).

    """
    logger.info("Executing master feature engineering pipeline.")

    # Targets
    targets = build_target_labels(transactions, cutoff_date)

    # Features
    rfm = build_rfm_features(transactions, cutoff_date)
    eng = build_engagement_features(user_logs, cutoff_date)

    # Merge all (Left join on targets to keep everyone we have a target for)
    base = pd.merge(targets[["msno"]], members, on="msno", how="left")
    base = pd.merge(base, rfm, on="msno", how="left")
    base = pd.merge(base, eng, on="msno", how="left")

    # Handle Demographics Features (e.g., tenure)
    base["tenure_days"] = (cutoff_date - base["registration_init_time"]).dt.days
    base["tenure_days"] = base["tenure_days"].fillna(0).clip(lower=0)

    # Drop irrelevant/raw identifier columns for X
    # 'city', 'gender', 'registered_via' need one-hot encoding which will happen in the model pipeline
    features_to_drop = ["registration_init_time"]
    X = pd.DataFrame(base.drop(columns=features_to_drop))
    y = pd.DataFrame(targets[["msno", "is_churn"]])

    return X, y  # type: ignore


if __name__ == "__main__":
    from src.data_loader import load_all_data

    m, t, u = load_all_data()

    # Set CUTOFF dynamically based on the dataset's date range
    # Mock data is 2023-2024. Kaggle data is early 2017.
    max_date = t["transaction_date"].max()
    # Go back 30 days from the maximum date found in the dataset to create a realistic holdout period
    CUTOFF = max_date - pd.Timedelta(days=30)

    X, y = engineer_features(m, t, u, CUTOFF)

    logger.info(f"Generated Feature Matrix Shape: {X.shape}")
    logger.info(f"Target Distribution:\n{y['is_churn'].value_counts(normalize=True)}")
