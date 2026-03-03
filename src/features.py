import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prep_targets(transactions: pd.DataFrame, cutoff_date: datetime):
    """
    Python port of KKBox's Scala churn labeler.

    Main pipeline uses pre-computed train.csv labels directly - this is here
    for reference and the legacy models.py path.
    """
    tx_before = transactions[transactions["transaction_date"] <= cutoff_date].copy()
    if tx_before.empty:
        return pd.DataFrame(columns=["msno", "is_churn"])

    tx_before = tx_before.sort_values(
        ["msno", "transaction_date", "membership_expire_date", "is_cancel"],
        ascending=[True, True, True, True],
    )
    last_tx = tx_before.groupby("msno").last().reset_index()

    tx_after = transactions[transactions["transaction_date"] > cutoff_date].copy()
    # If no data after cutoff, everyone churns
    if tx_after.empty:
        return pd.DataFrame({"msno": last_tx["msno"], "is_churn": 1})

    tx_after = tx_after.sort_values(
        ["msno", "transaction_date", "membership_expire_date", "is_cancel"],
        ascending=[True, True, True, True],
    )

    # Initialize churn map with 1 (churned) for all users active before cutoff
    user_churn = pd.DataFrame({"msno": last_tx["msno"], "is_churn": 1})
    
    # Get last expiry from before cutoff
    last_expire_df = last_tx[["msno", "membership_expire_date"]].rename(
        columns={"membership_expire_date": "last_expire"}
    )
    
    # Filter to only users who had tx before cutoff
    tx_merged = tx_after.merge(last_expire_df, on="msno", how="inner")
    if tx_merged.empty:
        return user_churn

    # Calculate gap to next transaction
    tx_merged["gap"] = (tx_merged["transaction_date"] - tx_merged["last_expire"]).dt.days
    
    # Handle cancellations modifying expiry
    # Propagate last expire forward conditionally
    
    # The original logic:
    # 1. Iterate through forward txs.
    # 2. If it's a cancellation AND new expiry < last_expire, update last_expire.
    # 3. If it's NOT a cancellation, gap = (transaction_date - last_expire). break loop.
    # 4. is_churn = 1 if gap >= 30 else 0
    
    # We can vectorize this: find the FIRST non-cancellation transaction for each user.
    # However, cancellations BEFORE that first non-cancellation might lower the expiry date.
    
    # First, let's keep track of the minimum expiry date seen *so far* for each user
    tx_merged["running_min_expire"] = tx_merged.groupby("msno")["membership_expire_date"].cummin()
    
    # The effective expiry for gap calculation is the minimum of (last_expire, running_min_expire_of_PREVIOUS_rows)
    # Actually, if we just find the first non-cancel row, the effective last_expire is the min(last_expire, all previous cancel expiries).
    
    # Let's create a shifted running min expire
    shifted_min_expire = tx_merged.groupby("msno")["running_min_expire"].shift(1)
    
    # The effective expiry date before this transaction is the minimum of original last_expire and the shifted running min
    tx_merged["effective_expire"] = tx_merged["last_expire"]
    mask = shifted_min_expire.notna() & (shifted_min_expire < tx_merged["effective_expire"])
    tx_merged.loc[mask, "effective_expire"] = shifted_min_expire[mask]
    
    # Now calculate gap for all rows
    tx_merged["gap"] = (tx_merged["transaction_date"] - tx_merged["effective_expire"]).dt.days
    
    # Find the FIRST non-cancel row
    first_non_cancel = tx_merged[tx_merged["is_cancel"] == 0].groupby("msno").first().reset_index()
    
    # For these users, update churn status based on gap
    first_non_cancel["churn_update"] = (first_non_cancel["gap"] >= 30).astype(int)
    
    # Update the results
    user_churn = user_churn.merge(
        first_non_cancel[["msno", "churn_update"]], 
        on="msno", 
        how="left"
    )
    
    # If churn_update is present, use it. Otherwise, keep it as 1 (churned) because there was no non-cancel row
    user_churn["is_churn"] = user_churn["churn_update"].fillna(user_churn["is_churn"]).astype(int)
    user_churn = user_churn.drop(columns=["churn_update"])
    
    return user_churn


def build_rfm_features(transactions: pd.DataFrame, cutoff_date: datetime):
    """RFM from billing - cutoff keeps leakage out."""
    tx_hist = transactions[transactions["transaction_date"] <= cutoff_date].copy()

    rfm = (
        tx_hist.groupby("msno")
        .agg(
            recency=("transaction_date", lambda x: (cutoff_date - x.max()).days),
            frequency=("msno", "count"),
            monetary_total=("actual_amount_paid", "sum"),
            monetary_avg=("actual_amount_paid", "mean"),
            auto_renew_ratio=("is_auto_renew", "mean"),
        )
        .reset_index()
    )

    rfm.fillna(0, inplace=True)
    return rfm


def build_engagement_features(user_logs: pd.DataFrame, cutoff_date: datetime):
    logs_hist = user_logs[user_logs["date"] <= cutoff_date].copy()

    logs_30d = logs_hist[logs_hist["date"] > (cutoff_date - pd.Timedelta(days=30))]
    logs_60d = logs_hist[logs_hist["date"] > (cutoff_date - pd.Timedelta(days=60))]

    agg_30d = (
        logs_30d.groupby("msno")
        .agg(
            total_secs_30d=("total_secs", "sum"),
            active_days_30d=("date", "nunique"),
            unique_songs_30d=("num_unq", "sum"),
        )
        .reset_index()
    )

    # TODO: could add 90d window for longer-term patterns
    agg_60d = (
        logs_60d.groupby("msno")
        .agg(total_secs_60d=("total_secs", "sum"), active_days_60d=("date", "nunique"))
        .reset_index()
    )

    eng = pd.merge(agg_60d, agg_30d, on="msno", how="left").fillna(0)

    eng["secs_trend"] = np.where(
        eng["total_secs_60d"] > 0,
        (eng["total_secs_30d"] * 2) / eng["total_secs_60d"],
        0,
    )
    return eng


def engineer_features(
    members: pd.DataFrame,
    transactions: pd.DataFrame,
    user_logs: pd.DataFrame,
    cutoff_date: datetime,
):
    """Legacy path - uses heuristic labels. For Kaggle labels see train_predict.py"""
    targets = prep_targets(transactions, cutoff_date)
    rfm = build_rfm_features(transactions, cutoff_date)
    eng = build_engagement_features(user_logs, cutoff_date)

    base = pd.merge(targets[["msno"]], members, on="msno", how="left")
    base = pd.merge(base, rfm, on="msno", how="left")
    base = pd.merge(base, eng, on="msno", how="left")

    base["tenure_days"] = (cutoff_date - base["registration_init_time"]).dt.days
    base["tenure_days"] = base["tenure_days"].fillna(0).clip(lower=0)

    X = base.drop(columns=["registration_init_time"])
    y = targets[["msno", "is_churn"]]

    return X, y  # type: ignore


if __name__ == "__main__":
    from src.data_loader import load_all_data

    m, t, u = load_all_data()

    # Set cutoff dynamically: 30 days before latest transaction
    max_date = t["transaction_date"].max()
    CUTOFF = max_date - pd.Timedelta(days=30)
    logger.info(f"Engineering features with cutoff: {CUTOFF}")

    X, y = engineer_features(m, t, u, CUTOFF)

    # Save outputs for modeling
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "features"
    )
    os.makedirs(out_dir, exist_ok=True)

    X.to_csv(os.path.join(out_dir, "X.csv"), index=False)
    y.to_csv(os.path.join(out_dir, "y.csv"), index=False)
    logger.info(f"Saved processed features to {out_dir}")
