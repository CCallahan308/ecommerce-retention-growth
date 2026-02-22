import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prep_targets(transactions: pd.DataFrame, cutoff_date: datetime):
    # 1. Identify the 'last_expire' for each user as of cutoff_date
    tx_before = transactions[transactions["transaction_date"] <= cutoff_date].copy()
    if tx_before.empty:
        return pd.DataFrame(columns=["msno", "is_churn"])

    # Sorting to match Scala logic's record selection
    tx_before = tx_before.sort_values(
        ["msno", "transaction_date", "membership_expire_date", "is_cancel"],
        ascending=[True, True, True, True],
    )
    last_tx = (
        tx_before.groupby("msno").last().reset_index()
    )  # Takes the most recent expire

    # 2. Get future transactions to calculate the renewal gap
    tx_after = transactions[transactions["transaction_date"] > cutoff_date].copy()
    tx_after = tx_after.sort_values(
        ["msno", "transaction_date", "membership_expire_date", "is_cancel"],
        ascending=[True, True, True, True],
    )

    # We will compute the gap for each user
    user_expire_map = last_tx.set_index("msno")["membership_expire_date"].to_dict()
    user_churn_map = {}

    # All users who had data before cutoff start as potential churn (if no future data)
    msnos_with_future = set(tx_after["msno"].unique())

    # Pre-calculate churn for users with no future activity
    for msno in user_expire_map.keys():
        if msno not in msnos_with_future:
            user_churn_map[msno] = 1

    # For users with activity, calculate the specific renewal gap
    if not tx_after.empty:
        # Iterate per-user to calculate renewal gaps
        for msno, group in tx_after.groupby("msno"):
            if msno not in user_expire_map:
                continue

            last_expire = user_expire_map[msno]
            gap = 9999  # Sentinel for no renewal found

            for row in group.itertuples():
                if row.is_cancel == 1:
                    # Update expiration if cancellation moves it earlier
                    if row.membership_expire_date < last_expire:
                        last_expire = row.membership_expire_date
                else:
                    # Found a renewal: calculate gap from last (possibly updated) expire
                    gap = (row.transaction_date - last_expire).days
                    break

            user_churn_map[msno] = 1 if gap >= 30 else 0

    targets = pd.DataFrame(list(user_churn_map.items()), columns=["msno", "is_churn"])
    return targets


def build_rfm_features(transactions: pd.DataFrame, cutoff_date: datetime):
    """RFM from transaction history."""
    # filter out future data
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
    """Engagement from logs - 30d and 60d windows."""
    logs_hist = user_logs[user_logs["date"] <= cutoff_date].copy()

    logs_30d = pd.DataFrame(
        logs_hist[logs_hist["date"] > (cutoff_date - pd.Timedelta(days=30))]
    )
    logs_60d = pd.DataFrame(
        logs_hist[logs_hist["date"] > (cutoff_date - pd.Timedelta(days=60))]
    )

    agg_30d = (
        logs_30d.groupby("msno")
        .agg(
            total_secs_30d=("total_secs", "sum"),
            active_days_30d=("date", "nunique"),
            unique_songs_30d=("num_unq", "sum"),
        )
        .reset_index()
    )

    agg_60d = (
        logs_60d.groupby("msno")
        .agg(total_secs_60d=("total_secs", "sum"), active_days_60d=("date", "nunique"))
        .reset_index()
    )

    eng = pd.merge(agg_60d, agg_30d, on="msno", how="left").fillna(0)

    # Ratio of 30d to 60d average
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
    """Main feature pipeline. Returns X, y."""
    # TODO: add more demographic features if needed - city/registered_via aren't used currently
    # Targets
    targets = prep_targets(transactions, cutoff_date)

    # Features
    rfm = build_rfm_features(transactions, cutoff_date)
    eng = build_engagement_features(user_logs, cutoff_date)

    base = pd.merge(targets[["msno"]], members, on="msno", how="left")
    base = pd.merge(base, rfm, on="msno", how="left")
    base = pd.merge(base, eng, on="msno", how="left")

    # Handle Demographics Features
    base["tenure_days"] = (cutoff_date - base["registration_init_time"]).dt.days
    base["tenure_days"] = base["tenure_days"].fillna(0).clip(lower=0)

    features_to_drop = ["registration_init_time"]
    X = pd.DataFrame(base.drop(columns=features_to_drop))
    y = pd.DataFrame(targets[["msno", "is_churn"]])

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
