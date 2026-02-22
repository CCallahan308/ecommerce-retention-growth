import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def prep_targets(transactions: pd.DataFrame, cutoff_date: datetime):
    # Sort by date so we can grab the last transaction before cutoff
    tx = transactions.sort_values(["msno", "transaction_date"]).copy()
    tx_hist = tx[tx["transaction_date"] <= cutoff_date].groupby("msno").last().reset_index()

    # if they cancelled, or their expiration is before the cutoff, they've churned
    tx_hist["is_churn"] = np.where(
        (tx_hist["is_cancel"] == 1) | (tx_hist["membership_expire_date"] <= cutoff_date),
        1,
        0
    )
    return pd.DataFrame(tx_hist[["msno", "is_churn"]])


def build_rfm_features(transactions: pd.DataFrame, cutoff_date: datetime):
    # filter out future data
    tx_hist = transactions[transactions["transaction_date"] <= cutoff_date].copy()

    rfm = tx_hist.groupby("msno").agg(
        recency=("transaction_date", lambda x: (cutoff_date - x.max()).days),
        frequency=("msno", "count"),
        monetary_total=("actual_amount_paid", "sum"),
        monetary_avg=("actual_amount_paid", "mean"),
        auto_renew_ratio=("is_auto_renew", "mean"),
    ).reset_index()

    rfm.fillna(0, inplace=True)
    return rfm


def build_engagement_features(user_logs: pd.DataFrame, cutoff_date: datetime):
    logs_hist = user_logs[user_logs["date"] <= cutoff_date].copy()

    logs_30d = pd.DataFrame(logs_hist[logs_hist["date"] > (cutoff_date - pd.Timedelta(days=30))])
    logs_60d = pd.DataFrame(logs_hist[logs_hist["date"] > (cutoff_date - pd.Timedelta(days=60))])

    agg_30d = logs_30d.groupby("msno").agg(
        total_secs_30d=("total_secs", "sum"),
        active_days_30d=("date", "nunique"),
        unique_songs_30d=("num_unq", "sum")
    ).reset_index()

    agg_60d = logs_60d.groupby("msno").agg(
        total_secs_60d=("total_secs", "sum"),
        active_days_60d=("date", "nunique")
    ).reset_index()

    eng = pd.merge(agg_60d, agg_30d, on="msno", how="left").fillna(0)

    # Ratio of 30d to 60d average
    eng["secs_trend"] = np.where(
        eng["total_secs_60d"] > 0,
        (eng["total_secs_30d"] * 2) / eng["total_secs_60d"],
        0
    )
    return eng


def engineer_features(members: pd.DataFrame, transactions: pd.DataFrame, user_logs: pd.DataFrame, cutoff_date: datetime):
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

    # we'll OHE these later
    # TODO: maybe add rolling average features if time permits
    features_to_drop = ["registration_init_time"]
    X = pd.DataFrame(base.drop(columns=features_to_drop))
    y = pd.DataFrame(targets[["msno", "is_churn"]])

    return X, y  # type: ignore


if __name__ == "__main__":
    from src.data_loader import load_all_data

    m, t, u = load_all_data()

    # set cutoff dynamically
    max_date = t["transaction_date"].max()
    CUTOFF = max_date - pd.Timedelta(days=30)

    X, y = engineer_features(m, t, u, CUTOFF)
