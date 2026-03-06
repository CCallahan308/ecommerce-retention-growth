import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def prep_targets(transactions: pd.DataFrame, cutoff_date: datetime) -> pd.DataFrame:
    """
    Generates legacy heuristic churn targets (Python port of KKBox's Scala churn labeler).
    Filters exclusively for active users at the cutoff date to prevent data leakage and 
    artificial class imbalance.

    Parameters
    ----------
    transactions : pd.DataFrame
        DataFrame containing transaction history.
    cutoff_date : datetime
        The cutoff date for evaluating historical status without leaking future information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'msno' and 'is_churn' target labels indicating user churn status.
    """
    tx_before = transactions[transactions["transaction_date"] <= cutoff_date].copy()
    if tx_before.empty:
        return pd.DataFrame(columns=["msno", "is_churn"])

    tx_before = tx_before.sort_values(
        ["msno", "transaction_date", "membership_expire_date", "is_cancel"],
        ascending=[True, True, True, True],
    )
    last_tx = tx_before.groupby("msno").last().reset_index()

    # FILTER: Only target users who are active (or in their 30-day grace period) at the cutoff date.
    # Users whose membership expired > 30 days before the cutoff have definitively churned in the past
    # and should not be included in the active prediction cohort.
    grace_period_start = cutoff_date - pd.Timedelta(days=30)
    last_tx = last_tx[last_tx["membership_expire_date"] > grace_period_start].copy()
    
    if last_tx.empty:
        return pd.DataFrame(columns=["msno", "is_churn"])

    tx_after = transactions[transactions["transaction_date"] > cutoff_date].copy()
    
    # Initialize churn map with 1 (churned) for all eligible users
    user_churn = pd.DataFrame({"msno": last_tx["msno"], "is_churn": 1})
    
    if tx_after.empty:
        return user_churn

    tx_after = tx_after.sort_values(
        ["msno", "transaction_date", "membership_expire_date", "is_cancel"],
        ascending=[True, True, True, True],
    )

    last_expire_df = last_tx[["msno", "membership_expire_date"]].rename(
        columns={"membership_expire_date": "last_expire"}
    )
    
    tx_merged = tx_after.merge(last_expire_df, on="msno", how="inner")
    if tx_merged.empty:
        return user_churn

    tx_merged["running_min_expire"] = tx_merged.groupby("msno")["membership_expire_date"].cummin()
    shifted_min_expire = tx_merged.groupby("msno")["running_min_expire"].shift(1)
    
    tx_merged["effective_expire"] = tx_merged["last_expire"]
    mask = shifted_min_expire.notna() & (shifted_min_expire < tx_merged["effective_expire"])
    tx_merged.loc[mask, "effective_expire"] = shifted_min_expire[mask]
    
    tx_merged["gap"] = (tx_merged["transaction_date"] - tx_merged["effective_expire"]).dt.days
    
    first_non_cancel = tx_merged[tx_merged["is_cancel"] == 0].groupby("msno").first().reset_index()
    first_non_cancel["churn_update"] = (first_non_cancel["gap"] >= 30).astype(int)
    
    user_churn = user_churn.merge(
        first_non_cancel[["msno", "churn_update"]], 
        on="msno", 
        how="left"
    )
    
    user_churn["is_churn"] = user_churn["churn_update"].fillna(user_churn["is_churn"]).astype(int)
    user_churn = user_churn.drop(columns=["churn_update"])
    
    return user_churn


class RFMFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer to generate RFM features from transactions.

    Parameters
    ----------
    transactions : pd.DataFrame
        The transactions dataset containing billing and plan history.
    cutoff_date : datetime
        The date to use as the cutoff for generating features to prevent data leakage.
    """
    def __init__(self, transactions: pd.DataFrame, cutoff_date: datetime):
        self.transactions = transactions
        self.cutoff_date = cutoff_date

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RFMFeatureTransformer":
        """
        Fit the transformer (no-op).

        Parameters
        ----------
        X : pd.DataFrame
            The input data containing 'msno' columns.
        y : pd.Series, optional
            The target labels, by default None.

        Returns
        -------
        RFMFeatureTransformer
            The fitted transformer instance.
        """
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "is_fitted_") and self.is_fitted_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate RFM features and merge with the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input data containing 'msno' columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with RFM features appended.
        """
        rfm = build_rfm_features(self.transactions, self.cutoff_date)
        rfm_cols = [c for c in rfm.columns if c != "msno"]
        X_clean = X.drop(columns=[c for c in rfm_cols if c in X.columns])
        res = pd.merge(X_clean, rfm, on="msno", how="left")
        res[rfm_cols] = res[rfm_cols].fillna(0)
        return res


def build_rfm_features(transactions: pd.DataFrame, cutoff_date: datetime) -> pd.DataFrame:
    """
    Compute RFM from billing history, masking data strictly past the cutoff date.

    Parameters
    ----------
    transactions : pd.DataFrame
        DataFrame of user transactions.
    cutoff_date : datetime
        Global cutoff boundary to prevent data leakage.

    Returns
    -------
    pd.DataFrame
        DataFrame of basic RFM features.
    """
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


class EngagementFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer to generate engagement features from user logs.

    Parameters
    ----------
    user_logs : pd.DataFrame
        The user logs dataset containing daily listening statistics.
    cutoff_date : datetime
        The date to use as the cutoff for generating features to prevent data leakage.
    """
    def __init__(self, user_logs: pd.DataFrame, cutoff_date: datetime):
        self.user_logs = user_logs
        self.cutoff_date = cutoff_date

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "EngagementFeatureTransformer":
        """
        Fit the transformer (no-op).

        Parameters
        ----------
        X : pd.DataFrame
            The input data containing 'msno' columns.
        y : pd.Series, optional
            The target labels, by default None.

        Returns
        -------
        EngagementFeatureTransformer
            The fitted transformer instance.
        """
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "is_fitted_") and self.is_fitted_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate engagement features and merge with the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input data containing 'msno' columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with engagement features appended.
        """
        eng = build_engagement_features(self.user_logs, self.cutoff_date)
        eng_cols = [c for c in eng.columns if c != "msno"]
        X_clean = X.drop(columns=[c for c in eng_cols if c in X.columns])
        res = pd.merge(X_clean, eng, on="msno", how="left")
        res[eng_cols] = res[eng_cols].fillna(0)
        return res


def build_engagement_features(user_logs: pd.DataFrame, cutoff_date: datetime) -> pd.DataFrame:
    """
    Compute listening engagement features over 30d/60d rolling windows from the cutoff.

    Parameters
    ----------
    user_logs : pd.DataFrame
        DataFrame of raw daily listening logs.
    cutoff_date : datetime
        Global cutoff boundary to prevent data leakage.

    Returns
    -------
    pd.DataFrame
        DataFrame of extracted engagement trends per user.
    """
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Legacy path using heuristic targets for early exploratory baselines.
    Generates the base DataFrame (demographics and tenure) to be passed into a scikit-learn Pipeline.
    
    Parameters
    ----------
    members : pd.DataFrame
        User demographic DataFrame.
    transactions : pd.DataFrame
        User transaction history DataFrame.
    user_logs : pd.DataFrame
        User logging history DataFrame.
    cutoff_date : datetime
        Temporal cutoff restricting future data leakage.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (X, y) containing base features and labels.
    """
    targets = prep_targets(transactions, cutoff_date)

    base = pd.merge(targets[["msno"]], members, on="msno", how="left")
    
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
