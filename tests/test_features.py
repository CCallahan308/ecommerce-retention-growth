"""
Tests for the feature engineering pipeline to ensure logic correctness and
guarantee no target leakage from future events.
"""

from datetime import datetime

import pandas as pd
import pytest

from src.features import (
    build_engagement_features,
    build_rfm_features,
    build_target_labels,
)

CUTOFF = datetime(2023, 12, 1)

@pytest.fixture
def mock_transactions():
    return pd.DataFrame(
        {
            "msno": ["U001", "U001", "U002"],
            "transaction_date": [
                pd.to_datetime("2023-10-01"), # Before cutoff
                pd.to_datetime("2023-12-15"), # AFTER cutoff (leakage risk)
                pd.to_datetime("2023-11-01"), # Before cutoff
            ],
            "membership_expire_date": [
                pd.to_datetime("2023-11-01"),
                pd.to_datetime("2024-01-15"),
                pd.to_datetime("2023-11-30"), # Expired before cutoff
            ],
            "actual_amount_paid": [100.0, 150.0, 100.0],
            "is_auto_renew": [1, 1, 0],
            "is_cancel": [0, 0, 1],
        }
    )

@pytest.fixture
def mock_logs():
    return pd.DataFrame(
        {
            "msno": ["U001", "U001", "U002"],
            "date": [
                pd.to_datetime("2023-11-15"), # Within 30d of cutoff
                pd.to_datetime("2023-10-15"), # Within 60d of cutoff
                pd.to_datetime("2023-12-10"), # AFTER cutoff (leakage risk)
            ],
            "total_secs": [3600.0, 1800.0, 7200.0],
            "num_unq": [10, 5, 20],
        }
    )

def test_build_target_labels(mock_transactions):
    """Ensure targets only consider historical facts up to cutoff."""
    targets = build_target_labels(mock_transactions, CUTOFF)

    assert len(targets) == 2

    u001_target = targets.loc[targets["msno"] == "U001", "is_churn"].values[0]
    u002_target = targets.loc[targets["msno"] == "U002", "is_churn"].values[0]

    # U001's last transaction BEFORE cutoff expired Nov 1. Thus it's past cutoff
    assert u001_target == 1

    # U002 explicitly cancelled
    assert u002_target == 1


def test_build_rfm_features_prevents_leakage(mock_transactions):
    """Ensure future transactions are not aggregated into monetary values."""
    rfm = build_rfm_features(mock_transactions, CUTOFF)

    # U001 should only have 1 transaction aggregated (amount: 100)
    u001_rfm = rfm.loc[rfm["msno"] == "U001"]

    assert u001_rfm["frequency"].values[0] == 1
    assert u001_rfm["monetary_total"].values[0] == 100.0


def test_build_engagement_features_prevents_leakage(mock_logs):
    """Ensure future logs do not artificially inflate engagement."""
    eng = build_engagement_features(mock_logs, CUTOFF)

    # U002 log is post-cutoff, so it should be excluded or return 0
    u002_eng = eng.loc[eng["msno"] == "U002"]
    if len(u002_eng) > 0:
        assert u002_eng["total_secs_30d"].values[0] == 0.0

    u001_eng = eng.loc[eng["msno"] == "U001"]
    assert u001_eng["total_secs_60d"].values[0] == 5400.0 # 3600 + 1800
