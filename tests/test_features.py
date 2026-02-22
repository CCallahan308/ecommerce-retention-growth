"""
Tests for the feature engineering pipeline to ensure logic correctness and
guarantee no target leakage from future events.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Allow running this file directly: `python tests/test_features.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    build_engagement_features,
    build_rfm_features,
    prep_targets,
)

CUTOFF = datetime(2023, 12, 1)


@pytest.fixture
def mock_transactions():
    return pd.DataFrame(
        {
            "msno": ["U001", "U001", "U002", "U003", "U003"],
            "transaction_date": [
                pd.to_datetime("2023-10-01"),  # U001 before cutoff
                pd.to_datetime(
                    "2023-12-15"
                ),  # U001 AFTER cutoff (renews within 30 days of expire)
                pd.to_datetime(
                    "2023-11-01"
                ),  # U002 before cutoff (no renewal -> churn)
                pd.to_datetime("2023-10-15"),  # U003 before cutoff
                pd.to_datetime(
                    "2024-01-20"
                ),  # U003 AFTER cutoff (renews > 30 days after expire -> churn)
            ],
            "membership_expire_date": [
                pd.to_datetime("2023-11-20"),  # U001 expire
                pd.to_datetime("2024-01-15"),  # U001 new expire
                pd.to_datetime("2023-11-30"),  # U002 expire
                pd.to_datetime("2023-11-15"),  # U003 expire
                pd.to_datetime("2024-02-20"),  # U003 new expire
            ],
            "actual_amount_paid": [100.0, 150.0, 100.0, 100.0, 100.0],
            "is_auto_renew": [1, 1, 0, 0, 0],
            "is_cancel": [0, 0, 1, 1, 0],
        }
    )


@pytest.fixture
def mock_logs():
    return pd.DataFrame(
        {
            "msno": ["U001", "U001", "U002"],
            "date": [
                pd.to_datetime("2023-11-15"),  # Within 30d of cutoff
                pd.to_datetime("2023-10-15"),  # Within 60d of cutoff
                pd.to_datetime("2023-12-10"),  # AFTER cutoff (leakage risk)
            ],
            "total_secs": [3600.0, 1800.0, 7200.0],
            "num_unq": [10, 5, 20],
        }
    )


def test_prep_targets(mock_transactions):
    """Ensure targets only consider historical facts up to cutoff."""
    targets = prep_targets(mock_transactions, CUTOFF)

    assert len(targets) == 3

    u001_target = targets.loc[targets["msno"] == "U001", "is_churn"].values[0]
    u002_target = targets.loc[targets["msno"] == "U002", "is_churn"].values[0]
    u003_target = targets.loc[targets["msno"] == "U003", "is_churn"].values[0]

    # U001 renewed on 12-15, which is within 30 days of 11-20 (25 days) -> NO CHURN
    assert u001_target == 0

    # U001 confirmed correctly handles its LAST transaction before cutoff:
    # U001 last tx before cutoff was 10-01, expire: 11-20
    # Future renew was 12-15. Gap: 2023-12-15 - 2023-11-20 = 25 days.

    # U002 has no transactions after cutoff -> CHURN
    assert u002_target == 1

    # U003 renewed on 01-20, which is > 30 days after 11-15 (66 days) -> CHURN
    assert u003_target == 1


def test_prep_targets_handles_mid_window_cancellation():
    """Verify target calculation follows the official Scala logic for mid-window cancels."""
    # Scenario: User cancels after cutoff but before renewing.
    # Expiration is moved forward. Gap is from NEW expiration.
    cutoff = pd.to_datetime("2023-12-01")
    data = pd.DataFrame(
        {
            "msno": ["U004", "U004", "U004"],
            "transaction_date": [
                pd.to_datetime("2023-11-01"),  # before cutoff
                pd.to_datetime("2023-12-05"),  # AFTER cutoff: ACTIVE CANCEL
                pd.to_datetime("2024-01-08"),  # AFTER cutoff: RENEWAL
            ],
            "membership_expire_date": [
                pd.to_datetime("2023-12-15"),  # initial expire
                pd.to_datetime("2023-12-10"),  # moved UP due to cancel
                pd.to_datetime("2024-02-10"),  # final renew expire
            ],
            "actual_amount_paid": [100.0, 0.0, 100.0],
            "is_auto_renew": [1, 1, 1],
            "is_cancel": [0, 1, 0],
        }
    )

    targets = prep_targets(data, cutoff)
    assert targets.loc[targets["msno"] == "U004", "is_churn"].values[0] == 0

    # Gap = 2024-01-08 - 2023-12-10 (cancel-adjusted expire) = 29 days.
    # 29 days is < 30 -> NO CHURN
    # If it didn't adjust, gap would be 2024-01-08 - 2023-12-15 = 24 days.
    # Still no churn, but the behavior is verified.
    # Actually let's make it 31 days to prove churn.
    data_churned = data.copy()
    data_churned.loc[2, "transaction_date"] = pd.to_datetime("2024-01-11")

    # Gap = 2024-01-11 - 2023-12-10 = 32 days -> CHURN!
    # If didn't adjust expire: gap would be 2024-01-11 - 2023-12-15 = 27 days -> NO CHURN.
    targets_churned = prep_targets(data_churned, cutoff)
    u004_churned = targets_churned.loc[
        targets_churned["msno"] == "U004", "is_churn"
    ].values[0]

    assert u004_churned == 1


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
    assert u001_eng["total_secs_60d"].values[0] == 5400.0  # 3600 + 1800
