"""
Feature engineering tests - checks for leakage and logic correctness.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

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
                pd.to_datetime("2023-10-01"),
                pd.to_datetime("2023-12-15"),  # after cutoff, renews within 30d
                pd.to_datetime("2023-11-01"),  # before cutoff, no renewal -> churn
                pd.to_datetime("2023-10-15"),
                pd.to_datetime("2024-01-20"),  # after cutoff, gap > 30d -> churn
            ],
            "membership_expire_date": [
                pd.to_datetime("2023-11-20"),
                pd.to_datetime("2024-01-15"),
                pd.to_datetime("2023-11-30"),
                pd.to_datetime("2023-11-15"),
                pd.to_datetime("2024-02-20"),
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
                pd.to_datetime("2023-11-15"),
                pd.to_datetime("2023-10-15"),
                pd.to_datetime("2023-12-10"),  # after cutoff - shouldn't count
            ],
            "total_secs": [3600.0, 1800.0, 7200.0],
            "num_unq": [10, 5, 20],
        }
    )


def test_prep_targets(mock_transactions):
    """Make sure we only look at historical data up to cutoff."""
    targets = prep_targets(mock_transactions, CUTOFF)

    assert len(targets) == 3

    u001 = targets.loc[targets["msno"] == "U001", "is_churn"].values[0]
    u002 = targets.loc[targets["msno"] == "U002", "is_churn"].values[0]
    u003 = targets.loc[targets["msno"] == "U003", "is_churn"].values[0]

    # U001: renewed 12-15, gap from 11-20 is 25 days -> not churn
    assert u001 == 0

    # U002: no future tx -> churn
    assert u002 == 1

    # U003: gap from 11-15 to 01-20 is 66 days -> churn
    assert u003 == 1


def test_prep_targets_mid_window_cancel():
    """Scala labeler adjusts expiry on mid-window cancellation."""
    cutoff = pd.to_datetime("2023-12-01")
    data = pd.DataFrame(
        {
            "msno": ["U004", "U004", "U004"],
            "transaction_date": [
                pd.to_datetime("2023-11-01"),
                pd.to_datetime("2023-12-05"),  # cancel after cutoff
                pd.to_datetime("2024-01-08"),  # renewal
            ],
            "membership_expire_date": [
                pd.to_datetime("2023-12-15"),
                pd.to_datetime("2023-12-10"),  # moved up by cancel
                pd.to_datetime("2024-02-10"),
            ],
            "actual_amount_paid": [100.0, 0.0, 100.0],
            "is_auto_renew": [1, 1, 1],
            "is_cancel": [0, 1, 0],
        }
    )

    targets = prep_targets(data, cutoff)
    # gap = 2024-01-08 - 2023-12-10 = 29 days -> not churn
    assert targets.loc[targets["msno"] == "U004", "is_churn"].values[0] == 0

    # now make it 31 days to prove churn
    data2 = data.copy()
    data2.loc[2, "transaction_date"] = pd.to_datetime("2024-01-11")
    # gap = 2024-01-11 - 2023-12-10 = 32 days -> churn
    targets2 = prep_targets(data2, cutoff)
    assert targets2.loc[targets2["msno"] == "U004", "is_churn"].values[0] == 1


def test_rfm_no_leakage(mock_transactions):
    """Future transactions shouldn't affect RFM."""
    rfm = build_rfm_features(mock_transactions, CUTOFF)

    u001 = rfm.loc[rfm["msno"] == "U001"]
    assert u001["frequency"].values[0] == 1
    assert u001["monetary_total"].values[0] == 100.0


def test_engagement_no_leakage(mock_logs):
    """Post-cutoff logs shouldn't inflate engagement."""
    eng = build_engagement_features(mock_logs, CUTOFF)

    u002 = eng.loc[eng["msno"] == "U002"]
    if len(u002) > 0:
        assert u002["total_secs_30d"].values[0] == 0.0

    u001 = eng.loc[eng["msno"] == "U001"]
    assert u001["total_secs_60d"].values[0] == 5400.0
