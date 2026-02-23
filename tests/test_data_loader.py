"""
Data loader tests - schema and handling of edge cases.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_members, load_transactions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


@pytest.fixture
def mock_members_path(tmp_path):
    df = pd.DataFrame(
        {
            "msno": ["U001", "U002"],
            "city": [1, 2],
            "bd": [28, None],
            "gender": ["male", None],
            "registered_via": [3, 4],
            "registration_init_time": ["2023-01-01", "2023-02-01"],
        }
    )
    filepath = tmp_path / "members.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def test_members_imputation(mock_members_path):
    df = load_members(mock_members_path)

    assert len(df) == 2
    assert df["gender"].dtype.name == "category" or pd.api.types.is_object_dtype(
        df["gender"]
    )

    assert not df["bd"].isnull().any()
    assert "Missing" in df["gender"].cat.categories
    assert df.loc[1, "gender"] == "Missing"


@pytest.fixture
def mock_transactions_path(tmp_path):
    df = pd.DataFrame(
        {
            "msno": ["U001", "U002"],
            "payment_method_id": [38, 39],
            "payment_plan_days": [30, 30],
            "plan_list_price": [149.0, 149.0],
            "actual_amount_paid": [149.0, 149.0],
            "is_auto_renew": [1, 0],
            "is_cancel": [0, 1],
            "transaction_date": ["2023-01-01", "2023-02-15"],
            "membership_expire_date": [
                "2023-01-31",
                "2023-02-01",
            ],  # U002: tx after expire
        }
    )
    filepath = tmp_path / "transactions.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def test_transactions_invalid_dates_dropped(mock_transactions_path):
    """U002 has tx_date > expire_date, should get dropped."""
    df = load_transactions(mock_transactions_path)

    assert len(df) == 1
    assert df.iloc[0]["msno"] == "U001"
