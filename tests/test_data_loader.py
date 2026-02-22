"""Tests for the data_loader module to ensure schema adherence and robust handling."""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow running this file directly: `python tests/test_data_loader.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_members,
    load_transactions,
)

# Use dummy data or the generated raw data for tests
# In a real environment, we'd use small fixtures
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


@pytest.fixture
def mock_members_path(tmp_path):
    df = pd.DataFrame(
        {
            "msno": ["U001", "U002"],
            "city": [1, 2],
            "bd": [28, None],  # One missing to test imputation
            "gender": ["male", None],  # One missing to test category imputation
            "registered_via": [3, 4],
            "registration_init_time": ["2023-01-01", "2023-02-01"],
        }
    )
    filepath = tmp_path / "members.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def test_load_members_schema_and_imputation(mock_members_path):
    """Test that missing demographic values are imputed properly."""
    df = load_members(mock_members_path)

    # Check shapes and types
    assert len(df) == 2
    assert (
        isinstance(df["gender"].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(df["gender"])
        or df["gender"].dtype.name == "category"
    )

    # Check imputation
    assert not pd.Series(df["bd"].isnull()).any(), (
        "bd shouldn't have nulls after imputation"
    )
    assert "Missing" in df["gender"].cat.categories, "Missing category should be added"
    assert df.loc[1, "gender"] == "Missing", (
        "Null gender should be replaced with 'Missing'"
    )


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
            # U002 has an invalid date scenario (transaction > expire)
            "transaction_date": ["2023-01-01", "2023-02-15"],
            "membership_expire_date": ["2023-01-31", "2023-02-01"],
        }
    )
    filepath = tmp_path / "transactions.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def test_load_transactions_drops_invalid_dates(mock_transactions_path):
    """Test that transactions with expiration before transaction date are dropped."""
    df = load_transactions(mock_transactions_path)

    # U002 should be dropped due to invalid temporal relationship
    assert len(df) == 1
    assert df.iloc[0]["msno"] == "U001"
