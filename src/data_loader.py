"""
Data Loading and Validation Module.

This module provides reproducible and typed functions to load raw subscription data
into memory, enforcing correct data types and handling missing values natively.
"""

import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Default paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

# Schema definitions mapping column names to Pandas datatypes
MEMBERS_SCHEMA: Any = {
    "msno": "string",
    "city": "category",
    "bd": "Int16",  # Integer with NA support
    "gender": "category",
    "registered_via": "category",
}

TRANSACTIONS_SCHEMA: Any = {
    "msno": "string",
    "payment_method_id": "category",
    "payment_plan_days": "Int16",
    "plan_list_price": "float32",
    "actual_amount_paid": "float32",
    "is_auto_renew": "Int8",
    "is_cancel": "Int8",
}

USER_LOGS_SCHEMA: Any = {
    "msno": "string",
    "num_25": "Int32",
    "num_50": "Int32",
    "num_75": "Int32",
    "num_985": "Int32",
    "num_100": "Int32",
    "num_unq": "Int32",
    "total_secs": "float64",
}


def load_members(filepath: str = os.path.join(DATA_DIR, "members.csv")) -> pd.DataFrame:
    """
    Load user demographics and registration data.

    Parameters
    ----------
    filepath : str
        Path to the members.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned members DataFrame.

    """
    logger.info(f"Loading members data from {filepath}")
    df = pd.read_csv(
        filepath,
        dtype=MEMBERS_SCHEMA,  # type: ignore
        parse_dates=["registration_init_time"],
    )

    # Impute missing demographics to prevent data loss
    df["gender"] = df["gender"].cat.add_categories(["Missing"]).fillna("Missing")
    df["bd"] = df["bd"].fillna(df["bd"].median())

    return df


def load_transactions(filepath: str = os.path.join(DATA_DIR, "transactions.csv")) -> pd.DataFrame:
    """
    Load subscription transaction logs.

    Parameters
    ----------
    filepath : str
        Path to the transactions.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned transactions DataFrame.

    """
    logger.info(f"Loading transactions data from {filepath}")
    df = pd.read_csv(
        filepath,
        dtype=TRANSACTIONS_SCHEMA,  # type: ignore
        parse_dates=["transaction_date", "membership_expire_date"],
    )

    # Ensure temporal consistency
    invalid_dates = df["transaction_date"] > df["membership_expire_date"]
    if invalid_dates.any():
        logger.warning(f"Found {invalid_dates.sum()} rows where transaction > expiration. Fixing.")
        # Optional: Clip or drop these depending on business logic
        df = df[~invalid_dates].copy()  # type: ignore

    return df  # type: ignore


def load_user_logs(filepath: str = os.path.join(DATA_DIR, "user_logs.csv")) -> pd.DataFrame:
    """
    Load daily usage telemetry logs.

    Parameters
    ----------
    filepath : str
        Path to the user_logs.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned usage telemetry DataFrame.

    """
    logger.info(f"Loading user logs data from {filepath}")
    df = pd.read_csv(
        filepath,
        dtype=USER_LOGS_SCHEMA,  # type: ignore
        parse_dates=["date"],
    )

    # Basic data quality checks
    df["total_secs"] = df["total_secs"].clip(lower=0)

    return df


def load_all_data(data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all three core datasets.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw CSVs.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple containing (members, transactions, user_logs) DataFrames.

    """
    members = load_members(os.path.join(data_dir, "members.csv"))
    transactions = load_transactions(os.path.join(data_dir, "transactions.csv"))
    user_logs = load_user_logs(os.path.join(data_dir, "user_logs.csv"))

    return members, transactions, user_logs


if __name__ == "__main__":
    # Smoke test data loading logic
    try:
        m, t, u = load_all_data()
        logger.info(f"Successfully loaded. Shapes: Members {m.shape}, Trans {t.shape}, Logs {u.shape}")
    except FileNotFoundError as e:
        logger.error(f"Missing data file: {e}. Please run generate_mock_data.py first.")
