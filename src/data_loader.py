import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Default paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

MEMBERS_SCHEMA: Any = {
    "msno": "string",
    "city": "category",
    "bd": "Int16",
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
    """Load demographics and handle missing ages/genders."""
    df = pd.read_csv(
        filepath,
        dtype=MEMBERS_SCHEMA,  # type: ignore
        parse_dates=["registration_init_time"],
    )

    df["gender"] = df["gender"].cat.add_categories(["Missing"]).fillna("Missing")
    df["bd"] = df["bd"].fillna(df["bd"].median())

    return df


def load_transactions(
    filepath: str = os.path.join(DATA_DIR, "transactions.csv"),
) -> pd.DataFrame:
    """Load billing history."""
    df = pd.read_csv(
        filepath,
        dtype=TRANSACTIONS_SCHEMA,  # type: ignore
        parse_dates=["transaction_date", "membership_expire_date"],
    )

    # Drop rows where transaction_date exceeds membership_expire_date (data quality)
    invalid_dates = df["transaction_date"] > df["membership_expire_date"]
    if invalid_dates.any():
        df = df[~invalid_dates].copy()  # type: ignore

    return df  # type: ignore


def load_user_logs(
    filepath: str = os.path.join(DATA_DIR, "user_logs.csv"),
) -> pd.DataFrame:
    """
    Load user listening logs.

    Parses dates and clips negative listening times to zero.
    """
    df = pd.read_csv(
        filepath,
        dtype=USER_LOGS_SCHEMA,  # type: ignore
        parse_dates=["date"],
    )
    df["total_secs"] = df["total_secs"].clip(lower=0)
    return df


def load_all_data(data_dir: str = DATA_DIR):
    """
    Load all three datasets (members, transactions, user logs).

    Returns a tuple of (members, transactions, user_logs) DataFrames.
    """
    members = load_members(os.path.join(data_dir, "members.csv"))
    transactions = load_transactions(os.path.join(data_dir, "transactions.csv"))
    user_logs = load_user_logs(os.path.join(data_dir, "user_logs.csv"))

    return members, transactions, user_logs


if __name__ == "__main__":
    try:
        m, t, u = load_all_data()
        print(f"Loaded ok. Members: {len(m)}, Trans: {len(t)}, Logs: {len(u)}")
    except FileNotFoundError as e:
        print(f"Missing file: {e}")
