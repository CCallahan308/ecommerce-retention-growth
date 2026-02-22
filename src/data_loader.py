import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

MEMBERS_DTYPE = {
    "msno": "string",
    "city": "category",
    "bd": "Int16",
    "gender": "category",
    "registered_via": "category",
}

TX_DTYPE = {
    "msno": "string",
    "payment_method_id": "category",
    "payment_plan_days": "Int16",
    "plan_list_price": "float32",
    "actual_amount_paid": "float32",
    "is_auto_renew": "Int8",
    "is_cancel": "Int8",
}

LOGS_DTYPE = {
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
    df = pd.read_csv(
        filepath,
        dtype=MEMBERS_DTYPE,
        parse_dates=["registration_init_time"],
    )

    df["gender"] = df["gender"].cat.add_categories(["Missing"]).fillna("Missing")
    df["bd"] = df["bd"].fillna(df["bd"].median())

    return df


def load_transactions(
    filepath: str = os.path.join(DATA_DIR, "transactions.csv"),
) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        dtype=TX_DTYPE,
        parse_dates=["transaction_date", "membership_expire_date"],
    )

    bad = df["transaction_date"] > df["membership_expire_date"]
    if bad.any():
        df = df[~bad].copy()

    return df


def load_user_logs(
    filepath: str = os.path.join(DATA_DIR, "user_logs.csv"),
) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        dtype=LOGS_DTYPE,
        parse_dates=["date"],
    )
    df["total_secs"] = df["total_secs"].clip(lower=0)
    return df


def load_all_data(data_dir: str = DATA_DIR):
    # load all 3 files
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
