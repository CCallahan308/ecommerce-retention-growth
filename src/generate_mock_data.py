"""
Synthetic data generator - members, transactions, user_logs.
Use this instead of downloading the huge Kaggle files.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Constants
NUM_USERS = 5000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 1, 1)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def generate_members(num_users: int) -> pd.DataFrame:
    """Generate mock members."""
    np.random.seed(42)
    user_ids = [f"U{str(i).zfill(5)}" for i in range(num_users)]
    cities = np.random.choice(range(1, 23), size=num_users)
    bd = np.random.normal(28, 10, size=num_users).astype(int)
    # Filter absurd ages
    bd = np.where((bd < 10) | (bd > 90), 0, bd)
    genders = np.random.choice(
        ["male", "female", "unknown"], size=num_users, p=[0.4, 0.4, 0.2]
    )
    registered_via = np.random.choice([3, 4, 7, 9, 13], size=num_users)

    # Registration dates spanning the past 2 years
    reg_dates = [
        START_DATE - timedelta(days=np.random.randint(0, 700)) for _ in range(num_users)
    ]

    return pd.DataFrame(
        {
            "msno": user_ids,
            "city": cities,
            "bd": bd,
            "gender": genders,
            "registered_via": registered_via,
            "registration_init_time": reg_dates,
        }
    )


def generate_transactions(members_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock transactions."""
    np.random.seed(42)
    transactions = []

    for _, row in members_df.iterrows():
        msno = row["msno"]
        reg_date = row["registration_init_time"]

        # 1 to 12 transactions per user
        num_trans = np.random.randint(1, 13)

        current_date = reg_date
        for _ in range(num_trans):
            payment_method_id = np.random.choice([38, 39, 40, 41])
            payment_plan_days = 30
            plan_list_price = 149
            actual_amount_paid = plan_list_price
            is_auto_renew = np.random.choice([0, 1], p=[0.3, 0.7])

            # Transaction happens a few days before or on the expiration
            transaction_date = current_date + timedelta(days=np.random.randint(-2, 2))
            membership_expire_date = transaction_date + timedelta(
                days=payment_plan_days
            )
            is_cancel = np.random.choice([0, 1], p=[0.95, 0.05])

            transactions.append(
                {
                    "msno": msno,
                    "payment_method_id": payment_method_id,
                    "payment_plan_days": payment_plan_days,
                    "plan_list_price": plan_list_price,
                    "actual_amount_paid": actual_amount_paid,
                    "is_auto_renew": is_auto_renew,
                    "transaction_date": transaction_date,
                    "membership_expire_date": membership_expire_date,
                    "is_cancel": is_cancel,
                }
            )

            if is_cancel:
                break

            current_date = membership_expire_date

    return pd.DataFrame(transactions)


def generate_user_logs(members_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock listening logs."""
    np.random.seed(42)
    logs = []

    # Take a sample of users to generate logs for, simulating inactive users
    active_users = members_df.sample(frac=0.8, random_state=42)["msno"]

    for msno in active_users:
        # Generate 10 to 50 random log days in the final 60 days of the year
        num_log_days = np.random.randint(10, 50)
        base_log_date = END_DATE - timedelta(days=60)

        for _ in range(num_log_days):
            date = base_log_date + timedelta(days=np.random.randint(0, 60))
            num_25 = np.random.randint(0, 10)
            num_50 = np.random.randint(0, 5)
            num_75 = np.random.randint(0, 3)
            num_985 = np.random.randint(0, 2)
            num_100 = np.random.randint(5, 100)
            num_unq = int((num_25 + num_50 + num_75 + num_985 + num_100) * 0.8)
            total_secs = num_100 * 240 + np.random.randint(0, 1000)

            logs.append(
                {
                    "msno": msno,
                    "date": date,
                    "num_25": num_25,
                    "num_50": num_50,
                    "num_75": num_75,
                    "num_985": num_985,
                    "num_100": num_100,
                    "num_unq": num_unq,
                    "total_secs": total_secs,
                }
            )

    return pd.DataFrame(logs)


def main() -> None:
    """Generate and save all mock datasets."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Generating data for {NUM_USERS} users...")
    members = generate_members(NUM_USERS)
    print("Generating transactions...")
    transactions = generate_transactions(members)
    print("Generating user logs...")
    user_logs = generate_user_logs(members)

    members.to_csv(os.path.join(DATA_DIR, "members.csv"), index=False)
    transactions.to_csv(os.path.join(DATA_DIR, "transactions.csv"), index=False)
    user_logs.to_csv(os.path.join(DATA_DIR, "user_logs.csv"), index=False)

    print(f"Data successfully generated in '{DATA_DIR}'.")


if __name__ == "__main__":
    main()
