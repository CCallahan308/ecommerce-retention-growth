"""
Sample full Kaggle data down to N users.
Filters transactions and logs to only include sampled users.
"""

import os
import sys

import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
SAMPLE_SIZE = 50_000


def sample_data(data_dir: str = DATA_DIR, sample_size: int = SAMPLE_SIZE):
    """Sample all raw CSVs to a fixed user cohort."""
    members_path = os.path.join(data_dir, "members.csv")
    transactions_path = os.path.join(data_dir, "transactions.csv")
    user_logs_path = os.path.join(data_dir, "user_logs.csv")

    try:
        members_df = pd.read_csv(members_path)
    except FileNotFoundError:
        print(
            "members.csv not found. Run extract_kaggle_data.py or generate_mock_data.py first."
        )
        sys.exit(1)

    sampled_members = members_df.sample(
        n=min(sample_size, len(members_df)), random_state=42
    )
    valid_users = set(sampled_members["msno"])
    sampled_members.to_csv(members_path, index=False)
    print(f"Kept {len(valid_users)} users")

    # Filter transactions in chunks
    print("Filtering transactions...")
    chunk_size = 1_000_000
    temp_trans = os.path.join(data_dir, "transactions_sampled.csv")
    first_chunk = True

    for chunk in tqdm(pd.read_csv(transactions_path, chunksize=chunk_size)):
        filtered_chunk = chunk[chunk["msno"].isin(valid_users)]
        filtered_chunk.to_csv(temp_trans, mode="a", header=first_chunk, index=False)
        first_chunk = False

    os.replace(temp_trans, transactions_path)

    # Filter user logs in chunks (the 30GB file)
    print("Filtering user logs (this may take a few minutes)...")
    temp_logs = os.path.join(data_dir, "user_logs_sampled.csv")
    first_chunk = True

    for chunk in tqdm(pd.read_csv(user_logs_path, chunksize=chunk_size)):
        filtered_chunk = chunk[chunk["msno"].isin(valid_users)]
        filtered_chunk.to_csv(temp_logs, mode="a", header=first_chunk, index=False)
        first_chunk = False

    os.replace(temp_logs, user_logs_path)
    print("Sampling complete.")


if __name__ == "__main__":
    sample_data()
