import os

import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
SAMPLE_SIZE = 50000

print("sampling data...")

members_path = os.path.join(DATA_DIR, "members.csv")
transactions_path = os.path.join(DATA_DIR, "transactions.csv")
user_logs_path = os.path.join(DATA_DIR, "user_logs.csv")

try:
    members_df = pd.read_csv(members_path)
except FileNotFoundError:
    print("Couldn't find members.csv. Did you download it?")
    exit(1)

sampled_members = members_df.sample(n=min(SAMPLE_SIZE, len(members_df)), random_state=42)
valid_users = set(sampled_members["msno"])

sampled_members.to_csv(members_path, index=False)
print(f"Kept {len(valid_users)} users")

# chunking through transactions
print("Chunking transactions...")
chunk_size = 1000000
temp_trans = os.path.join(DATA_DIR, "transactions_sampled.csv")
first_chunk = True

for chunk in tqdm(pd.read_csv(transactions_path, chunksize=chunk_size)):
    filtered_chunk = chunk[chunk["msno"].isin(valid_users)]
    filtered_chunk.to_csv(temp_trans, mode='a', header=first_chunk, index=False)
    first_chunk = False

os.replace(temp_trans, transactions_path)

# chunking logs (the 30GB file)
print("Chunking logs... this takes a bit")
temp_logs = os.path.join(DATA_DIR, "user_logs_sampled.csv")
first_chunk = True

for chunk in tqdm(pd.read_csv(user_logs_path, chunksize=chunk_size)):
    filtered_chunk = chunk[chunk["msno"].isin(valid_users)]
    filtered_chunk.to_csv(temp_logs, mode='a', header=first_chunk, index=False)
    first_chunk = False

os.replace(temp_logs, user_logs_path)
print("done!")
