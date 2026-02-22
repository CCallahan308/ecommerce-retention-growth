import os
import logging
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
SAMPLE_SIZE = 50000  # 50k users is plenty to train a robust model while fitting easily in 16GB RAM

def sample_large_dataset():
    """
    Reads larger-than-RAM Kaggle datasets in chunks, filtering them down 
    to a representative sample of users to prevent memory crashes.
    """
    logger.info("Starting memory-efficient data sampling...")
    
    members_path = os.path.join(DATA_DIR, "members.csv")
    transactions_path = os.path.join(DATA_DIR, "transactions.csv")
    user_logs_path = os.path.join(DATA_DIR, "user_logs.csv")
    
    # 1. Sample Members
    logger.info(f"Loading and sampling {SAMPLE_SIZE} users from members.csv...")
    try:
        members_df = pd.read_csv(members_path)
    except FileNotFoundError:
        logger.error("Could not find members.csv. Did you run the Kaggle download script?")
        return
        
    # Take a random sample of users
    sampled_members = members_df.sample(n=min(SAMPLE_SIZE, len(members_df)), random_state=42)
    valid_users = set(sampled_members["msno"])
    
    # Save the sampled members back to disk, overwriting the massive file
    sampled_members.to_csv(members_path, index=False)
    logger.info(f"Saved sampled members. (Retained {len(valid_users)} users)")
    
    # 2. Filter Transactions (Chunking)
    logger.info("Filtering transactions.csv in chunks to prevent RAM crashes...")
    chunk_size = 1000000  # Process 1 million rows at a time
    
    # We will write the filtered chunks to a temporary file, then replace the original
    temp_trans = os.path.join(DATA_DIR, "transactions_sampled.csv")
    first_chunk = True
    
    for chunk in tqdm(pd.read_csv(transactions_path, chunksize=chunk_size)):
        # Keep only transactions belonging to our sampled users
        filtered_chunk = chunk[chunk["msno"].isin(valid_users)]
        
        # Append to new file
        filtered_chunk.to_csv(temp_trans, mode='a', header=first_chunk, index=False)
        first_chunk = False
        
    os.replace(temp_trans, transactions_path)
    logger.info("Successfully downsampled transactions.csv")
    
    # 3. Filter User Logs (Chunking - This is the 30GB file!)
    logger.info("Filtering the massive user_logs.csv in chunks... This may take a few minutes.")
    temp_logs = os.path.join(DATA_DIR, "user_logs_sampled.csv")
    first_chunk = True
    
    for chunk in tqdm(pd.read_csv(user_logs_path, chunksize=chunk_size)):
        filtered_chunk = chunk[chunk["msno"].isin(valid_users)]
        filtered_chunk.to_csv(temp_logs, mode='a', header=first_chunk, index=False)
        first_chunk = False
        
    os.replace(temp_logs, user_logs_path)
    logger.info("Successfully downsampled user_logs.csv")
    
    logger.info("Data sampling complete! Your dataset is now safe to use with 16GB of RAM.")

if __name__ == "__main__":
    sample_large_dataset()
