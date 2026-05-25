"""
Sample the full KKBox dataset down to a fixed cohort of labeled users.

This is the reproducible bridge between ``download_real_data.py`` and ``make
train`` when you cannot fit the full ~30GB dataset in memory. It samples ``msno``
values from the (small) ``train.csv`` labels with a fixed seed, then filters
members / transactions / user_logs / train to that cohort **in chunks**, so the
multi-GB files are never loaded whole.

Run:
    python src/sample_kaggle_data.py            # default KAGGLE_SAMPLE_SIZE users
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import KAGGLE_SAMPLE_SIZE, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
LARGE_FILE_CHUNK = 1_000_000


def _filter_in_chunks(path: str, keep: set, chunk_size: int) -> None:
    """Rewrite ``path`` in place keeping only rows whose msno is in ``keep``."""
    if not os.path.exists(path):
        logger.warning("%s not found; skipping", os.path.basename(path))
        return
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    kept, first = 0, True
    for chunk in pd.read_csv(path, chunksize=chunk_size, dtype={"msno": "string"}):
        filtered = chunk[chunk["msno"].isin(keep)]
        filtered.to_csv(tmp, mode="a", header=first, index=False)
        kept += len(filtered)
        first = False
    os.replace(tmp, path)
    logger.info("%s: kept %s rows", os.path.basename(path), f"{kept:,}")


def sample_data(
    data_dir: str = DATA_DIR,
    sample_size: int = KAGGLE_SAMPLE_SIZE,
    seed: int = RANDOM_STATE,
) -> None:
    """Sample a fixed cohort of labeled users and filter every raw file to it."""
    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        logger.error(
            "train.csv not found in %s. Run download_real_data.py (or "
            "extract_kaggle_data.py) first.",
            data_dir,
        )
        sys.exit(1)

    labels = pd.read_csv(train_path, dtype={"msno": "string", "is_churn": "Int8"})
    sample = labels.sample(n=min(sample_size, len(labels)), random_state=seed)
    keep = set(sample["msno"])
    sample.to_csv(train_path, index=False)
    logger.info(
        "Sampled %s labeled users (seed=%d); churn rate %.3f",
        f"{len(sample):,}",
        seed,
        sample["is_churn"].mean(),
    )

    _filter_in_chunks(os.path.join(data_dir, "members.csv"), keep, 500_000)
    _filter_in_chunks(os.path.join(data_dir, "transactions.csv"), keep, 500_000)
    _filter_in_chunks(os.path.join(data_dir, "user_logs.csv"), keep, LARGE_FILE_CHUNK)
    logger.info("Sampling complete.")


if __name__ == "__main__":
    sample_data()
