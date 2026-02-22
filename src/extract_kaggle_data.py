"""
Extract KKBox Kaggle competition .7z archives into data/raw/.

Expects the archives to be in a `kkbox-churn-prediction-challenge/` folder
at the project root. Handles nested directory structures inside the archives.
"""

import logging
import os
import shutil
from pathlib import Path

import py7zr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Resolve paths relative to the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = _PROJECT_ROOT / "kkbox-churn-prediction-challenge"
RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"


def extract_and_rename_recursive(archive_name, final_csv_name):
    archive_path = KAGGLE_DIR / archive_name
    if not archive_path.exists():
        logger.warning(f"Archive {archive_name} not found in {KAGGLE_DIR}")
        return

    logger.info(f"Extracting {archive_name} to {RAW_DATA_DIR}...")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=RAW_DATA_DIR)

    # Search for any extracted CSV file (since they might be nested)
    csv_files = list(RAW_DATA_DIR.glob("**/*.csv"))
    # Filter for the one that likely corresponds to our archive
    # (avoiding the final_csv_name if it already exists from a previous step)
    target_csv = None
    for f in csv_files:
        # Ignore our target names from other extractions
        if f.name in ["members.csv", "transactions.csv", "user_logs.csv"]:
            continue
        # For members_v3.csv.7z -> matches "members"
        match_keyword = archive_name.split("_")[0]
        if match_keyword in f.name:
            target_csv = f
            break

    if target_csv:
        final_path = RAW_DATA_DIR / final_csv_name
        if final_path.exists():
            os.remove(final_path)

        logger.info(f"Moving {target_csv} to {final_path}")
        # Use shutil.move for potentially inter-drive moves and cleaner directory cleanup
        shutil.move(str(target_csv), str(final_path))
    else:
        logger.error(
            f"Could not find a matching CSV for {archive_name} in {RAW_DATA_DIR}"
        )


def cleanup_nested_dirs(root_dir):
    # clean up nested "data/churn_comp_refresh/" folders if they exist
    test_nested = root_dir / "data"
    if test_nested.exists():
        logger.info("Cleaning up nested extraction folders...")
        shutil.rmtree(str(test_nested))


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Members
    extract_and_rename_recursive("members_v3.csv.7z", "members.csv")

    # 2. Transactions
    extract_and_rename_recursive("transactions_v2.csv.7z", "transactions.csv")

    # 3. User Logs (THE BIG ONE)
    logger.info("Extracting user_logs_v2.csv.7z. This is the 30GB file. Please wait...")
    extract_and_rename_recursive("user_logs_v2.csv.7z", "user_logs.csv")

    # 4. Official train labels
    extract_and_rename_recursive("train_v2.csv.7z", "train.csv")

    # Cleanup nested artifacts
    cleanup_nested_dirs(RAW_DATA_DIR)

    logger.info(
        "Finished. Run 'python src/train_predict.py' to train and predict."
    )


if __name__ == "__main__":
    main()
