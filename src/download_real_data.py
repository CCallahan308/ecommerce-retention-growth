import logging
import os
import subprocess

import py7zr

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

COMPETITION = "kkbox-churn-prediction-challenge"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def run_kaggle_download(filename: str):
    logger.info(f"Downloading {filename} from Kaggle API...")
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-f",
            filename,
            "-p",
            DATA_DIR,
        ],
        check=True,
    )


def extract_7z(archive_path: str, extract_path: str):
    logger.info(f"Extracting {archive_path}...")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=extract_path)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    files_to_download = [
        "members_v3.csv.7z",
        "transactions_v2.csv.7z",
        "user_logs_v2.csv.7z",
    ]

    for f in files_to_download:
        archive_path = os.path.join(DATA_DIR, f)
        csv_name = f.replace(".7z", "")
        final_path = os.path.join(
            DATA_DIR, csv_name.replace("_v3", "").replace("_v2", "")
        )

        # 1. Download
        if not os.path.exists(archive_path) and not os.path.exists(final_path):
            try:
                run_kaggle_download(f)
            except subprocess.CalledProcessError:
                logger.error(
                    f"Failed to download {f}. Have you accepted the competition rules on Kaggle.com and set up your kaggle.json?"
                )
                return

        # 2. Extract
        if os.path.exists(archive_path) and not os.path.exists(final_path):
            extract_7z(archive_path, DATA_DIR)

            # The extracted file will have the version suffix, let's rename it to match our data loader
            extracted_file = os.path.join(DATA_DIR, csv_name)
            if os.path.exists(extracted_file):
                os.rename(extracted_file, final_path)
                logger.info(f"Renamed {extracted_file} to {final_path}")

            # Clean up the .7z archive to save disk space
            os.remove(archive_path)

    logger.info(
        "Real Kaggle data downloaded and extracted! Note: user_logs.csv is VERY large. You may want to sample it if your computer crashes during EDA/Modeling."
    )


if __name__ == "__main__":
    main()
