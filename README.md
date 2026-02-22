# Subscription Churn Prediction & Customer Value Segmentation

> Built on the [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) dataset.

Predicts 30-day subscription churn and segments users by lifetime value — trained on the **official Kaggle competition labels** (`train.csv`, 970k users), then predicts churn risk for every remaining member in the dataset.

## Approach

| Stage | What It Does |
| :--- | :--- |
| **Training Labels** | Uses the official competition `train.csv` (970,960 ground-truth labels). Also includes a Python port of the [Scala churn labeler](https://github.com/kkbox/wsdm-cup-2018-churn-prediction-challenge) in `features.py` for reference. |
| **Feature Engineering** | RFM from billing, 30/60-day engagement trends from logs, demographic tenure. Cutoff filtering prevents data leakage. |
| **Modeling** | XGBoost + Logistic Regression baseline. Uses LogLoss (official Kaggle metric), ROC-AUC, PR-AUC, and Brier Score. |
| **Prediction** | After training on labeled users, the best model scores every unlabeled member with a churn probability. |
| **Segmentation** | K-Means on RFM + engagement to find whales, power users, casuals. |
| **Business Impact** | ROI comparison of blanket vs. targeted retention campaigns. |

## Data

The pipeline uses the full KKBox competition dataset:

| File | Records | Description |
| :--- | ---: | :--- |
| `train.csv` | 970,960 | Official ground-truth churn labels (`msno`, `is_churn`) |
| `members.csv` | 6,769,473 | User demographics (city, age, gender, registration date) |
| `transactions.csv` | ~21M | Billing history (plans, payments, auto-renew, cancellations) |
| `user_logs.csv` | ~400M | Daily listening activity (songs played, listening time) |

After feature engineering, **484,496 labeled users** have sufficient activity data for training, and **205,338 unlabeled users** receive churn predictions.

## Results

### Model Performance (484k Labeled Users, 80/20 Holdout)

| Metric | Logistic Regression | XGBoost |
| :--- | :--- | :--- |
| **LogLoss** | 0.6153 | **0.4802** |
| ROC-AUC | 0.7264 | **0.8411** |
| PR-AUC | 0.2749 | **0.5157** |
| Brier Score | 0.2065 | **0.1608** |

*LogLoss is the official Kaggle competition metric.*

### Predictions on Unlabeled Users

| Stat | Value |
| :--- | :--- |
| Total scored | 205,338 |
| Predicted churn | 187,365 (91.25%) |
| Avg churn probability | 0.896 |

> Unlabeled users skew toward dormant/inactive accounts, which the model correctly identifies as high risk.

### Exploratory Data Analysis

![Transaction Trends](figures/02_transaction_trends.png)

### Confusion Matrix

![Confusion Matrix](figures/confusion_matrix.png)

### Drivers of Churn (SHAP)

![SHAP Summary](figures/shap_summary.png)

### Retention ROI: Blanket vs. Targeted Strategy

![ROI Comparison](figures/roi_comparison.png)

By targeting only the top 20% churn-risk users within high-value personas, the ML-driven strategy wastes less money on users who would have stayed anyway.

## Project Structure

```text
├── data/
│   ├── raw/                       # Raw CSVs (gitignored)
│   │   ├── train.csv              # Official Kaggle competition labels
│   │   ├── members.csv            # User demographics
│   │   ├── transactions.csv       # Billing history
│   │   └── user_logs.csv          # Daily listening logs
│   ├── processed/                 # Model outputs (gitignored)
│   │   ├── predictions.csv        # Churn scores for unlabeled users
│   │   └── train_predictions.csv  # Scores + actuals for labeled users
│   └── features/                  # Intermediate feature CSVs (gitignored)
├── figures/                       # Generated charts (committed for README)
├── notebooks/
│   └── 02_business_impact_scenarios.ipynb
├── src/
│   ├── data_loader.py             # Schema definitions and data loading
│   ├── eda.py                     # Exploratory visualizations
│   ├── extract_kaggle_data.py     # .7z archive extractor for local Kaggle files
│   ├── download_real_data.py      # Kaggle API downloader
│   ├── features.py                # Feature engineering and churn labeler (reference)
│   ├── generate_mock_data.py      # Synthetic data for quick testing
│   ├── models.py                  # Model definitions and evaluation utilities
│   ├── train_predict.py           # Main pipeline: train on train.csv, predict the rest
│   ├── sample_kaggle_data.py      # Chunked sampling for 30GB+ files
│   └── segmentation.py            # K-Means clustering
└── tests/                         # Unit tests (pytest)
```

## Quick Start

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the Data

**Option A — Mock Data (no download required):**
```bash
python src/generate_mock_data.py
```

**Option B — Real Kaggle Data:**

Download the [competition files](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data) and place the `.7z` archives in `kkbox-churn-prediction-challenge/` at the project root, then:
```bash
python src/extract_kaggle_data.py
```

Or use the Kaggle API directly:
```bash
python src/download_real_data.py
```

> **Optional:** If your machine has < 32 GB RAM, you can downsample with `python src/sample_kaggle_data.py` before training.

### 3. Train and Predict
```bash
# Main pipeline — trains on train.csv labels, predicts on unlabeled members
python src/train_predict.py
```

Outputs are saved to `data/processed/`:
- **`predictions.csv`** — churn probability for every unlabeled member
- **`train_predictions.csv`** — actual labels + predicted probabilities for training users

### 4. Supporting Scripts
```bash
python src/eda.py             # Generate EDA visualizations
python src/models.py          # Quick train/eval using heuristic labels (legacy)
python src/segmentation.py    # Run K-Means clustering
```

### 5. Business Analysis
```bash
jupyter notebook notebooks/02_business_impact_scenarios.ipynb
```

## Tests
```bash
pytest tests/
```

## Tech Stack

Python · Pandas · NumPy · XGBoost · Scikit-Learn · SHAP · Seaborn · Pytest

## License

MIT
