<div align="center">
  <h1>Subscription Churn Prediction & Retention ROI</h1>
  <p><strong>Predicting 30-day churn and segmenting users to maximize retention campaign ROI.</strong></p>

  <p>
    <a href="https://github.com/CCallahan308/ecommerce-retention-growth/actions/workflows/validation.yml"><img src="https://github.com/CCallahan308/ecommerce-retention-growth/actions/workflows/validation.yml/badge.svg" alt="Validation"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python Version"></a>
    <a href="https://github.com/CCallahan308/ecommerce-retention-growth/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
    <a href="https://github.com/CCallahan308/ecommerce-retention-growth/tree/main"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style: Black"></a>
  </p>
  
  <p>
    <a href="https://ccallahan308.github.io/ecommerce-retention-growth/"><strong>Full Documentation</strong></a>
  </p>
</div>

---

## What this does

Predicts which subscription users will churn in the next 30 days, segments them by value, and figures out where retention spend actually pays off.

Built on the WSDM KKBox dataset - 400M+ daily logs and 21M billing records. The pipeline trains an XGBoost classifier, clusters users by lifetime value, and runs an ROI simulator to show where marketing budget gets the best return.

## Results

Trained and tested on a 484K user holdout set:

| Metric | Logistic Regression | XGBoost | Improvement |
|:-------|:-------------------:|:-------:|:-----------:|
| LogLoss | 0.6153 | 0.4802 | -22% |
| ROC-AUC | 0.7264 | 0.8411 | +16% |
| PR-AUC | 0.2749 | 0.5157 | +88% |

XGBoost significantly outperformed the baseline, especially on precision-recall where class imbalance usually kills performance.

## How it works

### Churn model

XGBoost classifier optimized via stratified 5-fold cross-validation. The loss function is standard binary cross-entropy:

```
L(θ) = -1/N Σ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]
```

Hyperparameter search covers tree depth, learning rate, column/row subsampling, and estimator count.

### User segmentation

K-Means on behavioral + revenue features splits users into:
- **Whales** - high LTV, low churn risk
- **Power Users** - frequent usage, medium value
- **Casuals** - intermittent engagement

This matters because spending $10 to retain a $1000 LTV user makes sense. Spending $10 on someone worth $5 doesn't.

### ROI simulator

Instead of blasting everyone with the same retention offer, the simulator asks: for each segment, what's the expected return on a given intervention cost?

<div align="center">
  <img src="figures/roi_comparison.png" alt="ROI by segment" width="80%">
</div>

### Model interpretation

SHAP values show which features drive churn predictions:

<div align="center">
  <img src="figures/shap_summary.png" alt="SHAP feature importance" width="80%">
</div>

## Quick start

```bash
# clone and setup
git clone https://github.com/CCallahan308/ecommerce-retention-growth.git
cd ecommerce-retention-growth

uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .[dev]

# generate mock data (skip the 30GB Kaggle download)
make data

# train the model
make train

# run tests
make test
```

For the interactive business scenarios notebook:

```bash
jupyter notebook notebooks/02_business_impact_scenarios.ipynb
```

## Repo structure

```
├── data/          # raw and processed data (gitignored)
├── docs/          # MkDocs site
├── figures/       # plots and visualizations
├── notebooks/     # exploratory analysis
│   └── 02_business_impact_scenarios.ipynb  # (01_eda.ipynb not yet included)
├── src/           # pipeline code
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── segmentation.py
│   └── train_predict.py
├── tests/         # pytest
├── Makefile
└── pyproject.toml
```

## Data source

WSDM KKBox Churn Prediction Challenge on Kaggle. You'll need Kaggle API credentials to download the full dataset, or use `make data` to generate synthetic data for testing.

## Requirements

Python 3.9+, dependencies in `pyproject.toml`. Key packages: XGBoost, scikit-learn, pandas, SHAP.
