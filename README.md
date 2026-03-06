<div align="center">
<<<<<<< HEAD
=======

>>>>>>> fc96c3c8e8f5592916d7213f01e81960d5e991a4
  <h1>Subscription Churn Prediction & Retention ROI</h1>
  <p><strong>Predicting 30-day churn and segmenting users to maximize retention campaign ROI.</strong></p>

  <p>
    <a href="https://github.com/CCallahan308/ecommerce-retention-growth/actions/workflows/validation.yml"><img src="https://github.com/CCallahan308/ecommerce-retention-growth/actions/workflows/validation.yml/badge.svg" alt="Validation"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python Version"></a>
    <a href="https://github.com/CCallahan308/ecommerce-retention-growth/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
    <a href="https://github.com/CCallahan308/ecommerce-retention-growth/tree/main"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style: Black"></a>
  </p>
  
  <p>
<<<<<<< HEAD
    <a href="https://ccallahan308.github.io/ecommerce-retention-growth/"><strong>Explore the Full Documentation Site</strong></a>
=======
>>>>>>> fc96c3c8e8f5592916d7213f01e81960d5e991a4
  </p>
</div>

---

<<<<<<< HEAD
## Abstract
=======
##  Executive Summary
>>>>>>> fc96c3c8e8f5592916d7213f01e81960d5e991a4

This study presents a robust machine learning pipeline designed to predict customer churn in digital subscription models, enabling precise optimization of retention interventions. Utilizing the WSDM KKBox Churn Prediction Challenge dataset, comprising over 400 million daily log records and 21 million billing histories, models are developed to forecast subscriber departure within a 30-day window. By integrating a predictive XGBoost probability classifier with K-Means driven lifetime value segmentation, the framework shifts strategic focus from blanket marketing to targeted causal interventions. Performance evaluation on a holdout sample of 484,000 users validates the superiority of the optimized gradient boosting model, which achieves a 0.8411 ROC-AUC, outperforming the logistic regression baseline.

## Methodology

<<<<<<< HEAD
### Data Preprocessing and Leakage Prevention
=======
##  Business Impact
>>>>>>> fc96c3c8e8f5592916d7213f01e81960d5e991a4

The raw dataset undergoes rigorous temporal truncation relative to a standardized 30-day expiration window. Scikit-learn compatible `BaseEstimator` and `TransformerMixin` classes govern robust feature engineering across numerical and categorical modalities. This ensures independent transformations over Training and Validation splits.

### Model Architecture and Objective Formulation

The primary risk estimation engine employs eXtreme Gradient Boosting (XGBoost). The optimization protocol formally targets the reduction of the binary Cross Entropy (Log-Loss) metric over the instance probability distribution:

$$
\mathcal{L}(\theta) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

<<<<<<< HEAD
### Hyperparameter Optimization Search Space

A `RandomizedSearchCV` implements a stratified 5-fold cross-validation mechanism to regulate model capacity against overfitting constraints. Parameters searched include combinations evaluating the number of gradient estimators (`n_estimators`), maximum tree depth (`max_depth`), feature subsetting distributions (`colsample_bytree`), stochastic descent updates (`learning_rate`), and row-wise aggregation (`subsample`).
=======
##  Technical Implementation

My approach covers the complete Data Science lifecycle:
>>>>>>> fc96c3c8e8f5592916d7213f01e81960d5e991a4

### User Segmentation via K-Means Clustering

Behavioral signals representing engagement and historical fiscal contributions inform orthogonal separation boundaries. K-Means clustering algorithmically minimizes the sum of squared intra-cluster distances to group the population:

<<<<<<< HEAD
$$
J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \| x_i - \mu_j \|^2
$$
=======
##  Performance Metrics
>>>>>>> fc96c3c8e8f5592916d7213f01e81960d5e991a4

Empirical centroid alignment divides populations into high-value ('Whales'), frequent-usage ('Power Users'), and intermittent interaction ('Casuals') segments.

## Causal Intervention Strategy

Blanket retention strategies indiscriminately allocate capital, suffering severe inefficiency due to individuals who possess low departure probability, or, inversely, negative intervention responsiveness. The present architecture reformulates the business objective via a strategic causal inference framework.

Intervention assignments target a synthesized Average Treatment Effect (ATE) across the segmented 484,000 holdout observations. Instead of uniform budget depletion, marketing incentives act as the continuous treatment variable $T$, conditionally distributed upon the individual propensity of churn $P(Y=1 | X)$ and the forecasted fiscal utility of the user context $V_i$. The optimal target policy exclusively administers bounds where marginal LTV rescue probability exceeds treatment expenditures.

<div align="center">
  <img src="figures/roi_comparison.png" alt="ROI Intervention Effectiveness Comparison" width="80%">
</div>

The intervention boundaries conditionally optimize Average Treatment Effect on the Treated (ATT), enforcing efficient capital velocity parameters over indiscriminative control segments.

## Performance Metrics

Evaluation constraints enforce strict testing isolation against a labeled subset of 484,000 users. Structural metric comparisons against baseline estimates are indicated beneath.

| Evaluation Metric | Logistic Regression (Baseline) | XGBoost (Optimized) | Relative Delta |
| :--- | :--- | :--- | :--- |
| **LogLoss** | 0.6153 | **0.4802** | -22.0% |
| **ROC-AUC** | 0.7264 | **0.8411** | +15.8% |
| **PR-AUC**  | 0.2749 | **0.5157** | +87.6% |

<<<<<<< HEAD
Additionally, SHAP (SHapley Additive exPlanations) values provide local output explanations derived from cooperative game theory rules, ensuring high-dimension feature attribution transparency over individual risk scorings.
=======
##  Quick Start
>>>>>>> fc96c3c8e8f5592916d7213f01e81960d5e991a4

<div align="center">
  <img src="figures/shap_summary.png" alt="SHAP TreeExplainer Summary Dependencies" width="80%">
</div>

## Repository Infrastructure

<<<<<<< HEAD
The core computation pipelines, module classes, and reproducible evaluations operate inside standardized dependency topologies configured by `pyproject.toml` and managed by the `uv` build backend.
=======
# 2. Get Mock Data (For quick testing without the 30GB Kaggle download)
python src/generate_mock_data.py

# 3. Train & Predict
python src/train_predict.py
```

> Outputs will be saved to `data/processed/predictions.csv`.

For a more exploratory dive, launch the interactive notebooks:

```bash
jupyter notebook notebooks/02_business_impact_scenarios.ipynb
```

##  Repository Structure
>>>>>>> fc96c3c8e8f5592916d7213f01e81960d5e991a4

```text
├── data/                  # Transient computational stores
├── docs/                  # MkDocs GitHub Pages assets
├── figures/               # Artifact representations and visual distributions
├── notebooks/             # Exploratory spatial analysis structures
├── src/                   # Pipeline transformers and mathematical formulations
├── tests/                 # Automated cross-architecture PyTest verification routines
├── Makefile               # Automated compilation and execution directives
├── pyproject.toml         # Explicit deterministic dependency configurations
└── README.md              # Research document
```

### Execution Directives

Execution requires Python 3.9 or greater. A complete environment bootstrap utilizes modern generic build frameworks (`uv`).

```bash
# 1. Instantiate the deterministic dependency graph
uv venv
source .venv/bin/activate
uv pip install -e .[dev]

# 2. Replicate synthetic localized variables (avoids 30GB dependency retrieval)
make data

# 3. Model convergence validation and prediction execution
make train

# 4. Enforce PEP static analysis checks and CI/CD validation
make lint
make test
```

Interactive exploration formats simulating structural hypotheses evaluate business intervention parameters within isolated kernel sequences: `jupyter notebook notebooks/02_business_impact_scenarios.ipynb`.

---
<div align="center">
  <p>Engineered with formal mathematical robustness using Python</p>
</div>
