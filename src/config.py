"""
Central configuration constants.

Single source of truth for reproducibility seeds, split/CV settings, the churn
window, and synthetic-data sizes. Import these instead of scattering magic
numbers across modules.
"""

# Reproducibility
RANDOM_STATE: int = 42

# Train/test split and cross-validation
TEST_SIZE: float = 0.2
CV_FOLDS: int = 3
SEARCH_ITERATIONS: int = 5

# Churn definition: a user is labeled churned if they fail to renew within this
# many days of their membership expiring. Also used as the feature cutoff offset
# (features are built from data on or before max_transaction_date - CHURN_WINDOW_DAYS).
CHURN_WINDOW_DAYS: int = 30

# Segmentation
N_CLUSTERS: int = 3

# Synthetic data generation
MOCK_NUM_USERS: int = 5000
KAGGLE_SAMPLE_SIZE: int = 50_000
