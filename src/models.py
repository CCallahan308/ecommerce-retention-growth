import logging

import pandas as pd
import xgboost as xgb
from typing import Tuple, Dict, Any

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

logger = logging.getLogger(__name__)


def get_splits(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets, stratifying on the target.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame containing 'msno' and predictive columns.
    y : pd.DataFrame
        Targets DataFrame containing 'is_churn'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test arrays.
    """
    np.random.seed(42)
    y_clean = y["is_churn"]
    return train_test_split(
        X, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )


def make_pipeline() -> ColumnTransformer:
    """
    Prepare the scikit-learn ColumnTransformer for preprocessing numeric and categorical features.

    Returns
    -------
    ColumnTransformer
        A preprocessor ready to scale and encode features. dynamically captures numeric/categorical upon fit.
    """
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipe, make_column_selector(dtype_include=np.number, pattern="^(?!msno$).*$")),
            ("cat", cat_pipe, make_column_selector(dtype_exclude=np.number, pattern="^(?!msno$).*$")),
        ],
        remainder="drop"
    )

def train_models(X_train: pd.DataFrame, y_train: pd.Series, feature_pipeline: Pipeline, prep: ColumnTransformer) -> Tuple[Pipeline, Pipeline]:
    """
    Train and tune baseline and optimized machine learning models (Logistic Regression, XGBoost).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    feature_pipeline: Pipeline
        The scikit-learn pipeline representing our custom feature mergers.
    prep : ColumnTransformer
        Column transformation pipeline for standardizing and encoding.

    Returns
    -------
    Tuple[Pipeline, Pipeline]
        Tuple of fitted (logistic_regression_pipeline, xgboost_pipeline).
    """
    np.random.seed(42)
    logger.info("Training Logistic Regression...")
    lr = Pipeline(
        [
            ("features", feature_pipeline),
            ("preprocessor", prep),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", random_state=42, max_iter=1000
                ),
            ),
        ]
    )
    lr.fit(X_train, y_train)

    logger.info("Tuning XGBoost with RandomizedSearchCV...")
    # Base XGBoost without hardcoded params
    xgb_base = xgb.XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
        random_state=42,
    )

    xg_pipe = Pipeline(
        [
            ("features", feature_pipeline),
            ("preprocessor", prep),
            ("classifier", xgb_base),
        ]
    )

    # Simplified search space to avoid extremely long train times during testing
    param_distributions = {
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__n_estimators": [50, 100, 200],
        "classifier__subsample": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    xg_search = RandomizedSearchCV(
        estimator=xg_pipe,
        param_distributions=param_distributions,
        n_iter=5, # Keep it small for reasonable runtime
        scoring="neg_log_loss",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=1,
    )
    
    xg_search.fit(X_train, y_train)
    
    logger.info(f"Best XGBoost params: {xg_search.best_params_}")
    
    return lr, xg_search.best_estimator_


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> Dict[str, float]:
    """
    Evaluate a fitted model on test data logging ROC-AUC, PR-AUC, LogLoss, and Brier-Score.

    Parameters
    ----------
    model : Any
        The fitted estimator pipeline with `predict_proba` access.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test actual labels.
    name : str
        Name of model for standard output logging.

    Returns
    -------
    Dict[str, float]
        Dictionary mapped metric names to scores.
    """
    proba = model.predict_proba(X_test)[:, 1].clip(1e-15, 1 - 1e-15)

    metrics = {
        "ROC-AUC": float(roc_auc_score(y_test, proba)),
        "PR-AUC": float(average_precision_score(y_test, proba)),
        "LogLoss": float(log_loss(y_test, proba)),
        "Brier-Score": float(brier_score_loss(y_test, proba)),
    }

    print(f"--- {name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    # NOTE: for the real pipeline using Kaggle labels, run train_predict.py
    # this one uses heuristic labels
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.data_loader import load_all_data
    from src.features import engineer_features

    from src.features import RFMFeatureTransformer, EngagementFeatureTransformer

    m, t, u = load_all_data()

    max_date = t["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=30)
    
    feature_pipeline = Pipeline([
        ('rfm', RFMFeatureTransformer(t, cutoff)),
        ('eng', EngagementFeatureTransformer(u, cutoff))
    ])

    X, y = engineer_features(m, t, u, cutoff)
    X_train, X_test, y_train, y_test = get_splits(X, y)
    prep = make_pipeline()

    lr, xgb_model = train_models(X_train, y_train, feature_pipeline, prep)
    evaluate_model(lr, X_test, y_test, "Logistic Regression")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")
