"""
Predictive Modeling Module.

Trains and evaluates predictive models for churn forecasting. Uses strict pipeline
architectures to prevent data leakage during scaling/imputation. Includes a simple
interpretable baseline (Logistic Regression) and a complex non-linear model (XGBoost).
"""

import logging

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def prepare_data_splits(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Parameters
    ----------
    X : pd.DataFrame
        Features dataframe.
    y : pd.DataFrame
        Target dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test

    """
    # Drop MSNO as it's an identifier, not a feature
    X_clean = X.drop(columns=["msno"])
    y_clean = y["is_churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Construct a scikit-learn preprocessing pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features to infer column types.

    Returns
    -------
    ColumnTransformer
        Transformer mapping categorical to OneHot and numeric to Scaled.

    """
    numeric_features = X_train.select_dtypes(include=["int64", "float64", "Int16", "Int32", "Int8", "float32"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Pipeline:
    """Train a baseline Logistic Regression model."""
    logger.info("Training Logistic Regression baseline...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Pipeline:
    """Train an XGBoost classifier."""
    logger.info("Training XGBoost model...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(), # Handle imbalance
            random_state=42,
            max_depth=4, # Prevent overfitting on small mock data
            learning_rate=0.1,
            n_estimators=100
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict[str, float]:
    """
    Evaluate the model on holdout set using business-relevant metrics.

    Parameters
    ----------
    model : Pipeline
        Trained scikit-learn pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test targets.
    model_name : str
        Name for logging.

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics (ROC-AUC, PR-AUC, Brier).

    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "ROC-AUC": float(roc_auc_score(y_test, y_pred_proba)),
        "PR-AUC": float(average_precision_score(y_test, y_pred_proba)),
        "Brier-Score": float(brier_score_loss(y_test, y_pred_proba))
    }

    logger.info(f"--- {model_name} Evaluation ---")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    import os
    import sys
    from datetime import datetime
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.data_loader import load_all_data
    from src.features import engineer_features

    m, t, u = load_all_data()
    
    # Set CUTOFF dynamically based on the dataset's date range
    max_date = t["transaction_date"].max()
    CUTOFF = max_date - pd.Timedelta(days=30)
    
    X, y = engineer_features(m, t, u, CUTOFF)

    X_train, X_test, y_train, y_test = prepare_data_splits(X, y)
    preprocessor = build_preprocessor(X_train)

    lr_model = train_logistic_regression(X_train, y_train, preprocessor)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    xgb_model = train_xgboost(X_train, y_train, preprocessor)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
