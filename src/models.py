import logging

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def get_splits(X, y):
    X_clean = X.drop(columns=["msno"])
    y_clean = y["is_churn"]
    return train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )


def make_pipeline(X_train: pd.DataFrame):
    """Prep pipeline - handles both numeric and categorical."""
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64", "Int16", "Int32", "Int8", "float32"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # could experiment with different strategies here
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

def train_models(X_train, y_train, prep):
    logger.info("Training Logistic Regression...")
    lr = Pipeline(
        [
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
        n_jobs=-1,
    )
    
    xg_search.fit(X_train, y_train)
    
    logger.info(f"Best XGBoost params: {xg_search.best_params_}")
    
    return lr, xg_search.best_estimator_


def evaluate_model(model, X_test, y_test, name):
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

    m, t, u = load_all_data()

    max_date = t["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=30)

    X, y = engineer_features(m, t, u, cutoff)
    X_train, X_test, y_train, y_test = get_splits(X, y)
    prep = make_pipeline(X_train)

    lr, xgb = train_models(X_train, y_train, prep)
    evaluate_model(lr, X_test, y_test, "Logistic Regression")
    evaluate_model(xgb, X_test, y_test, "XGBoost")
