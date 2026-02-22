
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_splits(X, y):
    X_clean = X.drop(columns=["msno"])
    y_clean = y["is_churn"]

    return train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )


def make_pipeline(X_train: pd.DataFrame):
    """Basic prep pipeline for numerical and categorical columns."""
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

    prep = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return prep


def train_models(X_train, y_train, prep):
    print("training LR...")
    lr = Pipeline(steps=[
        ("preprocessor", prep),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
    ])
    lr.fit(X_train, y_train)

    print("training XGBoost...")
    # TODO: try lightgbm later
    xg = Pipeline(steps=[
        ("preprocessor", prep),
        ("classifier", xgb.XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
            random_state=42,
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100
        ))
    ])
    xg.fit(X_train, y_train)

    return lr, xg


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "ROC-AUC": float(roc_auc_score(y_test, y_pred_proba)),
        "PR-AUC": float(average_precision_score(y_test, y_pred_proba)),
        "Brier-Score": float(brier_score_loss(y_test, y_pred_proba))
    }

    print(f"--- {model_name} Evaluation ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.data_loader import load_all_data
    from src.features import engineer_features

    m, t, u = load_all_data()

    max_date = t["transaction_date"].max()
    CUTOFF = max_date - pd.Timedelta(days=30)

    X, y = engineer_features(m, t, u, CUTOFF)

    X_train, X_test, y_train, y_test = get_splits(X, y)
    prep = make_pipeline(X_train)

    lr_model, xgb_model = train_models(X_train, y_train, prep)

    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")
