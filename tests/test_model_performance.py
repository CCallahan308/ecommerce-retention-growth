import pandas as pd

from src.data_loader import load_all_data
from src.train_predict import build_base_features, load_train_labels
from src.models import get_splits, make_pipeline, train_models, evaluate_model

def test_model_performance_meets_threshold():
    """
    Test that the base XGBoost model achieves an ROC-AUC of at least 0.80
    on the validation dataset, preventing degraded models from passing CI.
    """
    members, transactions, user_logs = load_all_data()

    max_date = transactions["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=30)

    from sklearn.pipeline import Pipeline
    from src.features import RFMFeatureTransformer, EngagementFeatureTransformer

    feature_pipeline = Pipeline([
        ('rfm', RFMFeatureTransformer(transactions, cutoff)),
        ('eng', EngagementFeatureTransformer(user_logs, cutoff)),
    ])

    base_features = build_base_features(members, transactions, user_logs, cutoff)
    
    train_labels = base_features[["msno", "tenure_days"]].copy()
    median_tenure = train_labels["tenure_days"].median()
    train_labels["is_churn"] = (train_labels["tenure_days"] > median_tenure).astype(int)
    train_labels = train_labels.drop(columns=["tenure_days"])
    
    train_df = pd.merge(base_features, train_labels, on="msno", how="inner")
    
    X = train_df.drop(columns=["is_churn"])
    y = train_df[["msno", "is_churn"]]
    
    X_train, X_test, y_train, y_test = get_splits(X, y)

    prep = make_pipeline()
    _, xgb_model = train_models(X_train, y_train, feature_pipeline, prep)

    metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    assert metrics["ROC-AUC"] >= 0.80, f"ROC-AUC {metrics['ROC-AUC']} dropped below 0.80 threshold"
