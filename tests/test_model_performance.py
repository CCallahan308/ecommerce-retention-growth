import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import CHURN_WINDOW_DAYS
from src.data_loader import load_all_data
from src.features import EngagementFeatureTransformer, RFMFeatureTransformer
from src.models import evaluate_model, get_splits, make_pipeline, train_models
from src.train_predict import build_base_features, load_train_labels


def test_pipeline_trains_and_predicts_on_real_labels():
    """
    Integration guard for the full train -> predict pipeline using the genuine
    heuristic churn labels (prep_targets) written to train.csv by `make data`.

    CI runs on synthetic data, so this is deliberately an *integration* test,
    not a model-quality gate -- a fixed accuracy threshold on synthetic data
    would be meaningless. It asserts the pipeline produces well-formed,
    discriminative, non-degenerate probabilities, which catches:
      - broken wiring / exceptions in the feature + model pipeline,
      - NaN or out-of-range probabilities,
      - collapse to a constant prediction,
      - a model no better than random.

    NOTE: the earlier version of this test fabricated its label as
    ``is_churn = (tenure_days > median)`` while also feeding ``tenure_days`` to
    the model, so a high ROC-AUC was mathematically guaranteed and tested
    nothing. Labels here come from ``load_train_labels`` and are independent of
    the model's input features.
    """
    members, transactions, user_logs = load_all_data()

    max_date = transactions["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=CHURN_WINDOW_DAYS)

    feature_pipeline = Pipeline(
        [
            ("rfm", RFMFeatureTransformer(transactions, cutoff)),
            ("eng", EngagementFeatureTransformer(user_logs, cutoff)),
        ]
    )

    base_features = build_base_features(members, transactions, user_logs, cutoff)

    train_labels = load_train_labels()
    train_df = pd.merge(base_features, train_labels, on="msno", how="inner")
    assert not train_df.empty, "No labeled users with features; did `make data` run?"

    X = train_df.drop(columns=["is_churn"])
    y = train_df[["msno", "is_churn"]]

    X_train, X_test, y_train, y_test = get_splits(X, y)

    prep = make_pipeline()
    _, xgb_model = train_models(X_train, y_train, feature_pipeline, prep)

    proba = xgb_model.predict_proba(X_test)[:, 1]

    # Well-formed probabilities.
    assert proba.shape == (len(X_test),)
    assert np.all(np.isfinite(proba))
    assert proba.min() >= 0.0 and proba.max() <= 1.0

    # Not a degenerate constant predictor.
    assert proba.std() > 0.0, "Model collapsed to a constant prediction."

    # Discriminative: comfortably better than random on held-out labels.
    metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    assert (
        metrics["ROC-AUC"] > 0.55
    ), f"ROC-AUC {metrics['ROC-AUC']:.3f} is not meaningfully better than random"
