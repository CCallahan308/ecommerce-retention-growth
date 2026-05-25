"""Tests for modeling utilities that don't require the full data pipeline."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import calibrate_model


def _toy_model_and_data():
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        weights=[0.8, 0.2],  # mild imbalance, like churn
        random_state=0,
    )
    import pandas as pd

    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X, y)
    return model, X, pd.Series(y)


def test_calibrate_model_returns_valid_probabilities():
    """Calibration must return a fitted classifier with well-formed probabilities."""
    model, X, y = _toy_model_and_data()

    calibrated = calibrate_model(model, X, y)
    proba = calibrated.predict_proba(X)[:, 1]

    assert proba.shape == (len(X),)
    assert np.all(np.isfinite(proba))
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert proba.std() > 0.0  # not a constant predictor
