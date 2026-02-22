"""
Unsupervised Segmentation Module.

Clusters users into actionable business segments using KMeans based on
historical RFM and engagement features. Maps quantitative clusters into
qualitative personas suitable for marketing strategy.
"""

import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def build_clustering_pipeline(features: list) -> Pipeline:
    """
    Build a scaling and KMeans pipeline.

    Parameters
    ----------
    features : list
        List of numeric features to use for clustering.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline for clustering.

    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=10))
    ])

    return pipeline


def segment_users_kmeans(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply KMeans clustering to identify user personas.

    Parameters
    ----------
    X : pd.DataFrame
        Engineered feature dataframe containing 'msno' and numeric features.

    Returns
    -------
    pd.DataFrame
        Original dataframe with 'cluster' and 'persona' columns appended.

    """
    logger.info("Performing KMeans segmentation...")
    X_segment = X.copy()

    # We select features that indicate value and engagement
    cluster_features = [
        "recency",
        "frequency",
        "monetary_total",
        "total_secs_60d",
        "active_days_60d"
    ]

    pipeline = build_clustering_pipeline(cluster_features)

    # Fit and predict
    X_segment["cluster"] = pipeline.fit_predict(X_segment[cluster_features])

    # Analyze cluster centroids to assign logical names
    centroids = X_segment.groupby("cluster")[cluster_features].mean()

    # Heuristic mapping based on centroids
    # Identify the highest value cluster and highest engagement
    high_value_cluster = centroids["monetary_total"].idxmax()
    high_eng_cluster = centroids["total_secs_60d"].idxmax()

    def map_persona(c: int) -> str:
        if c == high_value_cluster:
            return "High-Value Whales"
        elif c == high_eng_cluster:
            return "Power Users"
        else:
            return "Standard/Casual Users"

    X_segment["persona"] = X_segment["cluster"].apply(map_persona)

    logger.info(f"Segment Distribution:\n{X_segment['persona'].value_counts(normalize=True)}")
    return X_segment


def rule_based_segmentation(X: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline heuristic segmentation using explicit business rules.

    Parameters
    ----------
    X : pd.DataFrame
        Engineered features.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'rule_segment' appended.

    """
    logger.info("Applying rule-based (RFM) segmentation baseline...")
    X_seg = X.copy()

    # Example logic mapping to the Business Case Study:
    # 1) High-Value At-Risk: Monetary > Median AND Recency > 30
    # 2) Engaged Advocates: Active Days > 15 in last 30 AND Recency < 10
    # 3) At-Risk Low-Value: Remaining

    med_monetary = X_seg["monetary_total"].median()

    def assign_rule(row: pd.Series) -> str:
        if row["monetary_total"] > med_monetary and row["recency"] > 30:
            return "High-Value Dormant"
        elif row["active_days_30d"] > 15 and row["recency"] < 10:
            return "Highly Engaged Active"
        elif row["recency"] > 45:
            return "Churned/Lost"
        else:
            return "Average Active"

    X_seg["rule_segment"] = X_seg.apply(assign_rule, axis=1)

    return X_seg


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

    # Unsupervised doesn't strictly need train/test split, applying to all
    segmented_kmeans = segment_users_kmeans(X)
    segmented_rules = rule_based_segmentation(segmented_kmeans)

    print("\nFinal Persona Cross-tab with Rule Baseline:")
    print(pd.crosstab(segmented_rules["persona"], segmented_rules["rule_segment"]))
