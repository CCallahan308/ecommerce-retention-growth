import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def segment_users_kmeans(X: pd.DataFrame):
    """Cluster users to find the whales."""
    print("running kmeans...")
    X_segment = X.copy()

    # subset of features for clustering
    cluster_features = [
        "recency",
        "frequency",
        "monetary_total",
        "total_secs_60d",
        "active_days_60d"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), cluster_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=10))
    ])

    X_segment["cluster"] = pipeline.fit_predict(X_segment[cluster_features])

    centroids = X_segment.groupby("cluster")[cluster_features].mean()

    # Heuristic mapping
    high_value_cluster = centroids["monetary_total"].idxmax()
    high_eng_cluster = centroids["total_secs_60d"].idxmax()

    def get_label(c: int) -> str:
        if c == high_value_cluster:
            return "High-Value Whales"
        elif c == high_eng_cluster:
            return "Power Users"
        else:
            return "Casual"

    X_segment["persona"] = X_segment["cluster"].apply(get_label)
    return X_segment


def baseline_segments(X: pd.DataFrame):
    # basic rule of thumb baseline
    X_seg = X.copy()

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
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.data_loader import load_all_data
    from src.features import engineer_features

    m, t, u = load_all_data()

    max_date = t["transaction_date"].max()
    CUTOFF = max_date - pd.Timedelta(days=30)

    X, y = engineer_features(m, t, u, CUTOFF)

    segmented_kmeans = segment_users_kmeans(X)
    segmented_rules = baseline_segments(segmented_kmeans)

    print("\nCross-tab:")
    print(pd.crosstab(segmented_rules["persona"], segmented_rules["rule_segment"]))
