import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def segment_users_kmeans(X: pd.DataFrame):
    """
    K-Means clustering on RFM + engagement.
    Groups users into personas: whales, power users, casual.
    """
    print("running kmeans...")
    X_segment = X.copy()

    cluster_features = [
        "recency",
        "frequency",
        "monetary_total",
        "total_secs_60d",
        "active_days_60d",
    ]

    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                cluster_features,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", prep),
            ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=10)),
        ]
    )

    X_segment["cluster"] = pipeline.fit_predict(X_segment[cluster_features])

    centroids = X_segment.groupby("cluster")[cluster_features].mean()
    high_val = centroids["monetary_total"].idxmax()
    high_eng = centroids["total_secs_60d"].idxmax()

    def label(c):
        if c == high_val:
            return "High-Value Whales"
        elif c == high_eng:
            return "Power Users"
        return "Casual"

    X_segment["persona"] = X_segment["cluster"].apply(label)
    return X_segment


def baseline_segments(X: pd.DataFrame):
    """Rule-based segments for comparison with ML approach."""
    X_seg = X.copy()
    med = X_seg["monetary_total"].median()

    def assign(row):
        if row["monetary_total"] > med and row["recency"] > 30:
            return "High-Value Dormant"
        if row["active_days_30d"] > 15 and row["recency"] < 10:
            return "Highly Engaged Active"
        if row["recency"] > 45:
            return "Churned/Lost"
        return "Average Active"

    X_seg["rule_segment"] = X_seg.apply(assign, axis=1)
    return X_seg


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.data_loader import load_all_data
    from src.features import engineer_features

    m, t, u = load_all_data()

    max_date = t["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=30)

    X, y = engineer_features(m, t, u, cutoff)

    km = segment_users_kmeans(X)
    rules = baseline_segments(km)

    print("\nCross-tab:")
    print(pd.crosstab(rules["persona"], rules["rule_segment"]))
