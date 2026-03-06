import sys; sys.path.insert(0, '.')
from src.data_loader import load_all_data
from src.train_predict import build_base_features
from src.features import RFMFeatureTransformer, EngagementFeatureTransformer
from src.models import get_splits, make_pipeline
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import xgboost as xgb

members, transactions, user_logs = load_all_data()
max_date = transactions['transaction_date'].max()
cutoff = max_date - pd.Timedelta(days=30)
base_features = build_base_features(members, transactions, user_logs, cutoff)

train_labels = base_features[["msno", "tenure_days"]].copy()
median_tenure = train_labels["tenure_days"].median()
train_labels["is_churn"] = (train_labels["tenure_days"] > median_tenure).astype(int)
train_labels = train_labels.drop(columns=["tenure_days"])

train_df = pd.merge(base_features, train_labels, on="msno", how="inner")
X = train_df.drop(columns=["is_churn"])
y = train_df[["msno", "is_churn"]]

X_train, X_test, y_train, y_test = get_splits(X, y)

feature_pipeline = Pipeline([
    ("rfm", RFMFeatureTransformer(transactions, cutoff)),
    ("eng", EngagementFeatureTransformer(user_logs, cutoff))
])
prep = make_pipeline()

xgb_base = xgb.XGBClassifier(random_state=42, eval_metric="logloss")

xg_pipe = Pipeline([
    ("features", feature_pipeline),
    ("preprocessor", prep),
    ("classifier", xgb_base),
])

param_dist = {
    "classifier__max_depth": [3],
    "classifier__learning_rate": [0.1],
    "classifier__n_estimators": [100],
    "classifier__subsample": [0.8],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    xg_pipe, param_distributions=param_dist, n_iter=1, scoring="roc_auc", cv=cv, verbose=3, error_score='raise'
)
try:
    search.fit(X_train, y_train)
except Exception as e:
    import traceback
    traceback.print_exc()
