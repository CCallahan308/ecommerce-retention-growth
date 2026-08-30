"""
Business-impact / ROI analysis.

Segments users by value, scores churn risk on a HELD-OUT test cohort (so the
business case reflects customers the model has not seen), and compares a
targeted retention strategy against a blanket discount. Produces the figures
referenced by ``docs/business_impact.md``.

Run:
    python src/business_impact.py

Outputs (saved to figures/ and mirrored to docs/figures/):
    confusion_matrix.png, shap_summary.png, roi_comparison.png

NOTE: the campaign constants below are ILLUSTRATIVE planning assumptions, not
measured outcomes. A production readout would source the save-rate from an
actual A/B holdout and LTV from finance, not from a fixed guess.
"""

import logging
import os
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CHURN_WINDOW_DAYS
from src.data_loader import load_all_data
from src.features import (
    EngagementFeatureTransformer,
    RFMFeatureTransformer,
    engineer_features,
)
from src.models import get_splits, make_pipeline, train_models
from src.segmentation import segment_users_kmeans

# Headless backend: this is a script, not an interactive session. Set after
# imports (no figures exist yet) to keep the import block contiguous.
matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Illustrative campaign assumptions (planning inputs, NOT measured results) ---
AVERAGE_LTV = 150.0  # assumed lifetime value of a retained user, in dollars
DISCOUNT_COST = 30.0  # cost of the retention offer per targeted user
CAMPAIGN_SUCCESS_RATE = 0.25  # assumed fraction of targeted true-churners saved
TARGET_PERSONAS = ("High-Value Whales", "Power Users")
TOP_RISK_SHARE = 0.20  # target the top 20% highest-risk users within those personas

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FIG_DIR = os.path.join(_PROJECT_ROOT, "figures")
DOCS_FIG_DIR = os.path.join(_PROJECT_ROOT, "docs", "figures")

sns.set_theme(style="white", context="talk")
plt.rcParams.update({"figure.dpi": 150, "axes.titleweight": "bold"})


def _save(fig_name: str) -> None:
    """Save the current figure to figures/ and mirror it into docs/figures/."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, fig_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    if os.path.isdir(DOCS_FIG_DIR):
        shutil.copyfile(path, os.path.join(DOCS_FIG_DIR, fig_name))
    logger.info("Saved %s", fig_name)


def plot_confusion_matrix(y_test: pd.Series, test_probs: np.ndarray) -> None:
    """Confusion matrix at the 0.5 threshold, on the held-out test split."""
    cm = confusion_matrix(y_test, test_probs > 0.5)
    _fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(
        cmap="GnBu", ax=ax, values_format="d", colorbar=False
    )
    ax.grid(False)
    ax.set_title("Prediction vs. Reality (test split)", pad=20)
    _save("confusion_matrix.png")


def plot_shap_summary(
    xgb_model: Pipeline, feature_pipeline: Pipeline, X_train: pd.DataFrame
) -> None:
    """SHAP summary of churn drivers, using the model's own fitted preprocessor."""
    classifier = xgb_model.named_steps["classifier"]
    fitted_prep = xgb_model.named_steps["preprocessor"]

    X_train_features = feature_pipeline.transform(X_train)
    X_train_transformed = fitted_prep.transform(X_train_features)
    feature_names = [c.split("__")[-1] for c in fitted_prep.get_feature_names_out()]

    sample = pd.DataFrame(X_train_transformed[:1000], columns=feature_names)
    shap_values = shap.TreeExplainer(classifier).shap_values(sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, show=False)
    plt.title("Top Drivers of Churn Risk (SHAP)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    _save("shap_summary.png")


def compute_roi(segmented_test: pd.DataFrame) -> pd.DataFrame:
    """
    Compare a blanket retention campaign against an ML-targeted one on the
    held-out test cohort. Returns a tidy DataFrame for plotting.
    """
    n_population = len(segmented_test)
    blanket_cost = n_population * DISCOUNT_COST
    blanket_saved = segmented_test["is_churn_actual"].sum() * CAMPAIGN_SUCCESS_RATE
    blanket_roi = blanket_saved * AVERAGE_LTV - blanket_cost

    high_value = segmented_test[segmented_test["persona"].isin(TARGET_PERSONAS)]
    pool = high_value if len(high_value) > 0 else segmented_test
    n_target = max(1, round(len(pool) * TOP_RISK_SHARE))
    smart = pool.nlargest(n_target, "churn_probability")

    smart_cost = len(smart) * DISCOUNT_COST
    smart_saved = smart["is_churn_actual"].sum() * CAMPAIGN_SUCCESS_RATE
    smart_roi = smart_saved * AVERAGE_LTV - smart_cost

    logger.info(
        "Blanket: target %d, cost $%.0f, net ROI $%.0f",
        n_population,
        blanket_cost,
        blanket_roi,
    )
    logger.info(
        "Smart:   target %d, cost $%.0f, net ROI $%.0f",
        len(smart),
        smart_cost,
        smart_roi,
    )
    return pd.DataFrame(
        {
            "Strategy": ["Blanket", "Smart (ML)", "Blanket", "Smart (ML)"],
            "Metric": ["Campaign Cost", "Campaign Cost", "Net ROI", "Net ROI"],
            "Value": [blanket_cost, smart_cost, blanket_roi, smart_roi],
        }
    )


def plot_roi_comparison(roi_data: pd.DataFrame) -> None:
    """Bar chart of campaign cost and net ROI for each strategy."""
    palette = {"Blanket": "#4A4E69", "Smart (ML)": "#2A9D8F"}

    def dollars(x: float, _pos: int = 0) -> str:
        return f"{'-' if x < 0 else ''}${abs(x):,.0f}"

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=roi_data, x="Metric", y="Value", hue="Strategy", palette=palette
    )
    ax.yaxis.set_major_formatter(FuncFormatter(dollars))
    ax.set_title("Blanket vs. Targeted Retention (test cohort)", pad=20)
    ax.set_ylabel("Dollars")
    ax.set_xlabel("")
    plt.axhline(0, color="#444444", linewidth=1.0)
    for patch in ax.patches:
        height = float(patch.get_height())
        if abs(height) < 1e-9:
            continue
        ax.annotate(
            dollars(height),
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom" if height > 0 else "top",
            xytext=(0, 6 if height > 0 else -10),
            textcoords="offset points",
            fontsize=11,
            weight="bold",
        )
    plt.tight_layout()
    _save("roi_comparison.png")


def build_test_segments(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_pipeline: Pipeline,
    xgb_model: Pipeline,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Segment the full cohort (clustering is unsupervised) on engineered features,
    then keep only the held-out test users and attach their predicted churn
    probability and actual label.
    """
    X_features = feature_pipeline.transform(X)
    segmented_all = segment_users_kmeans(X_features)

    test_probs = xgb_model.predict_proba(X_test)[:, 1]
    prob_by_id = pd.Series(test_probs, index=X_test["msno"].values)
    label_by_id = pd.Series(y_test.values, index=X_test["msno"].values)

    segmented = segmented_all[segmented_all["msno"].isin(X_test["msno"])].copy()
    segmented["churn_probability"] = segmented["msno"].map(prob_by_id)
    segmented["is_churn_actual"] = segmented["msno"].map(label_by_id)
    return segmented, test_probs


def main() -> None:
    members, transactions, user_logs = load_all_data()
    cutoff = transactions["transaction_date"].max() - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    logger.info("Feature cutoff date: %s", cutoff)

    X, y = engineer_features(members, transactions, user_logs, cutoff)
    X_train, X_test, y_train, y_test = get_splits(X, y)

    feature_pipeline = Pipeline(
        [
            ("rfm", RFMFeatureTransformer(transactions, cutoff)),
            ("eng", EngagementFeatureTransformer(user_logs, cutoff)),
        ]
    )
    preprocessor = make_pipeline()
    _, xgb_model = train_models(X_train, y_train, feature_pipeline, preprocessor)

    segmented_test, test_probs = build_test_segments(
        X, X_test, y_test, feature_pipeline, xgb_model
    )

    plot_confusion_matrix(y_test, test_probs)
    plot_shap_summary(xgb_model, feature_pipeline, X_train)
    plot_roi_comparison(compute_roi(segmented_test))

    logger.info("Business-impact figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
