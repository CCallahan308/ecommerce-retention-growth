"""
Exploratory Data Analysis Module.

Performs visual and statistical inspection of the subscription dataset to uncover
patterns in churn, engagement, and customer value. Generates static charts
styled appropriately for business reports.
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

# Allow running directly: python src/eda.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_loader import load_all_data  # noqa: E402

# Configuration for Standardized Portfolio Visuals
sns.set_theme(style="white", context="talk", palette="viridis")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#444444",
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "figure.dpi": 200,
    }
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def plot_registration_cohorts(members_df, save_path=None):
    """
    Plot the distribution of user registrations over time by gender.

    Parameters
    ----------
    members_df : pd.DataFrame
        Members dataframe.
    save_path : str, optional
        Path to save the figure.

    """
    logger.info("Plotting registration cohorts.")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract year-month for cohort grouping
    members_df["reg_month"] = (
        members_df["registration_init_time"].dt.to_period("M").dt.to_timestamp()
    )

    # Group and count
    cohort_counts = (
        members_df.groupby(["reg_month", "gender"]).size().unstack(fill_value=0)
    )

    cohort_counts.plot(
        kind="area", stacked=True, alpha=0.9, ax=ax, cmap="RdYlBu"
    )  # Nice contrasting colors for gender
    ax.set_title(
        "Long-term Registration Trends by Segment",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("Quantity of Monthly Signups")
    ax.set_xlabel("Registration Timeline")
    ax.legend(title="Gender", frameon=True, shadow=True, borderpad=1)
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_transaction_trends(transactions_df, save_path=None):
    """
    Plot the breakdown of payment methods and auto-renew status.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Transactions dataframe.
    save_path : str, optional
        Path to save the figure.

    """
    logger.info("Plotting transaction trends.")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.countplot(
        data=transactions_df,
        x="payment_method_id",
        hue="is_auto_renew",
        palette=["#E76F51", "#264653"],  # Modern Sunset-to-Ocean palette
        ax=ax[0],
    )
    ax[0].set_title("Payment Methods vs. Auto-Renew", fontweight="bold", pad=15)
    ax[0].set_xlabel("Payment Method ID")
    ax[0].set_ylabel("Count of Transactions")
    ax[0].legend(
        title="Auto-Renew (1=Yes)", loc="upper right", frameon=True, shadow=True
    )
    sns.despine(ax=ax[0])

    # Cancellations
    cancel_rates = (
        transactions_df.groupby("payment_plan_days")["is_cancel"].mean().reset_index()
    )
    cancel_rates = cancel_rates[
        cancel_rates["payment_plan_days"] > 0
    ]  # remove free/trial oddities

    sns.barplot(
        data=cancel_rates,
        x="payment_plan_days",
        y="is_cancel",
        color="#2A9D8F",  # Clean Teal... consistent with "Smart Strategy" in notebook...
        ax=ax[1],
    )
    ax[1].set_title("Cancellation Rate by Plan Length", fontweight="bold", pad=15)
    ax[1].set_xlabel("Subscription Plan Duration (Days)")
    ax[1].set_ylabel("Avg Cancellation Rate (%)")
    sns.despine(ax=ax[1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_usage_intensity(user_logs_df, save_path=None):
    """
    Plot the distribution of daily listening time.

    Parameters
    ----------
    user_logs_df : pd.DataFrame
        User usage telemetry dataframe.
    save_path : str, optional
        Path to save the figure.

    """
    logger.info("Plotting usage intensity distributions.")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert seconds to hours for readability
    hours = user_logs_df["total_secs"] / 3600
    # Clip at 99th percentile to remove extreme anomalies for the plot
    p99 = hours.quantile(0.99)
    hours_clipped = hours[hours < p99]

    # Sample for KDE performance on large datasets
    plot_sample = hours_clipped.sample(
        n=min(50_000, len(hours_clipped)), random_state=42
    )
    sns.histplot(plot_sample, bins=40, kde=True, color="#264653", ax=ax, alpha=0.8)

    ax.set_title(
        "Analysis of User Engagement Patterns", fontsize=18, fontweight="bold", pad=20
    )
    ax.set_xlabel("Average Daily Hours Streamed")
    ax.set_ylabel("User-Day Observations")
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    """Run full EDA pipeline and generate reports."""
    os.makedirs(FIG_DIR, exist_ok=True)

    logger.info("Starting Exploratory Data Analysis...")
    members, transactions, user_logs = load_all_data()

    plot_registration_cohorts(
        members, os.path.join(FIG_DIR, "01_registration_cohorts.png")
    )
    plot_transaction_trends(
        transactions, os.path.join(FIG_DIR, "02_transaction_trends.png")
    )
    plot_usage_intensity(user_logs, os.path.join(FIG_DIR, "03_usage_intensity.png"))

    logger.info(f"EDA Complete. Visualizations saved to '{FIG_DIR}'")


if __name__ == "__main__":
    main()
