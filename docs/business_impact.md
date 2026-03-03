# Business Impact & ROI

Predictive modeling is only as valuable as the business decisions it enables.

## The Problem with Blanket Campaigns

When marketing teams notice an uptick in churn, the typical response is to send out a 20% off coupon to all inactive members. However, many of those users would have renewed at full price anyway.

We lose margin on the "Loyalists" to save the "Flight Risks."

## Targeted ML-Driven Campaigns

This project solves this by scoring every customer with a **Churn Probability** between `0.0` and `1.0`.

### Simulation Results

Below is the comparison of a simulated retention strategy targeting the entire user base versus targeting only users identified by our model as high-risk within high-lifetime-value clusters.

![ROI Comparison](../figures/roi_comparison.png)

### The Strategy Breakdown

1. **Segment Users**: We use K-Means on historic billing and engagement to find our top tier users (Whales).
2. **Predict Churn Risk**: We apply our XGBoost classifier to get a 30-day flight risk probability.
3. **Targeted Intervention**: Only users who are in the Top 20% of probability *and* belong to high profitability clusters receive the expensive marketing intervention.

> **Impact**: The ML-driven strategy creates a significant gap in wasted spend, fundamentally raising the ceiling on marketing ROI.
