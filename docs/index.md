# Subscription Churn Prediction

End-to-end ML pipeline for predicting 30-day subscription churn and optimizing retention spend.

## The problem

Subscription companies lose money when they offer blanket discounts to users who would have renewed anyway. The question is: can we predict who's actually at risk, and target our retention budget accordingly?

## What this does

1. Predicts churn probability for each user (30-day window)
2. Segments users by lifetime value using K-Means
3. Simulates ROI of targeted vs. blanket retention campaigns

The goal is to spend retention dollars only where they matter.

## Documentation

- [Business Impact & ROI](business_impact.md) - Why this matters financially
- [Data & Engineering](data_engineering.md) - How the pipeline works
- [Model Performance](modeling.md) - Metrics and interpretation

## Project structure

```
├── data/          # raw and processed data + saved model (gitignored)
├── docs/          # MkDocs site
├── figures/       # generated plots
├── reports/       # metrics.json model card (reproducible)
├── src/           # pipeline code
├── tests/         # pytest
├── Makefile
└── pyproject.toml
```
