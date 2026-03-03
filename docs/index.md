# Welcome to Ecommerce Retention Growth

This repository contains the full end-to-end Machine Learning pipeline for predicting 30-day subscription churn.

## 🌟 The Challenge

Subscription services live and die by their retention rates. When customers leave, marketing teams often scramble to offer discounts. But **blanket discounts waste money** when given to users who would have stayed anyway.

Can we predict who is going to leave, *before* they leave, and segment them by value?

## 🚀 Our Solution

By leveraging machine learning and customer segmentation, we target intervention budgets **strictly on users passing minimum ROI thresholds**.

* Check out the [Data Engineering](data_engineering.md) section to see how we handled 400M+ event logs.
* See our [Model Performance](modeling.md) for deeper insights into the XGBoost architecture.
* Dive into our [Business Impact](business_impact.md) for the actual financial ROI.

### Project Layout

```text
├── data/                  # Gitignored raw and processed CSVs
├── docs/                  # MkDocs GitHub Pages documentation 
├── figures/               # Automatically generated model evaluation charts
├── notebooks/             # Exploratory analysis and business scenario modeling
├── src/                   # Core Python application logic and pipelines
├── tests/                 # Execution tests via PyTest
├── mkdocs.yml             # Global MkDocs configuration
└── README.md              # Root documentation
```
