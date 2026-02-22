# Subscription Churn & Value Segmentation

A machine learning pipeline that predicts subscription churn and segments users by lifetime value to optimize retention campaign ROI. 

Instead of offering blanket discounts to save cancelling users, this project demonstrates how to use ML to target high-value, high-risk accounts—saving marketing budget and increasing net revenue.

## 🚀 Key Features

* **Predictive Modeling (XGBoost):** Forecasts 30-day churn risk based on historical telemetry and billing data. Optimized for Precision-Recall AUC to handle class imbalance.
* **Customer Segmentation (K-Means):** Clusters users based on RFM (Recency, Frequency, Monetary) and engagement trends to identify "High-Value Whales" vs. "Casual Users".
* **Out-of-Core Data Engineering:** Includes scripts to safely process 30GB+ datasets on local hardware using memory-efficient chunking.
* **Business Impact ROI:** Translates model probabilities into direct financial metrics, comparing ML-driven targeted campaigns against baseline blanket discounts.
* **Interpretability:** Uses SHAP values to explain global and local churn drivers to stakeholders.

## 🛠️ Tech Stack
**Core:** Python, Pandas, NumPy  
**Modeling:** XGBoost, Scikit-Learn  
**Interpretability & Viz:** SHAP, Matplotlib, Seaborn  
**Engineering & Quality:** Pytest, Ruff (PEP-8), out-of-core chunking  

## 📂 Project Structure
```text
├── data/raw/                  # Ignored in git; data goes here
├── notebooks/
│   └── 02_business_impact_scenarios.ipynb  # Final presentation & ROI analysis
├── src/
│   ├── download_real_data.py  # Kaggle API downloader
│   ├── sample_kaggle_data.py  # Out-of-core 30GB chunking script
│   ├── data_loader.py         # Schema validation & typing
│   ├── eda.py                 # Visual cohort analysis
│   ├── features.py            # Temporal feature engineering (No target leakage)
│   ├── models.py              # XGBoost & Logistic Regression pipelines
│   └── segmentation.py        # K-Means RFM clustering
└── tests/                     # Unit tests for data leakage prevention
```

## ⚙️ How to Run

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the Data
You can either generate synthetic data (fast) or download the real [WSDM KKBox Kaggle Dataset](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data) (~30GB).

**Option A: Generate Mock Data (Fastest)**
```bash
python src/generate_mock_data.py
```

**Option B: Download Real Kaggle Data**
*(Requires `kaggle.json` API key in `~/.kaggle/` and accepting competition rules)*
```bash
python src/download_real_data.py
# IMMEDIATELY run this to downsample the 30GB file so it fits in RAM:
python src/sample_kaggle_data.py
```

### 3. Run the Pipeline
Once data is generated or downloaded, execute the pipeline:
```bash
python src/eda.py           # Generates distribution & cohort plots in /figures
python src/models.py        # Trains and evaluates XGBoost model
python src/segmentation.py  # Clusters users into business personas
```

### 4. View Results
Open the Jupyter notebook to see the final ROI calculations and SHAP dependency plots:
```bash
jupyter notebook notebooks/02_business_impact_scenarios.ipynb
```

## ✅ Testing & Linting
Run the test suite (validates schema and prevents temporal target leakage):
```bash
pytest tests/
```
Check code formatting:
```bash
ruff check .
```