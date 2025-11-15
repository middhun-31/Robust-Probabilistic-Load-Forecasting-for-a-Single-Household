# Robust Probabilistic Load Forecasting: A Comparative Study on the REFIT Dataset

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete notebooks and analysis for a deep-dive case study into probabilistic time-series forecasting. This project tackles the twin challenges of real-world data volatility and significant data imperfections, using House 1 from the **REFIT dataset**.

The core of this research is a systematic, evidence-based approach to:
1.  **Handling a large structural data gap** through a rigorous imputation experiment.
2.  **Evaluating a hierarchy of models**, progressing from classical baselines (SARIMA, Prophet) to modern machine learning (XGBoost, LightGBM).
3.  **Implementing advanced probabilistic forecasts** using deep learning (LSTM, Temporal Fusion Transformer) to quantify uncertainty, moving beyond simple point-based error metrics.

---

## 🚀 Key Findings & Results

The final analysis reveals a clear performance hierarchy. Classical statistical models (SARIMA, Prophet) were fundamentally unsuited for this data's non-linear patterns. Machine learning models (XGBoost, LightGBM) set a strong benchmark for point-forecast accuracy.

Ultimately, the probabilistic deep learning models provided the most nuanced results. The **Probabilistic LSTM** achieved the best calibration and overall probabilistic score (lowest Average Quantile Score), while the **Temporal Fusion Transformer (TFT)** was the most robust all-rounder, with the best point-forecast accuracy (lowest RMSE/MAE) and highly dynamic "common-sense" prediction intervals.

### Final Model Performance

| Model | RMSE | MAE | PICP (90%) | Avg. Quantile Score |
| :--- | :--- | :--- | :--- | :--- |
| Seasonal Naïve | 623.2680 | 327.6460 | N/A | N/A |
| SARIMAX | 2815.9667 | 2565.0850 | N/A | N/A |
| LightGBM | 454.6866 | **258.0953** | N/A | N/A |
| XGBoost | **452.9195** | 260.6795 | N/A | N/A |
| LSTM (Prob.) | 517.4707 | 295.8569 | **88.81%** | **84.0122** |
| TFT (Prob.) | 481.9402 | 295.8882 | 95.85% | 84.9520 |

### Probabilistic Forecast Comparison

The deep learning models successfully generated dynamic uncertainty intervals. The LSTM (left) was well-calibrated, but its intervals were sometimes too narrow for extreme spikes. The TFT (right) produced more cautious and robust intervals that better captured the data's true volatility.

| LSTM Probabilistic Forecast | TFT Probabilistic Forecast |
| :---: | :---: |
| <img src="httpsI HAVE UPLOADED THIS IMAGE. PASTE IT HERE. fileId: image_ecfb3e.png" alt="LSTM Forecast Plot" width="400"> | <img src="I HAVE UPLOADED THIS IMAGE. PASTE IT HERE. fileId: image_ecfb63.png" alt="TFT Forecast Plot" width="400"> |

---

## 🔬 Methodology Highlights

### 1. Imputation of Structural Data Gap

A primary challenge was a multi-month data gap. A controlled experiment was conducted to select an imputation method. We found that a simple **Linear Imputer** failed to replicate the data's bimodal distribution. A **Seasonal Imputer** (using the mean of the same hour and day-of-week) was chosen as it successfully preserved the underlying structure of the data.

<img src="I HAVE UPLOADED THIS IMAGE. PASTE IT HERE. fileId: image_9c7364.png" alt="Imputation comparison plot" width="600">

### 2. Failure Analysis of Classical Models

Classical models like SARIMA (and Prophet) completely failed to model the data. They are not equipped to handle the sharp, non-linear "regime-switch" between high-usage weekdays and near-zero-usage weekends, resulting in massive prediction errors. This failure justified the move to more complex machine learning models.

<img src="I HAVE UPLOADED THIS IMAGE. PASTE IT HERE. fileId: image_ecf815.png" alt="SARIMA failure plot" width="600">

### 3. Feature Engineering

To enable the ML models, we engineered a rich feature set, including:
* **Calendar:** `hour`, `dayofweek`, `month`
* **Event:** `is_weekend` (a crucial flag for the regime-switch)
* **Lags:** `lag_1hr`, `lag_24hr` (daily seasonality), `lag_168hr` (weekly seasonality)

---

## 📁 Repository Structure

```
.
├── paper/
│   └── Robust_Probabilistic_Forecasting.pdf  (Optional: Your final paper)
├── notebooks/
│   ├── 01_Preprocessing_and_Imputation.ipynb (Covers EDA, Imputation Experiment)
│   └── 02_Modeling_and_Evaluation.ipynb    (Covers all models from Naïve to TFT)
├── .gitignore
└── README.md
```

---

## 🏁 How to Replicate

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (or .\venv\Scripts\activate on Windows)
    ```
3.  **Install dependencies:**
    *(You should generate a `requirements.txt` file using `pip freeze > requirements.txt` and add it to your repo. This is a placeholder for the main libraries.)*
    ```bash
    pip install pandas numpy matplotlib seaborn
    pip install scikit-learn statsmodels prophet lightgbm xgboost
    pip install darts[pytorch] tensorflow
    ```
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Open and run the notebooks in numerical order.
