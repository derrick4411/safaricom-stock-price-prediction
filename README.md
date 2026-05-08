# 📈 Safaricom (SCOM.KE) Stock Price Prediction

> A comparative time-series forecasting study using **LSTM**, **Prophet**, and **XGBoost** on Nairobi Securities Exchange (NSE) data.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Source](#data-source)
3. [Repository Structure](#repository-structure)
4. [Methodology](#methodology)
5. [Model Comparison Rationale](#model-comparison-rationale)
6. [Results Summary](#results-summary)
7. [Setup & Installation](#setup--installation)
8. [Usage](#usage)
9. [Next Steps](#next-steps)

---

## Project Overview

Safaricom PLC (ticker: **SCOM**) is the largest telco and fintech operator in East Africa and among the most actively traded equities on the Nairobi Securities Exchange. This project builds, tunes, and compares three distinct forecasting approaches to predict SCOM's 30-day forward closing price, with the goal of understanding which model best captures the stock's structural trends, seasonal patterns, and short-term volatility.

**Key Questions:**
- Can deep learning (LSTM) outperform classical additive decomposition (Prophet) on a frontier-market stock?
- How well does gradient-boosted regression (XGBoost) translate engineered lag features into price signals?
- Which model generalises best across NSE's lower-liquidity, higher-volatility environment?

---

## Data Source

| Field | Detail |
|-------|--------|
| **Exchange** | Nairobi Securities Exchange (NSE) |
| **Ticker** | SCOM.KE — Safaricom PLC |
| **Provider** | [myStocks.co.ke](https://mystocks.co.ke) / NSE Historical Data |
| **File** | `NSE_SCOM_Safaricom.csv` |
| **Columns** | `Date`, `Open`, `High`, `Low`, `Close`, `Vol.`, `Change %` |
| **Frequency** | Daily (trading days only) |
| **Coverage** | June 2012 – present |
| **Target Variable** | `Close` price (KES) |

> **Note:** Volume is encoded as `M` (millions) or `K` (thousands) in the raw CSV and is normalised to integers during preprocessing.

---

## Repository Structure

```
scom-stock-prediction/
│
├── saf_stock_predict.ipynb   # Main analysis notebook
├── NSE_SCOM_Safaricom.csv    # Raw NSE daily price data
├── README.md                 # This file
│
├── models/
│   ├── xgb_tuned.pkl         # Saved XGBoost model (GridSearchCV best)
│   ├── lstm_model.h5         # Saved LSTM weights
│   └── prophet_model.json    # Serialised Prophet model
│
└── outputs/
    ├── 30day_forecast.png    # Forecast visualisation
    └── model_comparison.csv  # RMSE / MAE / R² across models
```

---

## Methodology

### 1. Data Preprocessing
- Parse `Date` column to `datetime` and sort chronologically
- Clean `Vol.` strings (strip `M`/`K` suffixes, convert to float)
- Strip `%` from `Change %` and cast to float
- Impute any missing values using forward-fill (appropriate for time series — avoids lookahead bias)
- Scale all numeric features to `[0, 1]` with `MinMaxScaler` for LSTM and Prophet

### 2. Feature Engineering (for XGBoost)
| Feature | Description |
|---------|-------------|
| `lag_1` … `lag_30` | Previous 1–30 day closing prices |
| `rolling_7_mean` | 7-day rolling mean (short trend) |
| `rolling_30_mean` | 30-day rolling mean (medium trend) |
| `rolling_7_std` | 7-day rolling standard deviation (volatility proxy) |
| `day_of_week` | Day-of-week encoding (0=Mon … 4=Fri) |
| `month` | Month of year (seasonality) |
| `pct_change` | Daily % change in close price |

### 3. Train / Test Split
- **Chronological split** — last 20% of trading days held out as the test set
- **No random shuffling** — preserves temporal order to prevent data leakage

### 4. Evaluation Metrics
| Metric | Symbol | Why it matters |
|--------|--------|---------------|
| Root Mean Squared Error | RMSE | Penalises large price misses more heavily |
| Mean Absolute Error | MAE | Interpretable in KES units |
| Coefficient of Determination | R² | Proportion of variance explained |

---

## Model Comparison Rationale

### 🧠 LSTM (Long Short-Term Memory Neural Network)
LSTM is a recurrent neural network architecture specifically designed to learn **long-range temporal dependencies** in sequential data. For SCOM, this means the model can learn that a sustained 15-day downtrend historically precedes a consolidation phase, even if individual daily moves are noisy.

**Why LSTM for SCOM?**
- NSE stocks exhibit mean-reverting behaviour over multi-week windows — a pattern LSTM can encode in its cell state
- Handles non-linear, non-stationary price dynamics without manual feature engineering
- `look_back` window (tuned: 30–60 days) controls how far back the model "remembers"
- `Dropout` layers (tuned: 0.2–0.4) combat overfitting on SCOM's relatively short price history

**Trade-offs:** Computationally expensive, requires more data to generalise, and is a black-box — difficult to explain to non-technical stakeholders.

---

### 🔮 Prophet (Facebook/Meta Additive Decomposition Model)
Prophet decomposes the time series into **trend + seasonality + holiday effects**, fitting each component independently. It is exceptionally robust to missing data and structural breaks — both common on the NSE around earnings announcements, dividend dates, and Central Bank of Kenya rate decisions.

**Why Prophet for SCOM?**
- SCOM has clear **annual seasonality** tied to M-PESA revenue reporting cycles (H1 ends September, H2 ends March)
- Prophet's `changepoint_prior_scale` controls sensitivity to structural trend breaks — critical for a stock that has traded between KES 5 and KES 42 in its listed history
- Built-in uncertainty intervals give probabilistic forecasts, useful for risk management
- Requires only `ds` (date) and `y` (price) columns — simple API

**Trade-offs:** Assumes additive decomposition (may underfit during explosive momentum moves); does not incorporate exogenous features like volume or macro indicators without extra regressors.

---

### ⚡ XGBoost (Extreme Gradient Boosting Regressor)
XGBoost treats price prediction as a **supervised regression problem** using hand-crafted lag features and rolling statistics. It builds an ensemble of decision trees, each correcting the residuals of the previous, producing a powerful non-linear regression model.

**Why XGBoost for SCOM?**
- Excellent at capturing **cross-feature interactions** — e.g. how trading volume interacts with price momentum
- Fast to train and highly interpretable via SHAP feature importance values
- `GridSearchCV` over `max_depth`, `learning_rate`, and `n_estimators` prevents overfitting to SCOM's regime changes
- Naturally handles the engineered technical-indicator features

**Trade-offs:** Does not natively model temporal ordering — relies entirely on lag features to encode sequence; cannot extrapolate beyond the training distribution.

---

## Results Summary

*(Populated after running the full notebook)*

| Model | RMSE (KES) | MAE (KES) | R² |
|-------|-----------|-----------|-----|
| Linear Regression (baseline) | — | — | — |
| Random Forest | — | — | — |
| XGBoost (Tuned) | — | — | — |
| LSTM (Tuned) | — | — | — |
| Prophet | — | — | — |

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/scom-stock-prediction.git
cd scom-stock-prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost \
            tensorflow prophet joblib
```

---

## Usage

```bash
# Run the full notebook
jupyter notebook saf_stock_predict.ipynb

# Or execute as a script
jupyter nbconvert --to script saf_stock_predict.ipynb
python saf_stock_predict.py
```

---

## Next Steps
- [ ] Incorporate **exogenous regressors** into Prophet: KES/USD exchange rate, CBK base rate
- [ ] Add **SHAP values** for XGBoost feature importance explainability
- [ ] Experiment with **Transformer-based** time-series models (e.g. Temporal Fusion Transformer)
- [ ] Build a lightweight **Streamlit dashboard** for real-time 30-day SCOM forecasts
- [ ] Extend to other NSE equities: Equity Bank (EQTY), KCB Group, East African Breweries

---

*Data sourced from the Nairobi Securities Exchange via myStocks.co.ke. This project is for educational and research purposes only and does not constitute financial advice.*
