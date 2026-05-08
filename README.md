# Safaricom Stock Price Prediction Using Machine Learning and Deep Learning

##  Project Overview

This project focuses on predicting future stock prices of **Safaricom PLC** using historical stock market data and multiple machine learning techniques. The study combines both traditional supervised machine learning models and deep learning methods to analyze stock market trends and generate future price forecasts.

The project compares the performance of:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LSTM (Long Short-Term Memory)

A Streamlit dashboard was also developed to visualize historical stock prices and 30-day future forecasts interactively using Linear Regression and LSTM models.

---

# Problem Statement

Predicting stock prices is a difficult task due to the volatility and dynamic nature of financial markets. Investors and analysts require reliable forecasting techniques to help reduce uncertainty and improve decision-making.

Traditional statistical and machine learning approaches may struggle to capture complex sequential patterns found in stock market data. This project investigates whether advanced deep learning techniques such as LSTM can improve forecasting accuracy compared to traditional supervised learning models.

---

#  Objectives

The main objectives of this project are to:

- Analyze historical Safaricom stock data
- Preprocess and clean financial data
- Build supervised machine learning models
- Build deep learning forecasting models
- Predict future stock prices for the next 30 days using Linear Regression and LSTM
- Compare model performance using evaluation metrics
- Develop an interactive Streamlit dashboard

---

# Dataset

The dataset contains historical stock market information for Safaricom PLC from the Nairobi Securities Exchange (NSE).

### Features Used

- Date
- Open Price
- High Price
- Low Price
- Close Price
- Volume
- Change Percentage

---

#  Technologies Used

## Programming Language
- Python

## Libraries and Frameworks

- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- tensorflow
- streamlit
- joblib

---

# Machine Learning Models

## 1. Linear Regression
Used as a baseline supervised learning model for predicting stock prices.

## 2. Random Forest Regressor
An ensemble learning algorithm that combines multiple decision trees for improved predictions.

## 3. XGBoost Regressor
A gradient boosting algorithm designed for high-performance predictive modeling.

## 4. LSTM (Long Short-Term Memory)
A deep learning recurrent neural network specialized for sequential and time-series forecasting.

---

#  Data Preprocessing

The following preprocessing steps were performed:

- Handling missing values
- Converting percentage values
- Cleaning volume data
- Feature scaling using MinMaxScaler
- Sequence generation for LSTM
- Feature engineering using moving averages and returns

---

#  Model Evaluation Metrics

The models were evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

Lower values indicate better model performance.

---

# Visualization

The project includes visualizations for:

- Historical stock prices
- Actual vs predicted prices
- 30-day future forecasts
- Model comparison charts
- MAE and RMSE evaluation plots

---

#  Streamlit Dashboard

An interactive Streamlit dashboard was developed to:

- Display historical stock prices
- Show 30-day forecasts
- Compare predictions from Linear Regression and LSTM
- Visualize forecast trends interactively

---

# Project Structure

```bash
safaricom-stock-prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── NSE_SCOM_Safaricom.csv
├── model.pkl
├── lstm_model.h5
├── scaler.pkl
└── notebooks/
```

#  Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

---

#  Deployment

The application can be deployed using:

- Streamlit Community Cloud
- Hugging Face Spaces
- Render
- Railway

Recommended deployment platform:

 Streamlit Community Cloud

---

# Expected Results

The project demonstrates that:

- Traditional supervised models provide baseline forecasting
- Ensemble methods improve prediction stability
- LSTM captures sequential patterns more effectively
- Deep learning models generally outperform traditional regression models in time-series forecasting

The 30-day stock price forecasting was implemented using:

- Linear Regression
- LSTM

---

# Future Improvements

Possible future enhancements include:

- Adding technical indicators (RSI, MACD)
- Real-time stock price updates
- Transformer-based forecasting models
- Sentiment analysis from financial news
- Advanced ensemble forecasting

---

#  Author

**Derrick Gachenga**

---

#  License

This project is for educational and research purposes.
