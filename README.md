# Stock Price Prediction & Financial News Sentiment Dashboard

## Overview

This project is an AI-powered Stock Market Analytics Dashboard that predicts the next trading day's closing stock price using Machine Learning and analyzes recent financial news using FinBERT sentiment analysis.

The system combines technical indicators, historical stock data, machine learning predictions, and financial news sentiment to provide actionable insights for investors and learners.

---

## Features

### Stock Price Prediction

* Upload historical stock dataset (CSV)
* Automatic feature engineering
* XGBoost-based prediction model
* Predicts next-day closing price
* Calculates expected percentage change
* Generates Buy, Sell, or Hold signals

### Technical Analysis

* SMA (Simple Moving Average)
* EMA (Exponential Moving Average)
* RSI (Relative Strength Index)
* MACD (Moving Average Convergence Divergence)
* Lag Features (1–10 Days)
* Daily Return Analysis
* Volume Change Analysis

### Financial News Sentiment Analysis

* Fetches latest stock-related news
* Uses FinBERT (Financial BERT)
* Classifies news as:

  * Positive
  * Negative
  * Neutral
* Generates:

  * Bullish Sentiment
  * Bearish Sentiment
  * Neutral Sentiment

### Final Recommendation Engine

Combines:

* Machine Learning Prediction
* Financial News Sentiment

Generates:

* Strong Buy
* Buy
* Hold
* Sell
* Strong Sell

### Visualization

* Historical Price Chart
* Interactive Plotly Graphs
* Prediction Metrics Dashboard

---

## Machine Learning Model

### Algorithm

* XGBoost Regressor

### Training Dataset

* NIFTY 50 Stocks
* Historical Data: 2014 – 2026
* Downloaded using Yahoo Finance (yFinance)

### Features Used

* Open
* High
* Low
* Volume
* SMA_10
* SMA_20
* EMA_10
* EMA_20
* RSI
* MACD
* MACD Signal
* Daily Return
* Volume Change
* Stock ID
* Lag_1 to Lag_10

### Target Variable

* Next Day Closing Price

---

## Model Performance

| Metric   | Value  |
| -------- | ------ |
| R² Score | 0.949  |
| MAPE     | 2.22%  |
| MAE      | 140.35 |
| RMSE     | 621.31 |

The model achieved approximately 97.8% historical prediction accuracy with an average prediction error of 2.22%.

---

## Technology Stack

### Frontend

* Streamlit

### Machine Learning

* XGBoost
* Scikit-Learn

### Data Processing

* Pandas
* NumPy

### Technical Indicators

* TA Library

### News Analysis

* FinBERT
* Transformers
* Feedparser

### Visualization

* Plotly

### Data Source

* Yahoo Finance (yFinance)
* Google News RSS

---

## Project Workflow

1. Upload Stock Dataset
2. Select Stock Symbol
3. Generate Technical Indicators
4. Predict Next-Day Closing Price
5. Calculate Trading Signal
6. Fetch Latest Financial News
7. Analyze News with FinBERT
8. Generate Market Sentiment
9. Produce Final Recommendation
10. Display Charts and Insights

---

## Future Enhancements

* Multi-Day Forecasting (3, 5, 7 Days)
* Real-Time Market Data Integration
* Portfolio Tracking
* Risk Analysis Module
* Deep Learning Models (LSTM, Transformer)
* Automated Trading Signals
* Cloud Deployment

---

Built as a Machine Learning and Financial Analytics Project using Python, XGBoost, FinBERT, and Streamlit.
