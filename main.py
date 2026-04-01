from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import requests
from datetime import datetime, timedelta, date

from utils.feature_engineering import create_features

app = FastAPI(title="Forex Prediction API")

# ==============================
# LOAD MODELS
# ==============================
arima_model = joblib.load("models/arima_model.pkl")
arimax_model = joblib.load("models/arimax_model.pkl")
sarima_model = joblib.load("models/sarima_model.pkl")
sarimax_model = joblib.load("models/sarimax_model.pkl")

xgb_model = joblib.load("models/xgboost_model.pkl")
lgb_model = joblib.load("models/lightgbm_model.pkl")
xgb_macro_model = joblib.load("models/xgboost_macro_model.pkl")

future_model = joblib.load("models/future_forecasting.pkl")


# ==============================
# DOWNLOAD DATA 
# ==============================
import time

def download_data():
    try:
        API_KEY = "GPCSF42YSKR1EO22"

        url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=INR&outputsize=compact&apikey={API_KEY}"

        response = requests.get(url)
        data = response.json()

        # HANDLE RATE LIMIT
        if "Note" in data:
            time.sleep(60)
            response = requests.get(url)
            data = response.json()

        if "Time Series FX (Daily)" not in data:
            raise ValueError("Invalid API response")

        time_series = data["Time Series FX (Daily)"]

        df = pd.DataFrame.from_dict(time_series, orient="index")

        df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close"
        }, inplace=True)

        df = df.astype(float)
        df.index = pd.to_datetime(df.index)

        df.reset_index(inplace=True)
        df.rename(columns={"index": "date"}, inplace=True)

        return df.sort_values("date").tail(60)

    except Exception as e:
        raise ValueError(f"Alpha Vantage failed: {str(e)}")
    if "Note" in data:
        raise ValueError("API rate limit exceeded")

# ==============================
# HOME
# ==============================
@app.get("/")
def home():
    return {"message": "Forex Forecast API is running"}


# ==============================
# ARIMA
# ==============================
@app.get("/predict/arima")
def predict_arima():

    try:
        raw = download_data()

        raw['log_close'] = np.log(raw['close'])
        raw['target'] = raw['log_close'].shift(-1) - raw['log_close']

        raw.dropna(inplace=True)

        if raw.empty:
            return {"error": "Not enough data"}

        last_close = raw['close'].iloc[-1]

        forecast = arima_model.forecast(steps=1)

        predicted_price = last_close * np.exp(forecast.iloc[0])

        return {"model": "ARIMA", "predicted_close": float(predicted_price)}

    except Exception as e:
        return {"error": str(e)}


# ==============================
# ARIMAX
# ==============================
@app.get("/predict/arimax")
def predict_arimax():

    try:
        raw = download_data()
        raw = create_features(raw)

        if raw.empty:
            return {"error": "Not enough data"}

        last_close = raw['close'].iloc[-1]

        feature_cols = [
            'range_pct',
            'volatility_return_5',
            'ma_return_5',
            'ma_return_10',
            'rsi_14'
        ]

        exog = raw[feature_cols].iloc[-1:]

        forecast = arimax_model.forecast(steps=1, exog=exog)

        predicted_price = last_close * np.exp(forecast.iloc[0])

        return {"model": "ARIMAX", "predicted_close": float(predicted_price)}

    except Exception as e:
        return {"error": str(e)}


# ==============================
# SARIMA
# ==============================
@app.get("/predict/sarima")
def predict_sarima():

    try:
        raw = download_data()

        raw['log_close'] = np.log(raw['close'])
        raw['target'] = raw['log_close'].shift(-1) - raw['log_close']

        raw.dropna(inplace=True)

        if raw.empty:
            return {"error": "Not enough data"}

        last_close = raw['close'].iloc[-1]

        forecast = sarima_model.forecast(steps=1)

        predicted_price = last_close * np.exp(forecast.iloc[0])

        return {"model": "SARIMA", "predicted_close": float(predicted_price)}

    except Exception as e:
        return {"error": str(e)}


# ==============================
# SARIMAX
# ==============================
@app.get("/predict/sarimax")
def predict_sarimax():

    try:
        raw = download_data()
        raw = create_features(raw)

        if raw.empty:
            return {"error": "Not enough data"}

        last_close = raw['close'].iloc[-1]

        feature_cols = [
            'range_pct',
            'volatility_return_5',
            'ma_return_5',
            'ma_return_10',
            'rsi_14'
        ]

        exog = raw[feature_cols].iloc[-1:]

        forecast = sarimax_model.forecast(steps=1, exog=exog)

        predicted_price = last_close * np.exp(forecast.iloc[0])

        return {"model": "SARIMAX", "predicted_close": float(predicted_price)}

    except Exception as e:
        return {"error": str(e)}


# ==============================
# XGBOOST
# ==============================
@app.get("/predict/xgboost")
def predict_xgboost():

    try:
        raw = download_data()
        df = create_features(raw)

        if df.empty:
            return {"error": "Not enough data"}

        feature_cols = xgb_model.get_booster().feature_names
        df = df.reindex(columns=feature_cols, fill_value=0)

        latest_data = df.iloc[-1:]

        if latest_data.empty:
            return {"error": "No features available"}

        pred = xgb_model.predict(latest_data)[0]

        last_close = raw['close'].iloc[-1]
        predicted_price = last_close * np.exp(pred)

        return {
            "model": "XGBoost",
            "predicted_close": float(predicted_price)
        }

    except Exception as e:
        return {"error": str(e)}


# ==============================
# LIGHTGBM
# ==============================
@app.get("/predict/lightgbm")
def predict_lightgbm():

    try:
        raw = download_data()
        df = create_features(raw)

        if df.empty:
            return {"error": "Not enough data"}

        feature_cols = lgb_model.booster_.feature_name()
        df = df.reindex(columns=feature_cols, fill_value=0)

        latest_data = df.iloc[-1:]

        if latest_data.empty:
            return {"error": "No features available"}

        pred = lgb_model.predict(latest_data)[0]

        last_close = raw['close'].iloc[-1]
        predicted_price = last_close * np.exp(pred)

        return {
            "model": "LightGBM",
            "predicted_close": float(predicted_price)
        }

    except Exception as e:
        return {"error": str(e)}


# ==============================
# XGBOOST MACRO
# ==============================
@app.get("/predict/xgboost_macro")
def predict_xgboost_macro():

    try:
        raw = download_data()
        df = create_features(raw)

        if df.empty:
            return {"error": "Not enough data"}

        feature_cols = xgb_macro_model.get_booster().feature_names
        df = df.reindex(columns=feature_cols, fill_value=0)

        latest_data = df.iloc[-1:]

        if latest_data.empty:
            return {"error": "No features available"}

        pred = xgb_macro_model.predict(latest_data)[0]

        last_close = raw['close'].iloc[-1]
        predicted_price = last_close * np.exp(pred)

        return {
            "model": "XGBoost Macro",
            "predicted_close": float(predicted_price)
        }

    except Exception as e:
        return {"error": str(e)}


# ==============================
# FUTURE FORECAST
# ==============================
def predict_future_recursive(raw_df, future_days):

    df = create_features(raw_df)

    model_features = future_model.booster_.feature_name()

    required_cols = model_features + ['date', 'close', 'cc_log_return']

    df = df.reindex(columns=required_cols, fill_value=0).dropna()

    last_data = df.iloc[-150:].copy().reset_index(drop=True)

    predictions = []

    current_date = pd.to_datetime(datetime.today().date())

    for _ in range(future_days):

        X = last_data[model_features].iloc[-1:]

        pred = future_model.predict(X)[0]
        pred = np.clip(pred, -0.03, 0.03)

        last_close = last_data['close'].iloc[-1]
        next_close = last_close * np.exp(pred)

        next_close = 0.7 * next_close + 0.3 * last_close

        predictions.append({
            "date": str(current_date.date()),
            "predicted_close": float(next_close)
        })

        new_row = last_data.iloc[-1:].copy()
        new_row['date'] = current_date
        new_row['close'] = next_close
        new_row['cc_log_return'] = pred

        for lag in [1, 2, 3, 5, 10]:
            col = f'cc_return_lag_{lag}'
            new_row[col] = last_data[col].iloc[-1] if col in last_data.columns else 0

        last_data = pd.concat([last_data, new_row], ignore_index=True)

        current_date = current_date + timedelta(days=1)

    return predictions


@app.get("/predict/future")
def predict_future(
    date: date = Query(
        ...,
        description="Enter target date in format YYYY-MM-DD",
        examples={"example1": {"value": "2026-04-03"}}
    )
):

    try:
        future_date = pd.to_datetime(date)
    except:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    try:
        df = download_data()

        today = pd.to_datetime(datetime.today().date())

        future_days = (future_date - today).days + 1

        if future_days <= 0:
            return {"error": "Date must be in the future"}

        predictions = predict_future_recursive(df, future_days)

        return {
            "start_date": str(today.date()),
            "target_date": str(future_date.date()),
            "total_days": future_days,
            "predictions": predictions
        }

    except Exception as e:
        return {"error": str(e)}


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))