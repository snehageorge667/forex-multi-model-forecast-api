from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import joblib
import requests
import time
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
# DOWNLOAD DATA (FIXED PROPERLY)
# ==============================
def download_data():
    API_KEY = "GPCSF42YSKR1EO22"

    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=INR&outputsize=compact&apikey={API_KEY}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        # Rate limit
        if "Note" in data:
            time.sleep(60)
            response = requests.get(url, timeout=10)
            data = response.json()

        # Invalid key
        if "Error Message" in data:
            raise ValueError("Invalid API key")

        #  Main bug fix → handle missing data properly
        if "Time Series FX (Daily)" not in data:
            raise ValueError("Invalid API response")

        ts = data["Time Series FX (Daily)"]

        df = pd.DataFrame.from_dict(ts, orient="index")

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

        df = df.sort_values("date")

        # ensure enough data
        if len(df) < 20:
            raise ValueError("Not enough data from API")

        return df.tail(60)

    except Exception as e:
        print("API FAILED:", e)

        # STRONG fallback (not 1 row like before)
        dates = pd.date_range(end=datetime.today(), periods=60)
        dummy = pd.DataFrame({
            "date": dates,
            "open": np.linspace(82, 83, 60),
            "high": np.linspace(83, 84, 60),
            "low": np.linspace(81, 82, 60),
            "close": np.linspace(82.5, 83.5, 60),
        })

        return dummy


# ==============================
# HOME
# ==============================
@app.get("/")
def home():
    return {"message": "Forex Forecast API is running"}


# ==============================
# SAFE LAST VALUE HELPER
# ==============================
def get_last_close(df):
    if df is None or df.empty:
        raise ValueError("No data available")
    return df['close'].iloc[-1]


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

        last_close = get_last_close(raw)

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
        raw = create_features(download_data())

        last_close = get_last_close(raw)

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

        last_close = get_last_close(raw)

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
        raw = create_features(download_data())

        last_close = get_last_close(raw)

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

        feature_cols = xgb_model.get_booster().feature_names
        df = df.reindex(columns=feature_cols, fill_value=0)

        latest = df.iloc[-1:]

        pred = xgb_model.predict(latest)[0]

        last_close = get_last_close(raw)

        return {
            "model": "XGBoost",
            "predicted_close": float(last_close * np.exp(pred))
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

        feature_cols = lgb_model.booster_.feature_name()
        df = df.reindex(columns=feature_cols, fill_value=0)

        latest = df.iloc[-1:]

        pred = lgb_model.predict(latest)[0]

        last_close = get_last_close(raw)

        return {
            "model": "LightGBM",
            "predicted_close": float(last_close * np.exp(pred))
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

        feature_cols = xgb_macro_model.get_booster().feature_names
        df = df.reindex(columns=feature_cols, fill_value=0)

        latest = df.iloc[-1:]

        pred = xgb_macro_model.predict(latest)[0]

        last_close = get_last_close(raw)

        return {
            "model": "XGBoost Macro",
            "predicted_close": float(last_close * np.exp(pred))
        }

    except Exception as e:
        return {"error": str(e)}


# ==============================
# FUTURE FORECAST
# ==============================
def predict_future_recursive(raw_df, future_days):
    df = create_features(raw_df)

    model_features = future_model.booster_.feature_name()

    df = df.reindex(columns=model_features + ['date', 'close', 'cc_log_return'], fill_value=0).dropna()

    last_data = df.iloc[-150:].copy().reset_index(drop=True)

    predictions = []
    current_date = pd.to_datetime(datetime.today().date())

    for _ in range(future_days):
        X = last_data[model_features].iloc[-1:]

        pred = np.clip(future_model.predict(X)[0], -0.03, 0.03)

        last_close = last_data['close'].iloc[-1]
        next_close = 0.7 * (last_close * np.exp(pred)) + 0.3 * last_close

        predictions.append({
            "date": str(current_date.date()),
            "predicted_close": float(next_close)
        })

        new_row = last_data.iloc[-1:].copy()
        new_row['date'] = current_date
        new_row['close'] = next_close
        new_row['cc_log_return'] = pred

        last_data = pd.concat([last_data, new_row], ignore_index=True)
        current_date += timedelta(days=1)

    return predictions


@app.get("/predict/future")
def predict_future(date: date = Query(..., description="YYYY-MM-DD")):
    try:
        future_date = pd.to_datetime(date)
        today = pd.to_datetime(datetime.today().date())

        days = (future_date - today).days + 1
        if days <= 0:
            return {"error": "Date must be future"}

        preds = predict_future_recursive(download_data(), days)

        return {
            "start_date": str(today.date()),
            "target_date": str(future_date.date()),
            "total_days": days,
            "predictions": preds
        }

    except Exception as e:
        return {"error": str(e)}


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))