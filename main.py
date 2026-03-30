from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
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
def download_data():

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')

    raw = yf.download("INR=X", start=start_date, end=end_date, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.reset_index(inplace=True)

    raw.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close'
    }, inplace=True)

    raw = raw.sort_values('date')

    return raw


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

    raw = download_data()

    raw['log_close'] = np.log(raw['close'])
    raw['target'] = raw['log_close'].shift(-1) - raw['log_close']

    raw.dropna(inplace=True)

    last_close = raw['close'].iloc[-1]

    forecast = arima_model.forecast(steps=1)

    predicted_price = last_close * np.exp(forecast.iloc[0])

    return {"model": "ARIMA", "predicted_close": float(predicted_price)}


# ==============================
# ARIMAX
# ==============================
@app.get("/predict/arimax")
def predict_arimax():

    raw = download_data()
    raw = create_features(raw)

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


# ==============================
# SARIMA
# ==============================
@app.get("/predict/sarima")
def predict_sarima():

    raw = download_data()

    raw['log_close'] = np.log(raw['close'])
    raw['target'] = raw['log_close'].shift(-1) - raw['log_close']

    raw.dropna(inplace=True)

    last_close = raw['close'].iloc[-1]

    forecast = sarima_model.forecast(steps=1)

    predicted_price = last_close * np.exp(forecast.iloc[0])

    return {"model": "SARIMA", "predicted_close": float(predicted_price)}


# ==============================
# SARIMAX
# ==============================
@app.get("/predict/sarimax")
def predict_sarimax():

    raw = download_data()
    raw = create_features(raw)

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


# ==============================
# XGBOOST
# ==============================
@app.get("/predict/xgboost")
def predict_xgboost():

    raw = download_data()
    df = create_features(raw)

    feature_cols = xgb_model.get_booster().feature_names
    df = df.reindex(columns=feature_cols, fill_value=0)

    latest_data = df.iloc[-1:]

    pred = xgb_model.predict(latest_data)[0]

    last_close = raw['close'].iloc[-1]
    predicted_price = last_close * np.exp(pred)

    return {
        "model": "XGBoost",
        "predicted_close": float(predicted_price)
    }


# ==============================
# LIGHTGBM
# ==============================
@app.get("/predict/lightgbm")
def predict_lightgbm():

    raw = download_data()
    df = create_features(raw)

    feature_cols = lgb_model.booster_.feature_name()
    df = df.reindex(columns=feature_cols, fill_value=0)

    latest_data = df.iloc[-1:]

    pred = lgb_model.predict(latest_data)[0]

    last_close = raw['close'].iloc[-1]
    predicted_price = last_close * np.exp(pred)

    return {
        "model": "LightGBM",
        "predicted_close": float(predicted_price)
    }


# ==============================
# XGBOOST MACRO
# ==============================
@app.get("/predict/xgboost_macro")
def predict_xgboost_macro():

    raw = download_data()
    df = create_features(raw)

    feature_cols = xgb_macro_model.get_booster().feature_names
    df = df.reindex(columns=feature_cols, fill_value=0)

    latest_data = df.iloc[-1:]

    pred = xgb_macro_model.predict(latest_data)[0]

    last_close = raw['close'].iloc[-1]
    predicted_price = last_close * np.exp(pred)

    return {
        "model": "XGBoost Macro",
        "predicted_close": float(predicted_price)
    }


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
        examples={
            "example1": {
                "summary": "Valid date format",
                "value": "2026-04-03"
            }
        }
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
    uvicorn.run("main:app", host="0.0.0.0", port=10000)