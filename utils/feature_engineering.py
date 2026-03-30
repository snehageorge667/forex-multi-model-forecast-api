import numpy as np
import pandas as pd


def compute_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()

    rs = gain / (loss + 1e-9)  # avoid division by zero

    rsi = 100 - (100 / (1 + rs))

    return rsi


def create_features(df):

    # ✅ Prevent modifying original dataframe
    df = df.copy()

    # Log returns
    df['log_close'] = np.log(df['close'])
    df['target'] = df['log_close'].shift(-1) - df['log_close']

    # Price range percentage
    df['range_pct'] = (df['high'] - df['low']) / df['close']

    # Rolling features
    df['volatility_return_5'] = df['target'].rolling(5).std()
    df['ma_return_5'] = df['target'].rolling(5).mean()
    df['ma_return_10'] = df['target'].rolling(10).mean()

    # RSI
    df['rsi_14'] = compute_rsi(df['close'])

    # Drop missing values
    df = df.dropna()

    return df