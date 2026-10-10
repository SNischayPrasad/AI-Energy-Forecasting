"""
features.py
-----------
Feature engineering: turn a timestamp + weather into numbers a model can learn from.

IMPORTANT RULE (no data leakage): every feature must be known at the moment
we make the forecast. That is why the shortest lag is 24 hours - when we
forecast tomorrow, we already know today's consumption, but not tomorrow's.
"""
import numpy as np
import pandas as pd

from src import config

FEATURE_COLUMNS = [
    # calendar
    "hour", "day_of_week", "month", "day_of_year", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    # weather
    "temperature_c", "cooling_degree", "heating_degree",
    # history (lags and rolling statistics)
    "lag_24h", "lag_48h", "lag_168h", "rolling_mean_24h", "rolling_max_24h", "rolling_mean_168h",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek          # 0 = Monday
    df["month"] = idx.month
    df["day_of_year"] = idx.dayofyear
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    # Cyclical encoding: tells the model 23:00 is next to 00:00
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    # Degree-days: the standard energy-industry way to describe AC / heating need
    df["cooling_degree"] = (df["temperature_c"] - 24).clip(lower=0)
    df["heating_degree"] = (18 - df["temperature_c"]).clip(lower=0)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    y = df[config.TARGET]
    df["lag_24h"] = y.shift(24)        # same hour yesterday
    df["lag_48h"] = y.shift(48)        # same hour 2 days ago
    df["lag_168h"] = y.shift(168)      # same hour last week
    shifted = y.shift(24)              # only use data that is >= 24 h old
    df["rolling_mean_24h"] = shifted.rolling(24).mean()
    df["rolling_max_24h"] = shifted.rolling(24).max()
    df["rolling_mean_168h"] = shifted.rolling(168).mean()
    return df


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Return a copy of df with all model features added."""
    out = df.copy()
    out = add_calendar_features(out)
    out = add_weather_features(out)
    out = add_lag_features(out)
    if dropna:
        out = out.dropna(subset=FEATURE_COLUMNS + [config.TARGET])
    return out
