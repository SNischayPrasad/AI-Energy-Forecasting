"""
forecast.py
-----------
Uses the saved model to forecast the NEXT 7 days, hour by hour.

How it works (recursive day-ahead forecasting):
  The model needs "consumption 24 h ago" as an input. For day 1 of the
  forecast that value is real history. For day 2 it is day 1's *forecast*.
  So we predict one day at a time and feed each day's predictions back in.
"""
import json

import joblib
import pandas as pd

from src import config
from src.features import FEATURE_COLUMNS, build_features


def load_model():
    model = joblib.load(config.BEST_MODEL_FILE)
    metadata = json.loads(config.MODEL_METADATA_FILE.read_text())
    return model, metadata


def forecast_future(clean_df: pd.DataFrame, weather_df: pd.DataFrame,
                    horizon: int = config.FORECAST_HORIZON_HOURS) -> pd.DataFrame:
    model, meta = load_model()

    history = clean_df[[config.TARGET, "temperature_c", "is_holiday"]].copy()
    future = weather_df.iloc[:horizon].copy()
    future[config.TARGET] = float("nan")
    combined = pd.concat([history, future])

    for start in range(0, horizon, 24):
        block = future.index[start:start + 24]
        # Only the recent tail is needed to compute lags (saves time)
        window = combined.iloc[-(horizon + 24 * 14):]
        feats = build_features(window, dropna=False)
        combined.loc[block, config.TARGET] = model.predict(feats.loc[block, FEATURE_COLUMNS])

    out = pd.DataFrame(index=future.index)
    out["forecast_kwh"] = combined.loc[future.index, config.TARGET].round(2)
    # 90% prediction interval from historical test errors
    out["lower_kwh"] = (out["forecast_kwh"] + meta["residual_q05"]).round(2)
    out["upper_kwh"] = (out["forecast_kwh"] + meta["residual_q95"]).round(2)
    out["temperature_c"] = future["temperature_c"]
    out.to_csv(config.OUTPUTS_DIR / "future_forecast.csv", index_label="timestamp")
    return out
