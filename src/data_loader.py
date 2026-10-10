"""
data_loader.py
--------------
Reads the raw CSV and gives a quick summary.

To use a REAL public dataset instead of the simulation (for example the
Kaggle "Hourly Energy Consumption - PJM" dataset), rename its columns to
`timestamp`, `energy_kwh` (and optionally `temperature_c`, `is_holiday`)
and point config.RAW_DATA_FILE at it.
"""
import pandas as pd

from src import config


def load_raw_data(path=config.RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw meter export and parse timestamps."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python main.py --step generate` first."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_weather_forecast(path=config.WEATHER_FORECAST_FILE) -> pd.DataFrame:
    """Load next-week weather forecast (needed to forecast energy)."""
    return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")


def summarize(df: pd.DataFrame) -> dict:
    """Small data-profile dict - printed in the logs as proof of understanding."""
    return {
        "rows": len(df),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "missing_energy_values": int(df[config.TARGET].isna().sum()),
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "mean_kwh": round(float(df[config.TARGET].mean()), 2),
        "max_kwh": round(float(df[config.TARGET].max()), 2),
        "min_kwh": round(float(df[config.TARGET].min()), 2),
    }
