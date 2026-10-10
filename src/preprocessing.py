"""
preprocessing.py
----------------
Data cleaning. Garbage in = garbage out, so this step matters a lot.

Steps:
1. Remove duplicate timestamps
2. Make the timeline continuous (one row per hour, gaps become NaN)
3. Detect outliers (meter glitches) and mark them as missing
4. Fill missing values with time-based interpolation
"""
import pandas as pd

from src import config


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset="timestamp", keep="first")


def make_hourly_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Index by timestamp and insert any missing hours."""
    df = df.set_index("timestamp").sort_index()
    return df.asfreq("h")


def flag_outliers(series: pd.Series, window: int = 24 * 7, threshold: float = 5.0) -> pd.Series:
    """
    Robust outlier detection.
    Compare each reading with the same hour across the surrounding week
    (rolling median) and flag it if it is far away, measured in MADs
    (median absolute deviation - like a standard deviation that is not
    fooled by the outliers themselves).
    Zero readings are also flagged: a whole city zone never uses 0 kWh.
    """
    median = series.rolling(window, center=True, min_periods=24).median()
    deviation = (series - median).abs()
    mad = deviation.rolling(window, center=True, min_periods=24).median()
    robust_z = deviation / (1.4826 * mad)
    return (robust_z > threshold) | (series <= 0)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run every cleaning step and return (clean_df, report)."""
    report = {"rows_in": len(df)}

    df = remove_duplicates(df)
    report["duplicates_removed"] = report["rows_in"] - len(df)

    df = make_hourly_continuous(df)
    report["missing_before_fill"] = int(df[config.TARGET].isna().sum())

    outliers = flag_outliers(df[config.TARGET])
    report["outliers_detected"] = int(outliers.sum())
    df.loc[outliers, config.TARGET] = float("nan")

    # Time interpolation draws a straight line between neighbouring good values
    df[config.TARGET] = df[config.TARGET].interpolate(method="time").bfill().ffill()
    df["temperature_c"] = df["temperature_c"].interpolate(method="time").bfill().ffill()
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    report["missing_after_fill"] = int(df[config.TARGET].isna().sum())
    report["rows_out"] = len(df)
    return df, report
