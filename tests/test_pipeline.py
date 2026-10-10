"""
Quick sanity tests - run with:  python -m pytest -q
They prove the pipeline works and guard against the classic time-series bug: data leakage.
"""
import numpy as np
import pandas as pd

from src import config
from src.data_generator import generate_dataset
from src.evaluate import regression_metrics
from src.features import FEATURE_COLUMNS, build_features
from src.preprocessing import clean_data


def _clean():
    raw = generate_dataset(save=False)
    return clean_data(raw)


def test_generator_has_two_years_of_hourly_data():
    raw = generate_dataset(save=False)
    assert raw["timestamp"].nunique() == 2 * 365 * 24 + 24   # 2024 is a leap year
    assert {"timestamp", config.TARGET, "temperature_c", "is_holiday"} <= set(raw.columns)


def test_cleaning_removes_all_problems():
    clean, report = _clean()
    assert clean[config.TARGET].isna().sum() == 0
    assert clean.index.is_unique
    assert report["outliers_detected"] > 0
    assert (clean[config.TARGET] > 0).all()


def test_lag_features_do_not_leak_future():
    clean, _ = _clean()
    feats = build_features(clean)
    t = feats.index[500]
    # lag_24h at time t must equal the real value exactly 24 hours earlier
    assert feats.loc[t, "lag_24h"] == clean.loc[t - pd.Timedelta(hours=24), config.TARGET]
    assert not feats[FEATURE_COLUMNS].isna().any().any()


def test_metrics_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    m = regression_metrics(y, y)
    assert m["RMSE"] == 0 and m["MAE"] == 0 and m["R2"] == 1.0
