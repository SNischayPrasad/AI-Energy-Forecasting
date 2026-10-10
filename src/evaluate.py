"""
evaluate.py
-----------
Error metrics used by energy companies to judge a forecast.

MAE  - average error in kWh ("we are off by X kWh per hour on average")
RMSE - like MAE but punishes big misses more (big misses = expensive)
MAPE - average error in % (easy to explain to managers)
R2   - share of the variation the model explains (1.0 = perfect)
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAPE_%": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }
