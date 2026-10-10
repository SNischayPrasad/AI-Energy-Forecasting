"""
train.py
--------
Trains several models, compares them on unseen data, saves the best one.

Models (simple -> powerful):
  0. Seasonal Naive  - "same hour last week" (the baseline every model must beat)
  1. Linear Regression
  2. Random Forest
  3. Gradient Boosting (scikit-learn HistGradientBoosting - fast, LightGBM-style)

Time-series rule: NEVER shuffle. We train on the past and test on the most
recent 60 days, exactly like the model would be used in real life.
"""
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluate import regression_metrics
from src.features import FEATURE_COLUMNS, build_features

log = logging.getLogger(__name__)


def time_based_split(feat_df: pd.DataFrame):
    """Everything before the cutoff is training data, everything after is test data."""
    cutoff = feat_df.index.max() - pd.Timedelta(days=config.TEST_DAYS)
    train = feat_df[feat_df.index <= cutoff]
    test = feat_df[feat_df.index > cutoff]
    return train, test


def get_models() -> dict:
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=16, min_samples_leaf=3,
            n_jobs=-1, random_state=config.RANDOM_SEED),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            random_state=config.RANDOM_SEED),
    }


def train_and_evaluate(clean_df: pd.DataFrame) -> dict:
    feat_df = build_features(clean_df)
    train, test = time_based_split(feat_df)
    X_train, y_train = train[FEATURE_COLUMNS], train[config.TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[config.TARGET]
    log.info("Train: %s rows (%s -> %s)", f"{len(train):,}", train.index.min(), train.index.max())
    log.info("Test : %s rows (%s -> %s)", f"{len(test):,}", test.index.min(), test.index.max())

    predictions = pd.DataFrame({"actual": y_test})
    rows = []

    # Baseline: no machine learning at all
    predictions["Seasonal Naive"] = test["lag_168h"]
    m = regression_metrics(y_test, predictions["Seasonal Naive"])
    rows.append({"model": "Seasonal Naive (baseline)", **m, "train_seconds": 0.0})
    log.info("%-26s %s", "Seasonal Naive (baseline)", m)

    fitted = {}
    for name, model in get_models().items():
        t0 = datetime.now()
        model.fit(X_train, y_train)
        secs = (datetime.now() - t0).total_seconds()
        predictions[name] = model.predict(X_test)
        m = regression_metrics(y_test, predictions[name])
        rows.append({"model": name, **m, "train_seconds": round(secs, 2)})
        fitted[name] = model
        log.info("%-26s %s  (trained in %.1fs)", name, m, secs)

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = next(n for n in metrics["model"] if n in fitted)
    best_model = fitted[best_name]
    log.info("Best model: %s", best_name)

    # Which features matter? (permutation importance works for any model)
    imp = permutation_importance(best_model, X_test, y_test, n_repeats=5,
                                 random_state=config.RANDOM_SEED, n_jobs=-1)
    importance = (pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False))

    # Prediction interval from test errors (90% of errors fall inside it)
    residuals = y_test - predictions[best_name]
    q_low, q_high = np.percentile(residuals, [5, 95])

    # Production practice: refit the winning model on ALL available history
    final_model = get_models()[best_name]
    final_model.fit(feat_df[FEATURE_COLUMNS], feat_df[config.TARGET])
    joblib.dump(final_model, config.BEST_MODEL_FILE)

    metadata = {
        "best_model": best_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "features": FEATURE_COLUMNS,
        "test_days": config.TEST_DAYS,
        "test_metrics": metrics[metrics["model"] == best_name].iloc[0].drop("model").to_dict(),
        "residual_q05": round(float(q_low), 2),
        "residual_q95": round(float(q_high), 2),
        "train_rows": len(train),
        "test_rows": len(test),
    }
    config.MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    metrics.to_csv(config.OUTPUTS_DIR / "model_metrics.csv", index=False)
    predictions.to_csv(config.OUTPUTS_DIR / "test_predictions.csv", index_label="timestamp")
    importance.to_csv(config.OUTPUTS_DIR / "feature_importance.csv", index=False)
    log.info("Saved model -> %s", config.BEST_MODEL_FILE)

    return {"metrics": metrics, "predictions": predictions, "importance": importance,
            "best_name": best_name, "metadata": metadata}
