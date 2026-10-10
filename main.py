"""
main.py - one command runs the whole project.

Usage:
    python main.py                     # run everything (generate -> train -> forecast -> simulate)
    python main.py --step generate     # only create the simulated dataset
    python main.py --step train        # clean data + train & compare models
    python main.py --step forecast     # forecast next 7 days + business simulation
"""
import argparse
import json
import logging
import sys

import pandas as pd

from src import config, visualize
from src.data_generator import generate_dataset
from src.data_loader import load_raw_data, load_weather_forecast, summarize
from src.forecast import forecast_future
from src.preprocessing import clean_data
from src.simulation import run_business_simulation
from src.train import train_and_evaluate


def setup_logging():
    config.ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(config.OUTPUTS_DIR / "run_log.txt", mode="w", encoding="utf-8")],
    )
    return logging.getLogger("pipeline")


def step_generate(log):
    log.info("STEP 1/4  Generating virtual smart-meter dataset ...")
    df = generate_dataset()
    log.info("Saved %s rows -> %s", f"{len(df):,}", config.RAW_DATA_FILE.relative_to(config.ROOT_DIR))


def step_prepare(log):
    log.info("STEP 2/4  Loading and cleaning data ...")
    raw = load_raw_data()
    log.info("Raw data profile: %s", json.dumps(summarize(raw)))
    log.info("Preview:\n%s", raw.head().to_string(index=False))
    clean, report = clean_data(raw)
    clean.to_csv(config.CLEAN_DATA_FILE, index_label="timestamp")
    log.info("Cleaning report: %s", json.dumps(report))
    return raw, clean


def step_train(log, raw, clean):
    log.info("STEP 3/4  Training and comparing models ...")
    res = train_and_evaluate(clean)
    log.info("Model leaderboard (test set = last %d days):\n%s",
             config.TEST_DAYS, res["metrics"].to_string(index=False))

    visualize.plot_raw_timeseries(clean)
    visualize.plot_hour_weekday_heatmap(clean)
    visualize.plot_temperature_vs_energy(clean)
    visualize.plot_cleaning(raw, clean)
    visualize.plot_model_comparison(res["metrics"])
    visualize.plot_actual_vs_predicted(res["predictions"], res["best_name"])
    visualize.plot_feature_importance(res["importance"])
    visualize.plot_residuals(res["predictions"], res["best_name"])
    log.info("Saved charts 01-08 -> images/")


def step_forecast(log, clean):
    log.info("STEP 4/4  Forecasting next %d hours + business simulation ...", config.FORECAST_HORIZON_HOURS)
    weather = load_weather_forecast()
    forecast = forecast_future(clean, weather)
    log.info("Forecast preview:\n%s", forecast.head(12).to_string())

    meta = json.loads(config.MODEL_METADATA_FILE.read_text())
    test_preds = pd.read_csv(config.OUTPUTS_DIR / "test_predictions.csv",
                             parse_dates=["timestamp"], index_col="timestamp")
    sim = run_business_simulation(forecast, clean, test_preds, meta["best_model"])
    visualize.plot_future_forecast(clean, forecast, sim["limit"])
    visualize.plot_business_simulation(sim["plan"], sim["limit"])
    log.info("Saved charts 09-10 -> images/")
    print((config.OUTPUTS_DIR / "business_report.txt").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="AI-Powered Energy Consumption Forecasting System")
    parser.add_argument("--step", choices=["all", "generate", "train", "forecast"], default="all")
    args = parser.parse_args()
    log = setup_logging()

    if args.step in ("all", "generate"):
        step_generate(log)
    if args.step in ("all", "train", "forecast"):
        raw, clean = step_prepare(log)
        if args.step in ("all", "train"):
            step_train(log, raw, clean)
        if args.step in ("all", "forecast"):
            step_forecast(log, clean)
    log.info("Done.")


if __name__ == "__main__":
    main()
