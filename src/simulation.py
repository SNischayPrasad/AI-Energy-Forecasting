"""
simulation.py
-------------
VIRTUAL BUSINESS SIMULATION - turns a forecast into decisions and money.

An energy manager uses next week's forecast to answer three questions:

1. "Will we cross our contract demand limit?"      -> peak alerts
2. "How much can we save by moving flexible load?"  -> load-shifting plan
3. "How much power should we buy day-ahead, and how
    much does a better forecast save vs. guessing?"  -> procurement backtest
"""
import json

import numpy as np
import pandas as pd

from src import config

IMBALANCE_PENALTY_INR = 2.0   # cost per kWh of buying too much / too little power


def tariff_band(hour: int) -> str:
    if hour in config.PEAK_HOURS:
        return "peak"
    if hour in config.OFF_PEAK_HOURS:
        return "off_peak"
    return "normal"


def apply_tariff(df: pd.DataFrame, kwh_col: str) -> pd.Series:
    rates = df.index.hour.map(lambda h: config.TARIFF_INR_PER_KWH[tariff_band(h)])
    return df[kwh_col] * np.asarray(rates)


def peak_alerts(forecast: pd.DataFrame, clean_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    last_30_days = clean_df[config.TARGET].iloc[-24 * 30:]
    limit = float(np.percentile(last_30_days, config.DEMAND_LIMIT_PERCENTILE))
    alerts = forecast[forecast["forecast_kwh"] > limit].copy()
    alerts["demand_limit_kwh"] = round(limit, 2)
    alerts["excess_kwh"] = (alerts["forecast_kwh"] - limit).round(2)
    return alerts, limit


def load_shifting_plan(forecast: pd.DataFrame) -> pd.DataFrame:
    """Move FLEXIBLE_LOAD_SHARE of peak-hour energy into the same day's off-peak hours."""
    plan = forecast[["forecast_kwh"]].copy()
    plan["band"] = [tariff_band(h) for h in plan.index.hour]
    plan["optimized_kwh"] = plan["forecast_kwh"]
    for _, day in plan.groupby(plan.index.date):
        peak_idx = day.index[day["band"] == "peak"]
        off_idx = day.index[day["band"] == "off_peak"]
        moved = plan.loc[peak_idx, "forecast_kwh"].sum() * config.FLEXIBLE_LOAD_SHARE
        plan.loc[peak_idx, "optimized_kwh"] *= (1 - config.FLEXIBLE_LOAD_SHARE)
        plan.loc[off_idx, "optimized_kwh"] += moved / len(off_idx)
    plan["cost_before_inr"] = apply_tariff(plan, "forecast_kwh").round(2)
    plan["cost_after_inr"] = apply_tariff(plan, "optimized_kwh").round(2)
    return plan


def procurement_backtest(test_predictions: pd.DataFrame, best_name: str) -> dict:
    """On the 60 test days: what would buying power based on each forecast have cost in penalties?"""
    actual = test_predictions["actual"]
    naive_cost = (actual - test_predictions["Seasonal Naive"]).abs().sum() * IMBALANCE_PENALTY_INR
    ai_cost = (actual - test_predictions[best_name]).abs().sum() * IMBALANCE_PENALTY_INR
    return {
        "test_days": config.TEST_DAYS,
        "penalty_inr_per_kwh": IMBALANCE_PENALTY_INR,
        "imbalance_cost_naive_inr": round(float(naive_cost), 0),
        "imbalance_cost_ai_inr": round(float(ai_cost), 0),
        "savings_inr": round(float(naive_cost - ai_cost), 0),
        "savings_pct": round(float((naive_cost - ai_cost) / naive_cost * 100), 1),
    }


def run_business_simulation(forecast: pd.DataFrame, clean_df: pd.DataFrame,
                            test_predictions: pd.DataFrame, best_name: str) -> dict:
    alerts, limit = peak_alerts(forecast, clean_df)
    plan = load_shifting_plan(forecast)
    alerts_after = int((plan["optimized_kwh"] > limit).sum())
    backtest = procurement_backtest(test_predictions, best_name)

    cost_before = plan["cost_before_inr"].sum()
    cost_after = plan["cost_after_inr"].sum()
    report = {
        "site": config.SITE_NAME,
        "forecast_window": f"{forecast.index.min()} -> {forecast.index.max()}",
        "forecast_total_kwh": round(float(forecast["forecast_kwh"].sum()), 0),
        "forecast_peak_kwh": round(float(forecast["forecast_kwh"].max()), 1),
        "forecast_peak_time": str(forecast["forecast_kwh"].idxmax()),
        "demand_limit_kwh": round(limit, 1),
        "peak_alert_hours_before": len(alerts),
        "peak_alert_hours_after_shifting": alerts_after,
        "weekly_cost_before_inr": round(float(cost_before), 0),
        "weekly_cost_after_inr": round(float(cost_after), 0),
        "weekly_savings_inr": round(float(cost_before - cost_after), 0),
        "annualized_savings_inr": round(float((cost_before - cost_after) * 52), 0),
        "procurement_backtest": backtest,
    }

    alerts.to_csv(config.OUTPUTS_DIR / "peak_alerts.csv", index_label="timestamp")
    plan.to_csv(config.OUTPUTS_DIR / "load_shifting_plan.csv", index_label="timestamp")
    (config.OUTPUTS_DIR / "business_report.json").write_text(json.dumps(report, indent=2))
    (config.OUTPUTS_DIR / "business_report.txt").write_text(format_report(report), encoding="utf-8")
    return {"report": report, "alerts": alerts, "plan": plan, "limit": limit}


def format_report(r: dict) -> str:
    b = r["procurement_backtest"]
    return f"""
==================== ENERGY MANAGER WEEKLY BRIEFING ====================
Site            : {r['site']}
Forecast window : {r['forecast_window']}

1) DEMAND OUTLOOK
   Total energy expected : {r['forecast_total_kwh']:,.0f} kWh
   Highest hour          : {r['forecast_peak_kwh']:,.1f} kWh at {r['forecast_peak_time']}
   Contract demand limit : {r['demand_limit_kwh']:,.1f} kWh/h

2) PEAK ALERTS
   Hours above limit (as forecast)     : {r['peak_alert_hours_before']}
   Hours above limit (after shifting)  : {r['peak_alert_hours_after_shifting']}

3) LOAD SHIFTING (move {int(config.FLEXIBLE_LOAD_SHARE*100)}% of 18:00-22:00 load to off-peak)
   Weekly cost before : INR {r['weekly_cost_before_inr']:,.0f}
   Weekly cost after  : INR {r['weekly_cost_after_inr']:,.0f}
   Weekly savings     : INR {r['weekly_savings_inr']:,.0f}
   Annualized savings : INR {r['annualized_savings_inr']:,.0f}

4) PROCUREMENT BACKTEST ({b['test_days']} days, penalty INR {b['penalty_inr_per_kwh']}/kWh of error)
   Imbalance cost with naive guess : INR {b['imbalance_cost_naive_inr']:,.0f}
   Imbalance cost with AI forecast : INR {b['imbalance_cost_ai_inr']:,.0f}
   Saved by AI                     : INR {b['savings_inr']:,.0f} ({b['savings_pct']}%)
=========================================================================
"""
