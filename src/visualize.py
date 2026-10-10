"""
visualize.py
------------
Every chart in the project. All images are saved to images/ so they can be
shown in the README and used as proof of work.
"""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src import config

# ---- one consistent, colour-blind-safe style for every chart ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
ALERT = "#e34948"                                       # status: critical (limit / alerts only)
MUTED = "#8a8984"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SEQ_BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
    "lines.linewidth": 2, "text.color": INK,
})


def _date_axis(ax, interval_days: int = 2):
    """Readable, non-overlapping date labels such as '05 Jan'."""
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval_days))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def _save(fig, name: str):
    path = config.IMAGES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_raw_timeseries(clean_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    daily = clean_df[config.TARGET].resample("D").sum() / 1000
    ax1.plot(daily.index, daily.values, color=BLUE, linewidth=1.4)
    ax1.set_title("Daily energy consumption - 2 years (MWh/day)")
    ax1.set_ylabel("MWh per day")

    week = clean_df[config.TARGET].iloc[-24 * 14:]
    ax2.plot(week.index, week.values, color=BLUE, linewidth=1.6)
    ax2.set_title("Hourly zoom - last 14 days (daily peaks + quieter weekends)")
    ax2.set_ylabel("kWh per hour")
    _date_axis(ax2)
    return _save(fig, "01_energy_timeseries.png")


def plot_hour_weekday_heatmap(clean_df: pd.DataFrame):
    d = clean_df[[config.TARGET]].copy()
    d["hour"], d["dow"] = d.index.hour, d.index.dayofweek
    pivot = d.pivot_table(index="dow", columns="hour", values=config.TARGET, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(pivot.values, aspect="auto", cmap=SEQ_BLUES)
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.grid(False)
    ax.set_title("Average load by hour and weekday (kWh) - when does the city use power?")
    fig.colorbar(im, ax=ax, label="avg kWh per hour", shrink=0.9)
    return _save(fig, "02_hour_weekday_heatmap.png")


def plot_temperature_vs_energy(clean_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sample = clean_df.sample(4000, random_state=config.RANDOM_SEED)
    ax.scatter(sample["temperature_c"], sample[config.TARGET], s=8, alpha=0.35, color=BLUE, linewidths=0)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("kWh per hour")
    ax.set_title("Hotter hours need more power (air-conditioning load)")
    return _save(fig, "03_temperature_vs_energy.png")


def plot_cleaning(raw_df: pd.DataFrame, clean_df: pd.DataFrame):
    raw = raw_df.drop_duplicates("timestamp").set_index("timestamp")[config.TARGET]
    spike_time = raw.idxmax()
    lo, hi = spike_time - pd.Timedelta(days=4), spike_time + pd.Timedelta(days=4)
    r, c = raw.loc[lo:hi], clean_df[config.TARGET].loc[lo:hi]
    bad = r[(r.isna()) | ((r - c).abs() > 1)]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(r.index, r.values, color=MUTED, linewidth=1.2, label="Raw meter data")
    ax.plot(c.index, c.values, color=BLUE, linewidth=2, label="Cleaned data")
    ax.scatter(bad.index, r.reindex(bad.index).fillna(0).values, color=ALERT, s=40, zorder=5,
               edgecolors=SURFACE, linewidths=1.5, label="Glitch / missing (fixed)")
    ax.set_title("Data cleaning: meter glitches removed and gaps filled")
    ax.set_ylabel("kWh per hour")
    ax.legend(loc="upper left")
    return _save(fig, "04_data_cleaning.png")


def plot_model_comparison(metrics: pd.DataFrame):
    m = metrics.sort_values("RMSE", ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, title in [(ax1, "RMSE", "RMSE (kWh) - lower is better"),
                           (ax2, "MAPE_%", "MAPE (%) - lower is better")]:
        bars = ax.barh(m["model"], m[col], color=BLUE, height=0.6)
        ax.bar_label(bars, fmt="%.2f", padding=4, color=INK_2)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, m[col].max() * 1.18)
    return _save(fig, "05_model_comparison.png")


def plot_actual_vs_predicted(predictions: pd.DataFrame, best_name: str):
    p = predictions.iloc[-24 * 14:]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(p.index, p["actual"], color=BLUE, label="Actual")
    ax.plot(p.index, p[best_name], color=ORANGE, label=f"Predicted ({best_name})")
    ax.plot(p.index, p["Seasonal Naive"], color=MUTED, linewidth=1.2, linestyle="--",
            label="Baseline: same hour last week")
    ax.set_title("Actual vs predicted - last 14 days of unseen test data")
    ax.set_ylabel("kWh per hour")
    _date_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    return _save(fig, "06_actual_vs_predicted.png")


def plot_feature_importance(importance: pd.DataFrame):
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top["feature"], top["importance"], color=BLUE, height=0.6)
    ax.set_title("What drives the forecast? (permutation importance)")
    ax.set_xlabel("Increase in error when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    return _save(fig, "07_feature_importance.png")


def plot_residuals(predictions: pd.DataFrame, best_name: str):
    err = predictions["actual"] - predictions[best_name]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.hist(err, bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    ax1.axvline(0, color=INK_2, linewidth=1)
    ax1.set_title("Forecast errors are small and centred on zero")
    ax1.set_xlabel("Actual - predicted (kWh)")
    ax1.set_ylabel("Hours")

    ax2.scatter(predictions["actual"], predictions[best_name], s=8, alpha=0.4, color=BLUE, linewidths=0)
    lims = [predictions["actual"].min(), predictions["actual"].max()]
    ax2.plot(lims, lims, color=INK_2, linewidth=1, linestyle="--")
    ax2.set_title("Predicted vs actual (dots on the line = perfect)")
    ax2.set_xlabel("Actual kWh")
    ax2.set_ylabel("Predicted kWh")
    return _save(fig, "08_residuals.png")


def plot_future_forecast(clean_df: pd.DataFrame, forecast: pd.DataFrame, limit: float):
    hist = clean_df[config.TARGET].iloc[-24 * 14:]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist.index, hist.values, color=BLUE, label="History (actual)")
    ax.plot(forecast.index, forecast["forecast_kwh"], color=ORANGE, label="Forecast - next 7 days")
    ax.fill_between(forecast.index, forecast["lower_kwh"], forecast["upper_kwh"],
                    color=ORANGE, alpha=0.18, linewidth=0, label="90% prediction interval")
    ax.axhline(limit, color=ALERT, linewidth=1.4, linestyle="--", label="Contract demand limit")
    ax.axvline(forecast.index[0], color=INK_2, linewidth=1)
    ax.text(forecast.index[0], ax.get_ylim()[0], "  forecast starts", va="bottom", color=INK_2)
    ax.set_title("7-day energy forecast")
    ax.set_ylabel("kWh per hour")
    _date_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    return _save(fig, "09_future_forecast.png")


def plot_business_simulation(plan: pd.DataFrame, limit: float):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(plan.index, plan["forecast_kwh"], color=ORANGE, label="Forecast (no action)")
    ax1.plot(plan.index, plan["optimized_kwh"], color=AQUA, label="After load shifting")
    ax1.axhline(limit, color=ALERT, linewidth=1.4, linestyle="--", label="Contract demand limit")
    for day in pd.unique(plan.index.normalize()):
        ax1.axvspan(day + pd.Timedelta(hours=18), day + pd.Timedelta(hours=22), color=GRID, alpha=0.7, linewidth=0)
    ax1.set_title("Simulation: shift 10% of peak-hour load (grey bands = 18:00-22:00 peak tariff)")
    ax1.set_ylabel("kWh per hour")
    _date_axis(ax1, 1)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)

    savings = (plan["cost_before_inr"] - plan["cost_after_inr"]).resample("D").sum()
    bars = ax2.bar([d.strftime("%a %d %b") for d in savings.index], savings.values, color=AQUA, width=0.6)
    ax2.bar_label(bars, labels=[f"INR {v:,.0f}" for v in savings.values], padding=3, color=INK_2)
    ax2.set_ylim(0, savings.max() * 1.2)
    ax2.set_ylabel("INR saved per day")
    ax2.set_title(f"Daily bill savings from load shifting - week total INR {savings.sum():,.0f}")
    ax2.grid(axis="x", visible=False)
    return _save(fig, "10_business_simulation.png")
