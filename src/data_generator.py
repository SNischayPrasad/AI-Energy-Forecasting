"""
data_generator.py
-----------------
VIRTUAL SIMULATION of a real smart-meter feed.

Real utilities record energy every hour from smart meters. We cannot access a
company's SCADA system, so we simulate one. The simulated load is built from
the same drivers that shape real electricity demand:

    energy = base load
           x long-term growth (city grows ~3% per year)
           x daily routine (morning + evening peaks)
           x weekly routine (weekends / holidays are quieter)
           + weather (air-conditioning when it is hot, heating when cold)
           + random noise (things nobody can predict)

We also inject realistic data-quality problems (missing readings, meter
spikes, duplicate rows) so the cleaning step has real work to do.
"""
import numpy as np
import pandas as pd

from src import config

# Fixed public holidays (month, day) - offices and factories close on these days
HOLIDAYS = [(1, 1), (1, 26), (3, 14), (4, 14), (5, 1), (8, 15), (10, 2), (10, 20), (11, 1), (12, 25)]


def is_holiday(index: pd.DatetimeIndex) -> np.ndarray:
    """Return 1 for timestamps that fall on a holiday, else 0."""
    pairs = list(zip(index.month, index.day))
    return np.array([1 if p in HOLIDAYS else 0 for p in pairs])


def simulate_temperature(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Tropical climate: hottest in May, coolest in January, hottest at ~3 pm."""
    doy = index.dayofyear.values
    hour = index.hour.values
    seasonal = 27 + 7 * np.sin(2 * np.pi * (doy - 46) / 365.25)      # peak around mid-May
    daily = 4.5 * np.sin(2 * np.pi * (hour - 9) / 24)                # peak around 15:00
    # Weather "fronts": slowly-moving random noise (a random walk, smoothed)
    front = pd.Series(rng.normal(0, 0.35, len(index))).cumsum()
    front = (front - front.rolling(24 * 10, min_periods=1).mean()).values
    return np.round(seasonal + daily + front + rng.normal(0, 0.6, len(index)), 2)


def simulate_load(index: pd.DatetimeIndex, temperature: np.ndarray,
                  rng: np.random.Generator, add_noise: bool = True) -> np.ndarray:
    """Turn calendar + weather into an hourly energy value in kWh."""
    hour = index.hour.values
    dow = index.dayofweek.values

    base = 1200.0                                                    # kWh per hour
    years = (index - pd.Timestamp(config.SIM_START)).days.values / 365.25
    growth = 1 + 0.03 * years                                        # 3 % per year

    # Daily profile: low at night, morning ramp, office plateau, evening peak
    daily = (0.70
             + 0.25 * np.exp(-((hour - 9) ** 2) / 6)
             + 0.15 * ((hour >= 10) & (hour <= 17))
             + 0.40 * np.exp(-((hour - 19.5) ** 2) / 4))

    weekly = np.where(dow >= 5, 0.82, 1.0)                           # weekends lower
    holiday = np.where(is_holiday(index) == 1, 0.78, 1.0)

    cooling = 38 * np.clip(temperature - 24, 0, None)                # AC load
    heating = 15 * np.clip(18 - temperature, 0, None)                # heater load

    load = base * growth * daily * weekly * holiday + cooling + heating
    if add_noise:
        load = load * rng.normal(1.0, 0.03, len(index))
    return np.round(load, 2)


def inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Make the data messy, like a real meter export."""
    df = df.copy()
    n = len(df)

    # 1) ~0.6% missing readings (communication failures)
    missing_idx = rng.choice(n, size=int(n * 0.006), replace=False)
    df.loc[df.index[missing_idx], config.TARGET] = np.nan

    # 2) A 10-hour meter outage block
    start = rng.integers(1000, n - 1000)
    df.loc[df.index[start:start + 10], config.TARGET] = np.nan

    # 3) ~25 spikes / glitches (meter reports 3-5x the real value, or zero)
    spike_idx = rng.choice(n, size=25, replace=False)
    factors = rng.choice([0.0, 3.0, 4.0, 5.0], size=25)
    df.loc[df.index[spike_idx], config.TARGET] = df[config.TARGET].iloc[spike_idx].values * factors

    # 4) ~15 duplicated rows (same timestamp exported twice)
    dup_rows = df.iloc[rng.choice(n, size=15, replace=False)]
    df = pd.concat([df, dup_rows]).sort_values("timestamp").reset_index(drop=True)
    return df


def generate_dataset(save: bool = True) -> pd.DataFrame:
    """Create the 2-year historical dataset and the next-week weather forecast."""
    config.ensure_dirs()
    rng = np.random.default_rng(config.RANDOM_SEED)

    index = pd.date_range(config.SIM_START, config.SIM_END, freq="h")
    temperature = simulate_temperature(index, rng)
    energy = simulate_load(index, temperature, rng)

    df = pd.DataFrame({
        "timestamp": index,
        config.TARGET: energy,
        "temperature_c": temperature,
        "is_holiday": is_holiday(index),
    })
    df = inject_data_quality_issues(df, rng)

    # Weather forecast for the 7 days AFTER the history ends.
    # (In real life this comes from a weather API such as IMD / OpenWeather.)
    future_index = pd.date_range(index[-1] + pd.Timedelta(hours=1),
                                 periods=config.FORECAST_HORIZON_HOURS, freq="h")
    future_temp = simulate_temperature(future_index, rng)
    weather = pd.DataFrame({
        "timestamp": future_index,
        "temperature_c": future_temp,
        "is_holiday": is_holiday(future_index),
    })

    if save:
        df.to_csv(config.RAW_DATA_FILE, index=False)
        weather.to_csv(config.WEATHER_FORECAST_FILE, index=False)
    return df


if __name__ == "__main__":
    data = generate_dataset()
    print(data.head(10))
    print(f"\nSaved {len(data):,} rows -> {config.RAW_DATA_FILE}")
