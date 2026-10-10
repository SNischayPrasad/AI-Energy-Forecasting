"""
config.py
---------
One place for every setting in the project (paths, dates, model settings,
business rules). Change a value here and the whole pipeline picks it up.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (built relative to this file, so the project runs from any folder)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
IMAGES_DIR = ROOT_DIR / "images"

RAW_DATA_FILE = DATA_RAW_DIR / "energy_consumption.csv"          # historical meter data
WEATHER_FORECAST_FILE = DATA_RAW_DIR / "weather_forecast.csv"    # next-week weather
CLEAN_DATA_FILE = DATA_PROCESSED_DIR / "energy_clean.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Virtual simulation settings (synthetic smart-meter data)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
SIM_START = "2024-01-01 00:00"
SIM_END = "2025-12-31 23:00"          # 2 full years of hourly readings
SITE_NAME = "Virtual Smart-City Zone (commercial + residential feeder)"

# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------
TARGET = "energy_kwh"
TEST_DAYS = 60                        # last 60 days are held out for testing
FORECAST_HORIZON_HOURS = 168          # forecast the next 7 days (7 x 24 hours)

# ---------------------------------------------------------------------------
# Business rules used in the simulation (illustrative Indian ToU tariff)
# ---------------------------------------------------------------------------
TARIFF_INR_PER_KWH = {
    "peak": 9.0,       # 18:00 - 21:59
    "normal": 7.0,     # 06:00 - 17:59
    "off_peak": 5.0,   # 22:00 - 05:59
}
PEAK_HOURS = range(18, 22)
OFF_PEAK_HOURS = list(range(22, 24)) + list(range(0, 6))
FLEXIBLE_LOAD_SHARE = 0.10            # 10% of peak-hour load can be shifted (HVAC pre-cooling, EV charging, batch jobs)
DEMAND_LIMIT_PERCENTILE = 90          # contract demand limit = 90th percentile of last 30 days


def ensure_dirs() -> None:
    """Create every output folder if it does not exist yet."""
    for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)
