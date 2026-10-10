# Project Guide: AI-Powered Energy Consumption Forecasting System

This guide is written for beginners. It explains **what** the project does, **why** each step exists, and **how** to run and verify it. Read it once end to end, then keep it open while you work.

---

## A. Project Explanation

### What is AI-based energy consumption forecasting?

**Simple version:** It works like a weather forecast, but for electricity use. We look at how much power a place used in the past, and what affected it (time of day, weekday, temperature, holidays). Then a computer learns the pattern and predicts how much power will be used every hour next week.

**Technical version:** It is a **supervised time-series regression** problem. The target `y(t)` is energy (kWh) in hour `t`. Features are drawn from:
- the calendar (hour, weekday, holiday),
- exogenous drivers (temperature, degree-days),
- autoregressive history (lags at t−24h, t−48h, t−168h, rolling statistics).

A model `f(X) → ŷ` is trained on a chronological training window. It is validated on a later hold-out window and then used to produce a multi-step (168-hour) forecast. Quality is measured with MAE, RMSE, MAPE and R², and always compared against a naive baseline.

### What problem does it solve?

Electricity has to be produced **at the same moment** it is consumed. If a company guesses wrong:

| Guess too LOW | Guess too HIGH |
|---|---|
| Buys expensive emergency power at peak prices | Pays for power nobody used |
| Crosses its contract demand limit and pays penalty charges | Over-sizes generators, cooling and contracts |
| Risks blackouts and overloaded transformers | Wastes money and adds carbon emissions |

A good forecast reduces both kinds of error.

### Why it matters in each industry

- **Smart cities:** decide when to run water pumps, street lights and EV chargers, and spot overloaded feeders before they trip.
- **Electricity boards (DISCOMs):** must submit **day-ahead schedules** to the grid operator. Deviations are penalised under deviation-settlement rules, so better forecasts mean a smaller penalty bill.
- **Manufacturing plants:** the bill includes a **maximum-demand charge**. Forecasting peaks lets them stagger heavy machines.
- **Data centres:** cooling is 30–40% of energy. Forecasts drive cooling plans and power-purchase agreements.
- **Renewable companies:** solar and wind output varies, so they must match it to forecast demand and decide when to charge or discharge batteries.

### How companies save money with a forecast

1. **Day-ahead procurement:** buy close to the real need, which means smaller imbalance penalties.
2. **Peak shaving / load shifting:** move flexible load (pre-cooling, EV charging, batch jobs) from peak tariff hours to off-peak hours.
3. **Demand-limit alerts:** warn operators hours in advance.
4. **Maintenance planning:** schedule shutdowns in low-demand windows.
5. **Capacity planning:** long-term trend forecasts decide when to add transformers or solar panels.

### Full workflow

| Step | What happens | File |
|---|---|---|
| 1. Data collection | Hourly smart-meter readings + temperature (here simulated) | `data_generator.py` |
| 2. Preprocessing | Remove duplicates, fill missing hours, remove meter glitches | `preprocessing.py` |
| 3. Feature engineering | Hour, weekday, holiday, temperature, lags, rolling means | `features.py` |
| 4. Model training | Train 3 ML models + 1 baseline on the first ~22 months | `train.py` |
| 5. Evaluation | Score on the last 60 unseen days (MAE, RMSE, MAPE, R²) | `evaluate.py` |
| 6. Forecasting | Use the best model to predict the next 168 hours | `forecast.py` |
| 7. Simulation | Turn forecast into alerts, savings, procurement decisions | `simulation.py` |
| 8. Visualisation | 10 charts saved to `images/` | `visualize.py` |

---

## B. Tech Stack Options

| | **Option A: Easiest** | **Option B: Intermediate** ✅ | **Option C: Advanced** |
|---|---|---|---|
| Tools | Python, Pandas, Matplotlib, scikit-learn | Option A + feature engineering, HistGradientBoosting, joblib, pytest | TensorFlow/PyTorch, Prophet, statsmodels, Optuna, MLflow, Streamlit |
| Models | Linear Regression only | Naive baseline, Linear Regression, Random Forest, Gradient Boosting | ARIMA/SARIMA, Prophet, LSTM/GRU, Temporal Fusion Transformer |
| Difficulty | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| GPU needed? | No | No | Recommended for LSTM/TFT |
| Run time | Seconds | About 1 minute | 10 min to hours |
| Expected result | ~8–10% error, basic graph | ~3–4% error, model comparison, business report | ~3% error (similar!), but much more complexity |
| GitHub value | Low: looks like a tutorial | **High: industry pipeline, clean code, tests, business impact** | High, but hard to finish and explain |

### C. Selected approach: **Option B**

Why it is the best fit for a student:
- **Quick execution:** runs on any laptop in about a minute, with no GPU.
- **Strong GitHub proof:** modular code, tests, logs, 10 charts, and a business report with ₹ numbers.
- **Clear understanding:** you can explain every line in an interview. Tree models with good lag features are what many energy companies actually run in production, and they often beat LSTMs on tabular time-series data.
- **Upgrade path:** Option C items are listed as "Future Improvements" in the README.

---

## D. Architecture

### Block diagram

```
          ┌──────────────────────── OFFLINE (training) ──────────────────────────┐
          │                                                                       │
 [Smart-meter CSV] ─▶ [Load & profile] ─▶ [Clean] ─▶ [Features] ─▶ [Time split]  │
   2 yrs hourly        data_loader        preprocessing   features      train/test │
                                                                         │        │
                                       ┌─────────────────────────────────┘        │
                                       ▼                                          │
                        [Train 4 models] ─▶ [Evaluate on test] ─▶ [Pick best]     │
                         train.py            evaluate.py            │             │
                                                                     ▼             │
                                              models/best_model.joblib + metadata  │
          └───────────────────────────────────────────────────────────┬───────────┘
                                                                      │
          ┌──────────────────────── ONLINE (forecasting) ─────────────▼───────────┐
 [Weather forecast] ─▶ [Recursive 7-day forecast] ─▶ [Business simulation]        │
   next 168 h            forecast.py                   simulation.py              │
                                                         │                         │
                         ┌───────────────────────────────┴─────────────┐          │
                         ▼                                             ▼          │
                  images/*.png (charts)                outputs/*.csv, *.json, *.txt│
          └───────────────────────────────────────────────────────────────────────┘
```

### Module responsibilities

| Module | Input | Output | Key idea |
|---|---|---|---|
| `config.py` | n/a | constants | One place for every setting |
| `data_generator.py` | settings | raw CSV + weather CSV | Realistic drivers + injected data problems |
| `data_loader.py` | CSV | DataFrame + profile | Validates the file exists, sorts by time |
| `preprocessing.py` | raw df | clean hourly df + report | Robust MAD outlier detection, time interpolation |
| `features.py` | clean df | feature matrix | Lags ≥ 24 h only, so there is **no leakage** |
| `train.py` | clean df | best model, metrics, predictions | Chronological split, baseline comparison, refit on all data |
| `evaluate.py` | y, ŷ | metrics dict | MAE / RMSE / MAPE / R² |
| `forecast.py` | model + history + weather | 168-h forecast + interval | Day-by-day recursion |
| `simulation.py` | forecast | alerts, plan, report | Converts kWh into ₹ and decisions |
| `visualize.py` | dataframes | PNG charts | Consistent, colour-blind-safe style |
| `main.py` | CLI args | runs everything | `--step generate/train/forecast/all` |

### Data flow (shapes)

`17,559 raw rows` → dedupe → `17,544 hourly rows` → features (drop first 7 days needed for lags) → `17,353 rows × 19 features` → split → `15,913 train / 1,440 test` → best model → `168 forecast rows`.

---

## E. Folder Structure Explained

| Folder / file | Purpose | Commit to Git? |
|---|---|---|
| `data/raw/` | Original input data, never edited by hand | ✅ (small, reproducible) |
| `data/processed/` | Cleaned data produced by the pipeline | ❌ (regenerated) |
| `notebooks/` | Exploration and experiments (EDA) | ✅ |
| `src/` | Reusable, production-style Python modules | ✅ |
| `models/` | Saved model + `model_metadata.json` | metadata ✅, `.joblib` ❌ |
| `outputs/` | Metrics, predictions, business report, run log | ✅ (proof of work) |
| `images/` | Charts shown in the README | ✅ |
| `docs/` | This guide + GitHub guide | ✅ |
| `tests/` | Automated checks (`pytest`) | ✅ |
| `main.py` | Single entry point | ✅ |
| `requirements.txt` | Exact libraries needed | ✅ |
| `.gitignore` | Tells Git what NOT to upload | ✅ |

---

## F. Installation & Environment Setup

**Python version:** 3.10, 3.11, 3.12 or 3.13 (tested on 3.13).

### Windows (PowerShell)

```powershell
cd AI-Energy-Forecasting
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Mac / Linux

```bash
cd AI-Energy-Forecasting
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Verify

```bash
python -c "import pandas, sklearn, matplotlib; print('OK', sklearn.__version__)"
```

Run `deactivate` to leave the virtual environment.

---

## G. Code Walkthrough (what each file does)

All code lives in `src/` and is heavily commented. Read it in this order:

1. **`config.py`**: paths, dates (2024-01-01 → 2025-12-31), `TEST_DAYS = 60`, `FORECAST_HORIZON_HOURS = 168`, tariff (₹9 peak / ₹7 normal / ₹5 off-peak), `FLEXIBLE_LOAD_SHARE = 0.10`.
2. **`data_generator.py`**: builds `energy = base × growth × daily × weekly × holiday + cooling + heating`, adds 3% noise, then breaks the data on purpose (missing values, outage, spikes, duplicates). It also writes next week's weather forecast.
3. **`data_loader.py`**: `load_raw_data()` and `summarize()` (row count, missing values, duplicates, min/max).
4. **`preprocessing.py`**: `clean_data()` returns the clean frame and a report like `{"duplicates_removed": 15, "outliers_detected": 25, "missing_after_fill": 0}`.
5. **`features.py`**: `build_features()` adds 19 features. **Interview point:** the shortest lag is 24 h because, when forecasting tomorrow, you don't yet know tomorrow's consumption.
6. **`evaluate.py`**: `regression_metrics()`.
7. **`train.py`**: `train_and_evaluate()` handles the chronological split, a seasonal-naive baseline, 3 models, the leaderboard, permutation importance and a 90% prediction interval from test residuals. It then **refits the best model on all data** and saves it.
8. **`forecast.py`**: `forecast_future()` predicts 24 hours at a time and feeds predictions back in as lags for the next day (recursive strategy).
9. **`simulation.py`**: peak alerts (demand limit = 90th percentile of the last 30 days), load-shifting plan, ₹ cost before/after, and the procurement backtest (naive vs AI imbalance penalty).
10. **`visualize.py`**: the 10 charts.
11. **`main.py`**: wires it all together with logging to the console and `outputs/run_log.txt`.

---

## H. Virtual Simulation

### Why the simulated data represents reality

| Real-world effect | How it is simulated |
|---|---|
| People wake up, go to work, come home | Gaussian morning bump at 09:00 + office plateau + evening peak at 19:30 |
| Offices close on weekends / holidays | ×0.82 weekends, ×0.78 holidays |
| AC use rises with heat | +38 kWh per °C above 24 °C |
| Cities grow | +3% per year |
| Weather fronts | Smoothed random walk added to a seasonal + daily temperature cycle |
| Unpredictable behaviour | ±3% random noise |
| Faulty meters / comms | Missing values, a 10-h outage, 25 spikes/zeros, 15 duplicate rows |

### How the forecast simulates business decisions

1. **Step 1: Generate history.** `python main.py --step generate` creates 2 years of meter data and next week's weather forecast.
2. **Step 2: Clean the data** like a data engineer receiving a raw meter export.
3. **Step 3: Train and select a model** like a data scientist. The baseline must be beaten.
4. **Step 4: Forecast the next 7 days** like the utility's forecasting desk does every morning.
5. **Step 5: Peak alerts.** Compare each forecast hour with the contract demand limit. Result: **6 alert hours**.
6. **Step 6: Load shifting.** Move 10% of 18:00–22:00 load into the same day's off-peak hours. Result: **0 alert hours, ₹13,685/week saved**.
7. **Step 7: Procurement backtest.** If we had bought power based on the forecast over the last 60 days, how much imbalance penalty would we have paid compared with the naive "same as last week" guess? Result: **₹92,079 saved (46.5%)**.
8. **Step 8: Report.** `outputs/business_report.txt` is the weekly briefing for the energy manager.

### Graphs to generate (all automatic)

`01_energy_timeseries`, `02_hour_weekday_heatmap`, `03_temperature_vs_energy`, `04_data_cleaning`, `05_model_comparison`, `06_actual_vs_predicted`, `07_feature_importance`, `08_residuals`, `09_future_forecast`, `10_business_simulation`

### Outputs saved

| File | Contents |
|---|---|
| `outputs/run_log.txt` | Full training log with timestamps (proof of execution) |
| `outputs/model_metrics.csv` | Leaderboard |
| `outputs/test_predictions.csv` | Actual vs every model on the test set |
| `outputs/feature_importance.csv` | Feature ranking |
| `outputs/future_forecast.csv` | 168 hours with lower/upper bounds |
| `outputs/peak_alerts.csv` | Hours above the demand limit |
| `outputs/load_shifting_plan.csv` | Hour-by-hour plan and ₹ cost before/after |
| `outputs/business_report.txt/.json` | Weekly briefing |
| `models/model_metadata.json` | Best model, metrics, features, training time |

---

## I. How to Run

```bash
python main.py                    # everything, about 1 minute
python main.py --step generate    # dataset only
python main.py --step train       # clean + train + charts 01-08
python main.py --step forecast    # forecast + simulation + charts 09-10 (needs a trained model)
python -m pytest -q               # 4 tests should pass
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### Sample console output

```
STEP 2/4  Loading and cleaning data ...
Cleaning report: {"rows_in": 17559, "duplicates_removed": 15, "missing_before_fill": 115, "outliers_detected": 25, "missing_after_fill": 0, "rows_out": 17544}
STEP 3/4  Training and comparing models ...
Train: 15,913 rows (2024-01-08 23:00:00 -> 2025-11-01 23:00:00)
Test : 1,440 rows (2025-11-02 00:00:00 -> 2025-12-31 23:00:00)
Seasonal Naive (baseline)  {'MAE': 68.78, 'RMSE': 98.12, 'MAPE_%': 6.24, 'R2': 0.7766}
Linear Regression          {'MAE': 46.97, 'RMSE': 59.75, 'MAPE_%': 4.33, 'R2': 0.9171}
Random Forest              {'MAE': 38.38, 'RMSE': 50.1, 'MAPE_%': 3.47, 'R2': 0.9418}
Gradient Boosting          {'MAE': 36.8, 'RMSE': 46.43, 'MAPE_%': 3.36, 'R2': 0.95}
Best model: Gradient Boosting
```

### Expected graphs

- **06 Actual vs predicted:** the orange line hugs the blue line. The grey dashed baseline misses badly around 25 Dec (it copies the previous week).
- **09 Future forecast:** 2 weeks of blue history, then 7 days of orange forecast with a shaded band. Some peaks touch the red limit line.
- **10 Business simulation:** the aqua "after shifting" line stays under the red limit. The bars show ₹1.6k–2.2k saved per day.

---

## Implementation Phases (what, why, expected output, mistakes, verify)

| Phase | What to do | Why | Expected output | Common mistakes | How to verify |
|---|---|---|---|---|---|
| 1 Setup | Create folders, venv, install requirements | Reproducible environment | `import sklearn` works | Installing globally; forgetting to activate venv | `pip list` shows pandas, sklearn |
| 2 Dataset loading | `--step generate`, then `load_raw_data()` | Need history to learn from | 17,559 rows, 4 columns | Not parsing dates, so the timestamp stays a string | `df.dtypes` shows `datetime64` |
| 3 Data cleaning | `clean_data()` | Glitches teach the model wrong patterns | 17,544 rows, 0 NaN | Using mean/std outlier rules (spikes inflate std); `dropna` creates gaps | Cleaning report + chart 04 |
| 4 Feature engineering | `build_features()` | Models only understand numbers | 19 features | **Leakage**: using lag-1 h or centered rolling windows | `test_lag_features_do_not_leak_future` passes |
| 5 Model building | `train_and_evaluate()` | Find the best algorithm | 4 models trained | `train_test_split(shuffle=True)` on time series | Train dates < test dates in the log |
| 6 Evaluation | MAE, RMSE, MAPE, R² vs baseline | Prove the model adds value | Leaderboard CSV | Only reporting R²; no baseline | Best RMSE < baseline RMSE |
| 7 Forecasting | `forecast_future()` | The actual business deliverable | 168-row CSV | Using lags that don't exist yet in the future | Forecast has no NaN; chart 09 looks smooth |
| 8 Visualization | `visualize.py` | Humans decide from pictures | 10 PNGs | Unlabelled axes, overlapping legends | Open every image |
| 9 GitHub publishing | See `GITHUB_GUIDE.md` | Proof of work | Public repo | Uploading venv/ or big model files | `.gitignore` works; repo size < 5 MB |
| 10 Final output | README with results + images | Recruiters read the README first | Professional repo page | No results or images in the README | View the repo in an incognito window |
