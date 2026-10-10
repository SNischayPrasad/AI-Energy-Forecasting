# GitHub Guide: Publishing, the 7-Day Commit Plan, and a Proof Checklist

---

## J. GitHub Upload Strategy

### 1. Repository details (copy-paste)

- **Repository name:** `AI-Energy-Forecasting`
- **Description:** `⚡ ML system that forecasts hourly electricity demand 7 days ahead (3.4% MAPE) and simulates business decisions: peak alerts, load shifting & procurement savings. Python · scikit-learn · Pandas`
- **Topics (tags):** `machine-learning` `time-series` `energy-forecasting` `load-forecasting` `python` `scikit-learn` `data-science` `smart-grid` `gradient-boosting` `pandas` `sustainability` `forecasting`
- **Visibility:** Public · **License:** MIT · do **not** tick "Add README" (you already have one)

### 2. One-time Git setup

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 3. Create the repo and push

1. On github.com go to **New repository**, fill in the details above, and click **Create repository**.
2. In your project folder:

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: project structure and README"
git remote add origin https://github.com/SNischayPrasad/AI-Energy-Forecasting.git
git push -u origin main
```

After that, for every change:

```bash
git status                    # see what changed
git add <files>
git commit -m "feat: add lag features for day-ahead forecasting"
git push
```

### 4. Commit message style (looks professional)

| Prefix | Use for | Example |
|---|---|---|
| `feat:` | new functionality | `feat: add recursive 7-day forecast` |
| `fix:` | bug fix | `fix: remove data leakage from rolling mean` |
| `docs:` | README / docs | `docs: add results table and screenshots` |
| `test:` | tests | `test: add leakage test for lag features` |
| `refactor:` | cleanup, no behaviour change | `refactor: move settings into config.py` |
| `chore:` | setup files | `chore: add requirements and .gitignore` |

Make small, meaningful commits, not one giant "final upload". Recruiters look at the commit history.

### 5. Uploading visuals

The charts are in `images/` and are committed with the code. The README references them with relative paths (`![](images/06_actual_vs_predicted.png)`), so they render on GitHub automatically. For a demo GIF, record the terminal with **ScreenToGif** (Windows) or **Kap** (Mac). Save it as `images/demo.gif` and add it near the top of the README.

### 6. Making the repo professional

- [ ] Hero image at the top of the README (`09_future_forecast.png`)
- [ ] Results table with real numbers in the first screenful
- [ ] Topics/tags added (gear icon next to "About")
- [ ] Website field in "About": your LinkedIn post or portfolio
- [ ] `LICENSE` file has your name
- [ ] No `venv/`, `__pycache__/` or `.joblib` files in the repo
- [ ] **Pin** the repo on your GitHub profile
- [ ] Optional: add a GitHub Actions workflow that runs `pytest` (gives a green ✔ badge)
- [ ] Optional: tag a release `v1.0`

---

## L. Day-by-Day Commit Plan (7 days)

Build it **for real** over 7 days. The commit history then shows genuine progress and you understand each part. On each day, retype or reread that day's files rather than copying everything on Day 1.

| Day | Work | Files to commit | Commit messages | Proof to capture |
|---|---|---|---|---|
| **1: Setup** | Create repo, venv, folders, config | `README.md` (short draft), `requirements.txt`, `.gitignore`, `src/__init__.py`, `src/config.py`, empty folders with `.gitkeep` | `chore: initial project structure`, `chore: add requirements and gitignore`, `feat: add central config` | Screenshot of the folder tree in VS Code + terminal showing `pip install` success |
| **2: Dataset** | Simulator + loader + EDA notebook | `src/data_generator.py`, `src/data_loader.py`, `data/raw/*.csv`, `notebooks/01_exploratory_analysis.ipynb` | `feat: virtual smart-meter data simulator`, `feat: data loader with profiling`, `docs: EDA notebook with key observations` | `df.head()` output, chart 01 + 02 + 03 |
| **3: Preprocessing** | Cleaning + feature engineering | `src/preprocessing.py`, `src/features.py`, `tests/test_pipeline.py` | `feat: cleaning with robust outlier detection`, `feat: calendar, weather and lag features`, `test: leakage and cleaning tests` | Cleaning report from the log, chart 04, `pytest` passing |
| **4: Model** | Training of 4 models | `src/train.py`, `src/evaluate.py`, `models/model_metadata.json` | `feat: baseline + 3 ML models with time-based split`, `feat: evaluation metrics` | Training-log screenshot (the leaderboard) |
| **5: Evaluation** | Compare, explain, error analysis | `outputs/model_metrics.csv`, `outputs/feature_importance.csv`, charts 05–08 | `feat: permutation importance and residual analysis`, `docs: add model leaderboard to README` | Charts 05, 06, 07, 08 |
| **6: Visualization + Simulation** | Forecast + business simulation + charts | `src/forecast.py`, `src/simulation.py`, `src/visualize.py`, `main.py`, `outputs/*`, charts 09–10 | `feat: recursive 7-day forecast with intervals`, `feat: business simulation (alerts, load shifting, procurement)`, `feat: one-command pipeline` | `business_report.txt` screenshot, charts 09, 10 |
| **7: Upload & polish** | Final README, docs, release | `README.md`, `docs/*`, `LICENSE` | `docs: complete README with results and screenshots`, `docs: project guide`, `chore: v1.0 release` | Screenshot of the GitHub repo page + LinkedIn post |

**LinkedIn post template (Day 7):**

> 🚀 Built an **AI-Powered Energy Consumption Forecasting System**.
> 📊 Forecasts hourly demand 7 days ahead with **3.4% error (R² 0.95)**, about 53% lower RMSE than a naive baseline.
> 💡 Simulated business impact: **46% lower procurement penalties**, peak-limit violations cut from 6 hours to 0 via load shifting.
> 🛠 Python · Pandas · scikit-learn · Gradient Boosting · Matplotlib · pytest
> Repo 👉 github.com/SNischayPrasad/AI-Energy-Forecasting
> #MachineLearning #DataScience #EnergyAnalytics #TimeSeries

---

## M. Screenshots, Outputs & Proof Checklist

Store screenshots you take yourself in `images/screenshots/`. The pipeline charts are already in `images/`.

| # | Asset | How to get it | File name | Where it's used |
|---|---|---|---|---|
| 1 | Dataset preview | `df.head()` in the notebook, screenshot | `images/screenshots/01_dataset_preview.png` | README → Dataset |
| 2 | Preprocessing output | Cleaning report line from the console / chart 04 | `images/screenshots/02_cleaning_report.png` + `images/04_data_cleaning.png` | README → Screenshots |
| 3 | Training logs | Run `python main.py`, screenshot the leaderboard | `images/screenshots/03_training_log.png` (+ `outputs/run_log.txt`) | README → Results |
| 4 | Prediction graph | Auto | `images/09_future_forecast.png` | README hero image |
| 5 | Actual vs predicted | Auto | `images/06_actual_vs_predicted.png` | README → Screenshots |
| 6 | Error metrics | Auto (`outputs/model_metrics.csv`) + chart | `images/05_model_comparison.png` | README → Results table |
| 7 | Feature importance | Auto | `images/07_feature_importance.png` | README / interviews |
| 8 | Business report | Screenshot of the console briefing | `images/screenshots/08_business_report.png` | README → Results |
| 9 | Tests passing | `python -m pytest -q`, screenshot | `images/screenshots/09_tests_passing.png` | README badge proof |
| 10 | GitHub repo preview | Screenshot of your repo page | `images/screenshots/10_github_repo.png` | LinkedIn / resume |

### Final proof checklist

- [ ] `python main.py` runs end to end with no errors
- [ ] `python -m pytest -q` shows **4 passed**
- [ ] 10 charts in `images/`, 9 files in `outputs/`
- [ ] README shows results table + at least 4 images
- [ ] At least 15 meaningful commits spread over 7 days
- [ ] Repo pinned on your profile, topics added
- [ ] Resume bullet added, e.g.: *"Built an ML energy-demand forecasting system (Gradient Boosting, 3.4% MAPE, R² 0.95) with a business simulation showing 46% lower procurement penalties and elimination of peak-limit violations via load shifting."*
- [ ] You can explain these points in an interview: why a time-based split, why lags ≥ 24 h, why the model beats the baseline, and what MAPE means to a manager
