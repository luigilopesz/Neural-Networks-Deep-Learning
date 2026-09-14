# Project — Will this station flood tomorrow?

One question, asked 456,600 times: given everything a river monitoring station has recorded up
to today, will it report a flood tomorrow?

## Team

- Luigi Lopes

## The dataset

Kaggle, *ASIA-FLOOD: 25-Year Flood Risk Atlas (2000–2024)* by aliahmadmphil, CC BY-SA 4.0.
Three tables:

| file | rows | what it holds |
|---|---|---|
| `station_daily_data_full.csv` | 456,600 | one row per station per day: 50 stations × 9,132 days, 2000-01-01 to 2024-12-31 |
| `vulnerability_data.csv` | 10 | one row per river basin: area, population, poverty rate, hospitals, resilience scores |
| `asia_flood_geospatial.geojson` | 50 | station coordinates and basin polygons |

The daily table has 4 measured variables (`rainfall_mm`, `river_level_m`, `soil_moisture_percent`,
`temperature_celsius`), calendar fields (`date`, `season`, `monsoon_season`, `day_of_year`),
identifiers (`station_id`, `basin`, `country`, `latitude`, `longitude`), a `data_quality_score`,
and three outcome columns: `flood_event_occurred` (0/1), `severity_level`, `flood_risk_score`.

The data is kept out of git (`docs/projects/data/`, gitignored) and out of the built site. The
scripts download or expect it there.

!!! warning "This dataset is synthetic"
    Its README says it was *generated* (2026-07-23). Some station names do not match their
    coordinates (a "Luang Prabang" at 16.1°N 103.0°E, which is Thailand, not Laos), and, as shown
    below, floods occur *only* during the monsoon flag. Every method here is real; the numbers
    describe a simulator, not Asia. Nothing in this project is a claim about actual flood risk.

## What a first pass over the data says

These numbers shape every decision that follows. They come from
`station_daily_data_full.csv` before any modelling.

- The panel is complete: every station has all 9,132 days, no duplicates, no missing values.
- **2.17 % of station-days are floods** (9,889 of 456,600). This is an imbalanced problem;
  accuracy is meaningless (predicting "no flood" scores 97.8 %).
- **Two columns are the target in disguise.** `severity_level == "Extreme"` coincides with
  `flood_event_occurred == 1` on 9,889 of 9,921 rows, and `flood_risk_score > 7` separates the
  same-day event perfectly (ROC-AUC 1.000). They are outputs of whatever generated the floods,
  not measurements. They are excluded from the features. `flood_risk_score` appears once, as a
  reference baseline, labelled as such.
- **Floods happen only in monsoon season.** The flood rate is 5.17 % when `monsoon_season = 1`
  and exactly 0 % otherwise. A rule that says "never outside the monsoon" is therefore a
  baseline every model has to beat *inside* the monsoon, where the actual problem lives.
- **Floods are isolated days, not episodes.** P(flood tomorrow | flood today) = 7.6 %, against
  a 2.05 % base rate. Yesterday's label helps a little; it will not carry a model.
- **Today's measurements carry real signal for tomorrow.** Used alone, each gives ROC-AUC for
  the next-day event of: soil moisture 0.836, rainfall 0.809, temperature 0.796, river level
  0.795. The modelling bet is that windows over the past days (accumulated rain, rising river)
  do better than any single day.
- Events per year are stable (354–439), so a chronological split is fair to both ends.

## The task, precisely

For station \(s\) and day \(t\), predict \(y_{s,t+1} = \) `flood_event_occurred` on day \(t+1\),
using only rows of station \(s\) with date \(\le t\) plus static basin information. The
prediction is a probability; the operating threshold is chosen on validation data for a stated
recall target, because a flood warning system is judged on the floods it misses.

## Rules that stop leakage

1. **No same-day outcome columns as inputs.** `flood_event_occurred`, `severity_level` and
   `flood_risk_score` for day \(t\) never enter the feature vector for \(y_{t+1}\).
   (`flood_event_occurred` for days \(\le t\) may enter as an explicit lag feature; it is what a
   real system would know.)
2. **Every feature is computed from the past only.** Rolling windows end at day \(t\). Shifts
   are checked by construction: a test asserts that permuting the future leaves every feature
   unchanged.
3. **Chronological split, by date, same dates for every station:**
   train 2000–2016 (17 years), validation 2017–2020 (4), test 2021–2024 (4). No random
   shuffling: rows one day apart are near-duplicates, and a random split would leak.
4. **Every statistic is fitted on train only**: scalers, per-station means and standard
   deviations, category vocabularies, the decision threshold.
5. The test split is touched once, at the end, with the model and threshold frozen.

## Phase 1 — EDA (`projects/eda/`)

Deliverables, each with a figure and the numbers in the text:

1. Panel completeness and the class balance above, per station and per year.
2. Distributions of the four measurements, raw and log-transformed where skewed (rainfall).
3. Seasonality: flood rate by month and by `day_of_year`; the monsoon gate, shown plainly.
4. What a flood looks like: the mean trajectory of each measurement over the 14 days before and
   after a flood, against the station's normal. This picture decides the window lengths.
5. Between-station differences: flood rate, mean river level and rainfall per station and basin;
   whether the vulnerability table explains any of it.
6. Correlations among measurements and with the next-day target.
7. The leakage audit: the two outcome-derived columns, with the numbers that prove it.

## Phase 2 — Feature engineering and normalisation

Built by one function that takes a station's day-ordered frame and returns features aligned
to \(t\), so it is the same code for train, validation, test and any future day.

| group | features | why |
|---|---|---|
| today | the 4 measurements at \(t\) | the strongest single-day signal (AUC ≈ 0.8 each) |
| lags | each measurement at \(t-1, t-2, t-3\) | the direction things are moving |
| windows | rainfall sum over 3, 7, 14, 30 days; river level mean and max over 3, 7, 14 days; soil moisture mean over 3, 7 days | floods follow accumulated rain and a rising river, not one wet day |
| changes | day-over-day change of river level and soil moisture; 7-day slope of river level | rising versus falling |
| anomalies | each measurement as a z-score against *that station's* training mean and sd | "unusually high for here" is what matters, and it is what a station operator would say |
| target history | flood in the last 1, 7, 30 days (counts) | weak, but honest and cheap |
| calendar | \(\sin, \cos\) of day-of-year; `monsoon_season` | seasonality without a discontinuity at New Year |
| place | basin one-hot; the basin's `floodplain_percent`, `population_density_km2`, `vulnerability_score` | static context; 10 basins, so kept small |

Normalisation: `log1p` on rainfall and the rainfall windows (heavy right tail), then
standardisation of every numeric feature with train-split mean and sd. The station anomalies
are already standardised by construction. One-hot columns are left as 0/1.

Feature count target: about 40. Every feature has a one-line justification in the report, and
the ablation in Phase 3 removes each group in turn to show what it buys.

## Phase 3 — Classification (`projects/classification/`)

The model is the from-scratch MLP from the MLP exercise (`docs/exercises/mlp/code/mlp.py`),
reused unchanged: layer sizes as an argument, sigmoid output, binary cross-entropy, mini-batch
gradient descent. Additions the imbalance needs, implemented by hand in the same style:
class-weighted cross-entropy (or oversampling of the positive class in each mini-batch) and
early stopping on validation PR-AUC.

Baselines, in this order, each reported on validation and test:

1. Majority class ("never floods") — the accuracy trap, shown once.
2. Monsoon gate ("floods whenever `monsoon_season = 1`") — recall 100 %, precision 5 %.
3. Persistence ("floods tomorrow if it flooded today").
4. Logistic regression on the same features — the linear reference.
5. The dataset's own `flood_risk_score` at \(t\) — the leaky reference, labelled as such.
6. The MLP.

Metrics: ROC-AUC, PR-AUC (the headline), precision and recall at the chosen threshold, the
confusion matrix, and a calibration curve. The threshold is set on validation for 80 % recall,
and the report states the precision that buys. Training and validation loss curves per epoch.

Explainability, so every prediction can be argued with:

- ablation by feature group (drop one group, retrain, report the PR-AUC change);
- permutation importance on validation;
- partial-dependence curves for the top five features (rainfall 7-day sum, river-level anomaly,
  soil moisture, ...), which is where the report says *what* the model learned;
- error analysis: which stations, which months, and what the missed floods looked like the day
  before;
- one worked example: a single station-day, its feature vector, and the probability, traced.

## Phase 4 — Generative (`projects/generative/`)

Scoped after Phase 3. The candidate: a variational autoencoder over 14-day windows of the four
measurements, to synthesise plausible pre-flood sequences and to score anomalies, evaluated
against the monsoon-season distribution of real windows.

## Definition of done, per phase

A phase is done when its page has every required section from the course statement, every
number in the text matches its script's output, every figure has a title, axis labels and a
legend, the scripts run from a clean checkout with the dataset in place, and the site builds.
