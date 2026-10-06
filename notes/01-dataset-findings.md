# ASIA-FLOOD 25-Year Dataset — Exploration Findings

Internal working note for the EDA deliverable. Not part of the published site
(`mkdocs.yml` builds `docs/` only; this file lives in `notes/`).

- Dataset root: `docs/projects/data/ASIA_FLOOD_25YEAR_DATASET/` (gitignored, local only)
- Task: **binary classification — will station *s* flood on day *t+1*?**
- Every number below was computed with `pandas 3.0.5 / numpy 2.5.2 / scikit-learn 1.9.0`
  under Python 3.14 in `.venv`, `random_state=42`.

---

## 1. What is actually on disk

| File | Rows × Cols | Size | Role |
|---|---|---|---|
| `data/station_daily_data_full.csv` | 456,600 × 21 | 57.7 MB | **the dataset to use** — the complete panel |
| `data/station_daily_data.csv` | 100,000 × 21 | 12.6 MB | a uniform ~21.9 % random sample of the full file |
| `data/vulnerability_data.csv` | 10 × 18 | 1.2 KB | one row per river basin (static context) |
| `data/asia_flood_geospatial.geojson` | 10 features | 34 KB | **basin polygons** with risk/vulnerability labels |
| `documentation/metadata.json` | — | — | declares 456,600 records, 50 stations, 10 basins |

**Verified:** all 100,000 sampled `(station_id, date)` pairs exist in the full file;
the sample keeps all 50 stations and the whole 2000-01-01 → 2024-12-31 range; per-station
counts range 1,850–2,084 (expected ≈2,000). It is a genuine random subset, so **the full
file must be used** — the sample loses ~78 % of the rows and, more importantly, breaks the
daily time axis that lag features need.

> ⚠️ `docs/projects/index.md` says the GeoJSON holds "50 station coordinates". It does not —
> it holds **10 basin `Polygon` features** with properties `basin, country, area_km2,
> population_millions, flood_risk_level, vulnerability_level`. Station coordinates come from
> the CSV (`latitude`/`longitude`, constant per station). Correct this in the report.

**Column order and dtype are identical** between the two CSVs (verified element-wise). Both
are sorted `(station_id, date)` — station-major, and every station's block is date-monotonic.

---

## 2. Panel integrity — the dataset is *clean*

| Check | Result |
|---|---|
| Rows per station | **9,132 for every one of the 50 stations** (single unique value) |
| Expected days 2000-01-01 → 2024-12-31 | 9,132 — **exact match**, no gaps |
| Duplicated `(station_id, date)` | 0 |
| Fully duplicated rows | 0 |
| Missing values, all 21 columns | **0 (0.00 %)** |
| Duplicated `(station, rainfall, river, soil, temp)` | 0 |

This is a **balanced panel**: 50 stations × 9,132 days = 456,600 rows exactly.

**Consequence for the EDA:** stage 1B (missing values) and stage 4A (imputation strategy)
have nothing to impute. That is itself the finding — but it means the "missing values"
row of the results table reads `0 (0.00 %)` and the imputation strategy must be argued as a
*robustness* measure (e.g. `SimpleImputer` in the pipeline so a future partial day does not
crash it), not as a fix for an observed problem. Say that explicitly; do not invent missingness.

### Data source: this dataset is synthetic

`README.md` states "Generated on: 2026-07-23", the metadata declares version 3.0, and the
internal structure below is machine-regular. Treat every finding as a property of a
**simulator**, not of Asian hydrology. State this in the report's data dictionary.

Two concrete tells:

- **Station coordinates are wrong for their names.** "Luang Prabang" sits at 16.15 °N
  103.02 °E (Thailand, not Laos); the Mekong group is labelled `country = Vietnam`; the
  five "Helmand" stations are spread over 60.5–65.1 °E. Coordinates are unique per station
  but do not match the place names.
- **`latitude`/`longitude` are perfectly constant within a station** (1 unique value each),
  as are `station_name` and `basin`. `(station_name, latitude, longitude)` has 456,550
  duplicates — i.e. the geographic columns are **static per-station attributes repeated on
  every row**, not measurements.

---

## 3. Column taxonomy — 21 columns, and three of them are the label

### Identifiers / static (drop or use as grouping)
| Column | Type | Notes |
|---|---|---|
| `station_id` | str, 50 levels | `ASIA_0001`…`ASIA_0050`, perfectly balanced (9,132 each) |
| `station_name` | str, 50 levels | 1:1 with `station_id` — **redundant, drop one** |
| `basin` | str, 10 levels | 5 stations each, 45,660 rows each |
| `country` | str, 7 levels | `basin → country` is a **function** (verified: 1 unique country per basin) — **redundant with `basin`** |
| `latitude`, `longitude` | float | 1 unique value per station — constants within station |

### Calendar
| Column | Type | Notes |
|---|---|---|
| `date` | str → datetime | 9,132 unique days |
| `year` | int | 2000–2024, 25 levels |
| `month` | int | 1–12 |
| `day` | int | 1–31 |
| `day_of_year` | int | 1–366 |
| `season` | str, 4 levels | **deterministic from `month`**: Mar-May Spring, Jun-Aug Summer, Sep-Nov Autumn, Dec-Feb Winter |
| `monsoon_season` | int 0/1 | **deterministic from `month`**: 1 for Jun–Oct, 0 otherwise |

Verified: `date` vs `(year, month, day)` → **0 mismatches**; `day_of_year` → **0 mismatches**.
So `year/month/day/day_of_year/season/monsoon_season` are six deterministic re-encodings of
`date`. **`day` is pure noise** (r = 0.0006 with the next-day target) — drop it.
`season` and `monsoon_season` are redundant with `month` but useful as an explicit gate.

### Measurements (the only genuine sensor columns)
| Column | Unit | min | median | max | skew | kurtosis | IQR outliers |
|---|---|---|---|---|---|---|---|
| `rainfall_mm` | mm/day | 2.02 | 10.76 | **1,775.72** | **10.51** | **158.20** | 9,274 (2.03 %) |
| `river_level_m` | m | 0.66 | 4.42 | **204.67** | **9.53** | **140.61** | 9,273 (2.03 %) |
| `soil_moisture_percent` | % | 0.39 | 31.00 | 100.00 (censored) | 0.86 | 0.86 | 9,180 (2.01 %) |
| `temperature_celsius` | °C | −2.31 | 24.49 | 45.00 | **0.006** | −1.20 | 0 |

- `rainfall_mm` and `river_level_m` are **severely right-skewed** (skew ≈ 10, 140–158 kurtosis).
  `log1p` is the obvious transform.
- `soil_moisture_percent` is **censored at exactly 100.00** — a ceiling artefact, not a measurement.
- `temperature_celsius` has skew ≈ 0 and kurtosis −1.20: **flat/bimodal**, the signature of a
  smooth seasonal sinusoid rather than weather noise.
- The 1 % and 99 % quantiles of `latitude`/`longitude` equal the min/max — a direct
  consequence of only 50 distinct coordinate pairs.

### Engineered / outcome-derived — **the critical part**
| Column | Type | Status |
|---|---|---|
| `flood_risk_score` | float, 899 unique on a **0.01 grid**, range [1.00, 10.00] | **derived index → leakage for the same-day label** |
| `flood_event_occurred` | int 0/1 | **the label** |
| `severity_level` | str, 4 levels | **derived from `flood_risk_score` → leakage** |
| `data_quality_score` | float, 111 unique, [0.880, 0.990] | noise — see below |

---

## 4. 🔴 The headline finding: the label is an exact threshold on a leaked column

I verified both rules over **all 456,600 rows**:

```
flood_event_occurred == (flood_risk_score > 7.0)          → True for every row
severity_level       == cut(flood_risk_score,
                            bins=[1, 3, 5, 7, 10],
                            right=False)                   → Low / Moderate / High / Extreme
```

The severity bins are **left-closed, right-open** — `[1,3) Low`, `[3,5) Moderate`, `[5,7) High`,
`[7,10] Extreme` — verified forward and inverse on all 456,600 rows.

Evidence:

| `severity_level` | `flood_risk_score` min | max | n | floods |
|---|---|---|---|---|
| Low | 1.00 | 2.99 | 234,226 | 0 |
| Moderate | 3.00 | 4.99 | 129,974 | 0 |
| High | 5.00 | 6.99 | 82,479 | 0 |
| Extreme | 7.00 | 10.00 | 9,921 | 9,889 |

- `max(risk | no flood) = 7.00`, `min(risk | flood) = 7.01` — **the classes are linearly
  separable on `flood_risk_score` alone**, ROC-AUC = **1.0000** for the same-day label.
- `flood_risk_score` is itself **~deterministic from the four measurements**: a gradient-boosted
  regressor on `[rainfall, river_level, soil_moisture, temperature, day_of_year, lat, lon, year]`
  reaches **R² = 0.9426** (mean |residual| 0.32, max 1.90 on a 0–10 scale). It is a
  synthesised composite index, not an observation.
- `severity_level` is *also* an internal inconsistency: exactly **32 rows have
  `risk == 7.00`**, which the severity bin `[7,10]` calls `Extreme` but the flood rule
  (`> 7.0` strictly) calls *no flood*. Flag these 32 rows as the dataset's one genuine
  quality defect.

**Decision (already recorded in `docs/projects/index.md`):** `severity_level` is dropped
outright; `flood_risk_score` is dropped from the feature set and shown **once**, explicitly
labelled, as a leaky reference baseline.

> **Nuance the report must state honestly.** For a *same-day* target these are leakage. For
> the *one-day-ahead* target, `flood_risk_score(t)` is not strictly leakage — it is a value
> observed at time *t*. It is excluded anyway because it is **redundant, not informative**:
> it correlates 0.934 with `soil_moisture_percent`, 0.664 with `river_level_m`, 0.595 with
> `rainfall_mm`, and as a single predictor of the *next-day* label it scores **ROC-AUC 0.8088**
> — *below* `soil_moisture_percent` (0.8364) and barely above `rainfall_mm` (0.8091). Keeping
> it would add a generator artefact and buy nothing.

---

## 5. The label, and the one-day-ahead reframing

| | Same-day `flood_event_occurred` | Next-day target `flood_event_occurred(t+1)` |
|---|---|---|
| Positives | 9,889 | 9,889 |
| Negatives | 446,711 | 446,661 |
| Positive rate | **2.1658 %** | **2.1660 %** |
| Imbalance | **45.2 : 1** | 45.2 : 1 |
| Usable rows | 456,600 | **456,550** (last day of each station has no *t+1*) |

Accuracy of the majority class is **97.83 %** — state it once, then never use accuracy again.
The headline metric is **PR-AUC** with ROC-AUC alongside.

### The target is gated by season — floods do not exist outside Jul–Oct

| Month | n | floods | rate | mean rainfall | mean river level | mean soil moisture |
|---|---|---|---|---|---|---|
| Jan–Apr, Nov, Dec | 225,600 | **0** | **0.000 %** | ~7.1 mm | ~3.1–3.9 m | ~18–30 % |
| May | 38,750 | 1 | 0.003 % | 7.07 | 4.18 | 29.94 |
| Jun | 37,500 | 1 | 0.003 % | 19.95 | 6.32 | 44.98 |
| **Jul** | 38,750 | 3,128 | **8.072 %** | 39.43 | 8.47 | 55.90 |
| **Aug** | 38,750 | 3,141 | **8.106 %** | 45.56 | 9.36 | 59.78 |
| **Sep** | 37,500 | 3,130 | **8.347 %** | 52.68 | 10.25 | 60.78 |
| Oct | 38,750 | 488 | 1.259 % | 35.11 | 7.95 | 55.55 |

- **95.0 % of all floods fall in Jul/Aug/Sep** (31.63 %, 31.76 %, 31.65 % — near-perfectly equal),
  a further 4.9 % in October, and **2 floods in the entire rest of the year**.
- By the monsoon flag: **5.170 %** flood rate when `monsoon_season = 1` (9,888 of 191,250)
  vs **0.00038 %** when 0 (**1** of 265,350).
- A "never flood outside the monsoon" rule is therefore a free baseline. Every model must be
  judged **inside the monsoon window**, where the real problem is a 5.2 % base rate, not 2.2 %.

### Floods are isolated days, not episodes

- 9,135 flood episodes across the 50 stations: **8,448 are single days**, 624 last 2 days,
  59 last 3, 4 last 4. Mean length **1.083 days**, median 1.
- `P(flood tomorrow | flood today) = 7.62 %` vs a `2.05 %` base rate — a **3.7× lift**, but
  persistence still misses 92.4 % of next-day floods. Persistence alone is a weak baseline.
- Daily autocorrelation within station: `temperature` **0.968**, `soil_moisture` **0.820**,
  `flood_risk_score` 0.771, `river_level` 0.222, **`rainfall` only 0.164**.

---

## 6. Univariate signal for the **next-day** target

Single-feature ROC-AUC against `flood_event_occurred(t+1)`, over the 456,550 usable rows:

| Feature | ROC-AUC (next-day) | ROC-AUC (same-day, reference) |
|---|---|---|
| `soil_moisture_percent` | **0.8364** | 0.9969 |
| `rainfall_mm` | **0.8091** | 0.9991 |
| `flood_risk_score` *(excluded — leaky/redundant)* | 0.8088 | **1.0000** |
| `temperature_celsius` | 0.7956 | 0.7991 |
| `river_level_m` | 0.7948 | 0.9994 |
| `day_of_year` | 0.6313 | — |
| `longitude` | 0.5103 | — |
| `data_quality_score` | 0.5072 | — |
| `latitude` | 0.5203 (flipped) | — |

**Reading:** day *t*'s four measurements carry real, substantial signal for day *t+1*
(AUC 0.79–0.84 each) — the task is learnable. `latitude`, `longitude` and
`data_quality_score` are **noise** (AUC ≈ 0.50). `day_of_year` at 0.63 is pure seasonality.

Point-biserial correlation with the next-day target ranks them:
`soil_moisture` 0.177 > `monsoon_season` 0.175 > `flood_risk_score` 0.162 >
`temperature` 0.150 > `river_level` 0.095 > `rainfall` 0.085 > `flood_event_occurred(t)` 0.056.
Note the **inversion vs. the same-day picture**: today's soil moisture and today's *temperature*
(0.150) matter more for tomorrow than today's rainfall (0.085). Rain is a same-day trigger;
soil moisture is the standing precondition that persists.

### End-to-end sanity ceiling (not a deliverable — a signal check)

A gradient-boosted classifier on the ten numeric features, chronological split
(train 2000–2019 / test 2020–2024, n_test = 91,300, class-weighted):

```
ROC-AUC = 0.8701     PR-AUC = 0.0844     test base rate = 0.0231
```

**PR-AUC 0.0844 is 3.65× the base rate.** That is the honest ceiling a simple feature set
reaches; the project's window/lag/station-anomaly features (Phase 2 of the plan) must beat it.
Report these as motivation, not as EDA results — **no model is trained in this deliverable**.

---

## 7. Redundancy — the correlation structure

### The four measurements (n = 456,600)

Pearson above the diagonal, Spearman below:

| | rainfall | river level | soil moisture | temperature |
|---|---|---|---|---|
| **rainfall_mm** | — | **0.9604** | 0.6103 | 0.3082 |
| **river_level_m** | 0.8688 | — | 0.6652 | 0.3427 |
| **soil_moisture_percent** | 0.8455 | 0.8617 | — | **0.7197** |
| **temperature_celsius** | 0.6688 | 0.6188 | 0.7500 | — |

- **`rainfall_mm` ↔ `river_level_m` at Pearson r = 0.9604 is the redundant pair.** The river
  level is essentially a rescaled rainfall. Report it as the answer to results-table row 7.
  Spearman is much lower (0.8688) — the relationship is linear-with-heavy-tails, so **Pearson
  is inflated by the shared extreme tail** while Spearman reflects the rank agreement.
- **Which to use: Spearman.** All four variables are strongly right-skewed
  (`rainfall` skew 10.5, `river_level` skew 9.5), so Pearson is dominated by a handful of
  extreme monsoon days and understates the monotone association for the bulk of the data.
  Spearman is the honest choice; report Pearson once, side by side, to show the gap.
- Do **not** drop either of the pair on correlation alone — they diverge for the next-day
  target (`soil_moisture` 0.836 vs `rainfall` 0.809 vs `river_level` 0.795 AUC) and a river
  level lags rain physically. Keep both and let the ablation decide.
- `temperature` ↔ `soil_moisture` at 0.72 is a **seasonal** artefact: both track
  `day_of_year` (r = 0.721 and 0.455 respectively). It is confounding, not mechanism.

### With the engineered columns (Pearson, full matrix)

`flood_risk_score` ↔ `soil_moisture_percent` = **0.934** ← this is why the risk score is
redundant rather than informative. `month` ↔ `day_of_year` = **0.997** (drop one).
`latitude` ↔ `longitude` = **−0.494** (geography, only 50 points).
`year` correlates with nothing (|r| ≤ 0.037) — **no long-term drift**, which is what makes a
chronological split fair.
`data_quality_score` correlates with **everything at |r| ≤ 0.004** — pure noise, **drop it**.

---

## 8. Group structure — stations and basins barely differ

| `basin` | country | stations | floods | rate |
|---|---|---|---|---|
| Chao Phraya | Thailand | 5 | 1,128 | 2.470 % |
| Mekong | Vietnam | 5 | 1,093 | 2.394 % |
| Pearl | China | 5 | 1,020 | 2.234 % |
| Irrawaddy | Myanmar | 5 | 1,013 | 2.219 % |
| Salween | Myanmar | 5 | 994 | 2.177 % |
| Ganges | India | 5 | 979 | 2.144 % |
| Indus | Pakistan | 5 | 929 | 2.035 % |
| Helmand | Afghanistan | 5 | 925 | 2.026 % |
| Yangtze | China | 5 | 919 | 2.013 % |
| Brahmaputra | India | 5 | 889 | 1.947 % |

- **Basin flood rates span only 1.95 %–2.47 %**, a 1.27× ratio. Station rates span
  1.63 % (Pandu) – 2.58 % (Bangkok), a 1.58× ratio. **Geography barely matters.**
- Floods per year are **stable at 354–439** across all 25 years (mean ≈ 393) — no trend, no
  regime change. Confirms the chronological split is not a distribution shift trap.
- **Variance decomposition of rainfall:** between-station variance is **0.0423 %** of the
  total (between 0.743 vs within 1,756.9). Every station has the same climate. Station-level
  fixed effects are worthless here — which means the planned *per-station z-score anomalies*
  will mostly encode **season**, not local character. Say so; do not oversell them.
- Within-station **demeaned** rainfall autocorrelation is still ~0.164 at lag 1 and does not
  decay (lag 7 = 0.161, lag 30 = 0.126) — that is **seasonality**, not day-to-day persistence.
- `vulnerability_data.csv` joins perfectly: the 10 basins match exactly, `basin → country`
  is consistent in both tables. **But with basin rates spanning 1.95–2.47 % against a 2.17 %
  mean, the vulnerability table has almost no signal to explain.** Keep it as static context
  (the plan does), report the null result honestly.

---

## 9. Outliers: the "outliers" *are* the floods

| Fence (1.5 × IQR) | Rows flagged | % | Flood rate among flagged | Flood rate among unflagged |
|---|---|---|---|---|
| `rainfall_mm` > 56.30 mm | 9,274 | 2.03 % | **100.000 %** | 0.137 % |
| `river_level_m` > 12.23 m | 9,273 | 2.03 % | — | — |
| either | 9,280 | 2.03 % | — | — |

**This is the single most important preprocessing decision in the deliverable.** A textbook
"IQR-filter the outliers" step would **delete every flood in the dataset**. The extreme
rainfall values are not measurement error; they *are* the positive class.

Recommended strategy (state it as a decision with this table behind it):
1. **Never drop or clip them.** No row filtering, no winsorising.
2. **`log1p`-compress** `rainfall_mm` and `river_level_m` so the right tail cannot dominate a
   neural network's gradients, while keeping the extreme days present and ordered.
3. Flag them with an explicit **`is_extreme_rainfall`** indicator if the model needs the
   magnitude back.
4. Keep `soil_moisture_percent`'s **100.0 ceiling** as-is; note it is censored, so any
   transform must preserve the tie at the boundary.

---

## 10. Train / validation / test split

Chronological, by date — **the same dates for every station**. Random splitting is invalid
here: rows one day apart are near-duplicates and the label is 45:1 imbalanced, so a random
split leaks flood days across the boundary.

| Split | Years | Rows (window) | Rows (next-day usable) | Floods | Rate (next-day) | Dates |
|---|---|---|---|---|---|---|
| **train** | 2000–2016 | 310,500 | **310,500** | 6,566 | 2.1147 % | 2000-01-01 → 2016-12-31 |
| **validation** | 2017–2020 | 73,050 | **73,050** | 1,620 | 2.2177 % | 2017-01-01 → 2020-12-31 |
| **test** | 2021–2024 | 73,050 | **73,000** | 1,703 | 2.3329 % | 2021-01-01 → 2024-12-31 |

⚠️ **The test split loses 50 rows.** `add_next_day_target` drops the last day of each station
(2024-12-31), and all 50 of those rows fall inside the test window. So the test set is
**73,050 rows in the year window but 73,000 usable rows**. Train and validation are unaffected.
310,500 + 73,050 + 73,000 = **456,550**. Quote the usable counts in the report and explain the
50-row difference; do not let "73,050" appear as the test *set* size.

Next-day target rates are 2.115 % / 2.218 % / 2.333 % — **stable across the three splits**,
so no split is starved of positives. The mild upward drift (2.11 → 2.33 %) is visible in the
per-year counts and is the reason the test rate is quoted: the model will be scored on a
slightly more flood-prone era than it trained on.

All 50 stations appear in all three splits. **Every preprocessing statistic — scalers,
station means/sds, category vocabularies, the decision threshold — is fitted on train only.**

---

## 11. Risks for the modelling phase

| # | Risk | Evidence | Handling |
|---|---|---|---|
| R1 | **Two columns give the answer away** | `risk > 7.0` ⇒ flood, ROC-AUC 1.0000 | Drop `severity_level` and `flood_risk_score`; show the risk score once as a labelled leaky baseline |
| R2 | **Extreme values are the positive class** | 100 % of rows above the rainfall IQR fence are floods | No row removal, no winsorising; `log1p` + an explicit extreme flag |
| R3 | **45:1 imbalance** | 2.166 % positives; majority class = 97.83 % accuracy | PR-AUC as headline; class weights / positive oversampling; threshold chosen on validation for a stated recall |
| R4 | **Floods exist only in Jul–Oct** | 2 floods outside the window, 95 % in Jul–Sep | Always report metrics **inside the monsoon**; never quote a global AUC as if it were skill |
| R5 | **Only ~1 % of days have any local persistence** | 8,448 of 9,135 episodes are 1 day; P(flood\|flood) = 7.6 % | Lean on accumulated-rain / rising-river windows, not on the lagged label |
| R6 | **Rainfall is nearly white noise daily** | lag-1 autocorrelation 0.164 | Daily rainfall alone will not carry a window model; use multi-day sums |
| R7 | **Geography and vulnerability are ~useless** | basin rates 1.95–2.47 %; between-station rainfall variance 0.04 % | Keep as small static context, report the null result, do not build the model on them |
| R8 | **`data_quality_score` is noise** | \|r\| ≤ 0.004 with every other numeric column; next-day AUC 0.5072 | Drop the column; document the check |
| R9 | **The dataset is synthetic and self-inconsistent** | 32 rows with `risk == 7.00` are `Extreme` but not floods; station names contradict coordinates | Disclose it in the data dictionary; never frame results as real flood risk |
| R10 | **`latitude`/`longitude` are constants per station** | 1 unique value each; 456,550 duplicate triples | Treat as station identifiers, not features; if geography is wanted, use `basin` |
| R11 | **Six calendar columns encode one date** | 0 mismatches on all consistency checks; `month`↔`day_of_year` r = 0.997 | Keep `sin/cos(day_of_year)` + `monsoon_season`; drop `day`, `year`, `month`, `day_of_year` as raw integers |

---

## 12. Environment notes (will bite the subagents)

- The Python environment to use is **`.venv` at the workspace root**
  (`C:\Users\luigi\projects\Artificial-Neural-Networks-and-Deep-Learning\.venv`), Python 3.14.
  It has pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, matplotlib 3.11.1, seaborn 0.13.2.
  The repo-local `Neural-Networks-Deep-Learning/env/` is a separate Anaconda 3.13 tree —
  **do not use it.**
- **`umap-learn` was missing** (required by stage 4B) and has been installed into `.venv`
  (`umap-learn 0.5.12`, with `numba 0.68.0` / `llvmlite 0.50.0` / `pynndescent 0.6.0`).
- The `.venv` was created by **uv and has no `pip`**. Install with
  `uv pip install --python .venv\Scripts\python.exe <pkg>`. `uv`'s cache lives outside the
  workspace and is sandbox-blocked — set `$env:UV_CACHE_DIR` to a workspace path first.
- **The sandbox forbids named pipes.** Any library that builds a joblib `Parallel` pool —
  notably `HistGradientBoosting*` and anything with `n_jobs>1` — **crashes with
  `PermissionError: [WinError 5]`**. Set `OMP_NUM_THREADS=1`, `LOKY_MAX_CPU_COUNT=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` in every script. UMAP with `n_jobs=1`
  is fine. *The EDA trains no models, so this should rarely bite — but it will if a script
  reaches for `HistGradientBoosting` for a quick check.*
- `pandas 3.0` is a **major** version: `df.groupby(...).apply` on a Series, `.append`,
  and in-place `inplace=` patterns differ from 2.x. Write to the 3.0 API.
- Reading the full CSV takes a few seconds and ~231 MB in memory. **Load it once per script**
  and cache; do not re-read it inside a loop.
