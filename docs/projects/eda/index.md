---
project: eda
task: classification
dataset: https://www.kaggle.com/datasets/aliahmadmphil/asia-flood-25-year-flood-risk-atlas
team:
  - Luigi Lopes
ai_use: "AI-assisted: analysis scripts and report drafts produced with an AI coding agent; every number re-derived from the script output and reviewed by the team"
---

# 1. EDA — Will this station flood tomorrow?

**Project:** ASIA-FLOOD 25-Year Flood Risk Atlas · **Task:** binary classification ·
**Target:** `flood_event_occurred` on day *t+1*, for the same station

This is the first deliverable of the classification project: an exploratory analysis of the
dataset that will be used until the end of the semester, ending in a preprocessing pipeline
ready for modelling. **No model is trained here.**

The question is asked at 456,550 station-days: given everything a river monitoring station has recorded
up to and including day *t*, will it report a flood on day *t+1*?

!!! warning "This dataset is synthetic"
    The dataset's own README says it was *generated* (2026-07-23). Its internal structure is
    machine-regular, some station names contradict their coordinates, and — as stage 3 shows —
    floods occur *only* inside the monsoon flag. Every method in this report is a real method
    applied honestly; the numbers describe a simulator, not Asia. **Nothing here is a claim
    about actual flood risk.**

## 0. Proposal

| | |
|---|---|
| **Dataset** | ASIA-FLOOD: 25-Year Flood Risk Atlas (2000–2024) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/aliahmadmphil/asia-flood-25-year-flood-risk-atlas), by aliahmadmphil |
| **Licence** | CC BY-SA 4.0 |
| **Size** | 456,600 rows × 21 columns (`station_daily_data_full.csv`) |
| **Task** | Classification — will station *s* flood on day *t+1*? |
| **Target** | `flood_event_occurred`, shifted one day forward within each station |
| **Features** | raw: **14 numerical / 6 categorical** + `date` · modelled: 6 numerical + 2 categorical → **21** after the pipeline |
| **First risk** | two columns (`flood_risk_score`, `severity_level`) are the target in disguise — stage 1B quantifies it |

The file carries 25 years of daily records for 50 river-monitoring stations across 10 Asian
basins and 7 countries: rainfall, river level, soil moisture and temperature, plus static basin
context and three outcome columns. A one-day-ahead flood warning is the natural task: it is the
horizon at which a warning is still actionable, and the dataset is a daily panel built for it.

*Figures are numbered by the item they belong to — **Figure 3B.2** is the second figure of item
3B — so each item's figure budget is visible in the numbering itself. Every figure carries a
title, labelled axes with units and a legend, and is followed by one explicit conclusion.*


## 1. Initial inspection

Stage 1 fixes the vocabulary of the deliverable: what is in the file, what is wrong with it, what
exactly is being predicted, and which rows may be used to fit anything. Every number below is
printed by [`t1_inspection.py`](code/t1_inspection.py), which re-derives it from
`data/station_daily_data_full.csv` on every run.

### A. Data dictionary

**Source.** *ASIA-FLOOD: 25-Year Dataset* (Kaggle: "ASIA-FLOOD 25-Year Flood Risk Atlas"), author
`aliahmadmphil`, licence **CC BY-SA 4.0** — "free for academic and commercial use".
`documentation/metadata.json` declares version **3.0**, time period 2000–2024, **456,600** records,
**50** stations, **10** basins; the script reads and prints that file rather than hard-coding it.

!!! warning "The dataset is synthetic — no result here is a claim about real Asian hydrology"
    `README.md` states **"Generated on: 2026-07-23 21:27:42"**, and the internal structure is
    machine-regular to the point of self-contradiction: the label is an exact threshold on
    another column (Section B), station names contradict their coordinates, and one column is
    uncorrelated noise. Treat every finding as a property of a **simulator**, not of a
    catchment. The value of the deliverable is methodological, not hydrological.

**What a row is.** One **monitoring station on one calendar day**. The file is a *balanced daily
panel*: **456,600 rows × 21 columns** = 50 stations × 9,132 days, 2000-01-01 → 2024-12-31, with
no station missing a day.

| column | meaning | dtype | unit | role |
|---|---|---|---|---|
| `station_id` | Monitoring-station key, `ASIA_0001`…`ASIA_0050` | `str` | – | identifier |
| `station_name` | Place name of the station; 1:1 with `station_id` | `str` | – | identifier |
| `basin` | River basin the station belongs to (10 levels, 5 stations each) | `str` | – | static-per-station |
| `country` | Country label attached to the basin (7 levels) | `str` | – | static-per-station |
| `latitude` | Station latitude, repeated on all of the station's rows | `float64` | decimal degrees | static-per-station |
| `longitude` | Station longitude, repeated on all of the station's rows | `float64` | decimal degrees | static-per-station |
| `date` | Calendar day of the observation | `datetime64[us]` | – | calendar |
| `year` | Calendar year, 2000–2024 | `int64` | year | calendar |
| `month` | Calendar month, 1–12 | `int64` | month | calendar |
| `day` | Day of month, 1–31 | `int64` | day | calendar |
| `day_of_year` | Ordinal day inside the year, 1–366 | `int64` | day-of-year | calendar |
| `rainfall_mm` | Daily rainfall accumulation at the station (2.02 … 1,775.72) | `float64` | **mm/day** | measurement |
| `river_level_m` | River stage at the station gauge (0.66 … 204.67) | `float64` | **m** | measurement |
| `soil_moisture_percent` | Volumetric soil moisture, censored at 100.00 (0.39 … 100.00) | `float64` | **%** | measurement |
| `temperature_celsius` | Daily mean air temperature (−2.31 … 45) | `float64` | **°C** | measurement |
| `flood_risk_score` | Synthesised composite risk index on a 0.01 grid, 1.00 … 10.00 (899 distinct) | `float64` | index 1–10 (dimensionless) | outcome-derived |
| `flood_event_occurred` | Same-day flood label: 1 if the station flooded that day | `int64` | 0/1 | label |
| `severity_level` | 4-level binning of `flood_risk_score` (Low/Moderate/High/Extreme) | `str` | – | outcome-derived |
| `season` | Meteorological season, deterministic from `month` (Winter Dec–Feb, Spring Mar–May, Summer Jun–Aug, Autumn Sep–Nov) | `str` | – | calendar |
| `monsoon_season` | 1 for Jun–Oct, else 0; deterministic from `month` | `int64` | 0/1 | calendar |
| `data_quality_score` | Per-row quality score, 0.88 … 0.99 (111 distinct); uncorrelated with everything | `float64` | index 0.88–0.99 | measurement |

Six columns (`year`, `month`, `day`, `day_of_year`, `season`, `monsoon_season`) are deterministic
re-encodings of `date` — verified with **0 mismatches** on all 456,600 rows for `date` vs
`(year, month, day)`, for `day_of_year` vs `date.dt.dayofyear`, for `season` vs `month`, and for
`monsoon_season` vs `month ∈ Jun…Oct`. Three columns (`flood_risk_score`, `severity_level`,
`flood_event_occurred`) are outcome-derived and are dealt with in Section B.

**Companion files.**

| file | shape | what it is | used? |
|---|---|---|---|
| `data/station_daily_data_full.csv` | 456,600 × 21 | the complete daily panel | **yes — the only data source** |
| `data/station_daily_data.csv` | 100,000 × 21 | a uniform random **21.90 %** sample of the full file | **no** |
| `data/vulnerability_data.csv` | 10 × 18 | one row per basin: area, population, poverty, road density, hospitals, schools, floodplain %, three resilience scores, vulnerability score, impact potential, annual damage, priority | static context only |
| `data/asia_flood_geospatial.geojson` | 10 features | **10 basin `Polygon`s**, properties `basin, country, area_km2, population_millions, flood_risk_level, vulnerability_level` | static context only |

The 100,000-row sample is a genuine random subset — all 50 stations appear, all 100,000
`(station_id, date)` pairs exist in the full file, per-station counts span 1,850–2,084, and the
column order is identical. It is **not used**: it keeps only **21.90 %** of the rows and,
decisively, it **breaks the daily time axis** that the project's lag and rolling-window features
depend on. A sampled station-day has no guarantee of a *t*−1 or *t*−7 neighbour, so every window
feature built on it would silently be a different quantity.

!!! note "Correction to `docs/projects/index.md`"
    That page lists `asia_flood_geospatial.geojson` as holding **"50 station coordinates"**. It
    does not: it holds **10 basin polygons**. Station coordinates live in the CSV, as the
    `latitude`/`longitude` columns — and they are constants repeated on every row (Section B).

### B. Quality

**Missing values — there are none.** All 21 columns report **count 0 (0.00 %)**, 0 of 9,588,600
cells, and **0 columns with any missing value**. That *is* the finding: no imputation is needed,
and none may be claimed. Any imputer in the modelling pipeline is a robustness measure against a
future partial day, not a fix for an observed problem.

| column | missing | % | column | missing | % | column | missing | % |
|---|---|---|---|---|---|---|---|---|
| `station_id` | 0 | 0.00 | `rainfall_mm` | 0 | 0.00 | `severity_level` | 0 | 0.00 |
| `station_name` | 0 | 0.00 | `river_level_m` | 0 | 0.00 | `season` | 0 | 0.00 |
| `basin` | 0 | 0.00 | `soil_moisture_percent` | 0 | 0.00 | `monsoon_season` | 0 | 0.00 |
| `country` | 0 | 0.00 | `temperature_celsius` | 0 | 0.00 | `data_quality_score` | 0 | 0.00 |
| `latitude` | 0 | 0.00 | `flood_risk_score` | 0 | 0.00 | `date` | 0 | 0.00 |
| `longitude` | 0 | 0.00 | `flood_event_occurred` | 0 | 0.00 | `year` | 0 | 0.00 |
| `month` | 0 | 0.00 | `day` | 0 | 0.00 | `day_of_year` | 0 | 0.00 |

**Duplicates and panel completeness.** **0** fully duplicated rows and **0** duplicated
`(station_id, date)` pairs. Rows per station take a **single unique value, 9,132**, for all 50
stations; the range 2000-01-01 → 2024-12-31 contains exactly **9,132** days and **0** of them are
absent from the data; every station carries the same date set. 50 × 9,132 = **456,600**, exactly
the row count. The panel is rectangular and complete.

![Figure 1.1](figures/fig01_panel_completeness.png)

**Figure 1.1.** Panel completeness. **(a)** Rows per station, against the expected 9,132 — every
one of the 50 bars is identical. **(b)** Date coverage per station over 2000-01-01 → 2024-12-31,
with the three split windows marked. *Conclusion:* the panel is **balanced and gap-free**, so no
station or period needs special handling and every cross-station statistic is directly
comparable.

**Inconsistencies, quantified.**

1. **The one genuine internal contradiction — 32 rows.** Exactly **32 rows have
   `flood_risk_score == 7.00`**; all 32 are `severity_level == "Extreme"` and all 32 have
   `flood_event_occurred == 0`. Because the label is the strict rule `risk > 7.0` while the
   severity bin includes 7.00, this is the only place where the dataset disagrees with itself.
2. **`soil_moisture_percent` is censored at exactly 100.00** — **9,153 rows (2.0046 %)** sit on
   that ceiling, which is a clipping artefact, not a measurement. Any transform must preserve the
   tie at the boundary.
3. **`latitude`/`longitude` are constants, not measurements.** Each of the 50 stations has
   **exactly 1 unique value** of `latitude` and of `longitude` (and of `station_name`, `basin`,
   `country`). Only **50 distinct coordinate pairs** exist, and `(station_name, latitude,
   longitude)` is duplicated on **456,550** rows: geography is static per-station metadata
   repeated on every row, so it cannot carry day-to-day information.
4. **`country` is redundant.** `basin → country` is a **function**: all **10** basins map to
   exactly **1** country. 7 countries cover 10 basins — India, China and Myanmar hold 91,320 rows
   each (2 basins), Vietnam, Pakistan, Afghanistan and Thailand 45,660 each (1 basin). `country`
   adds no information that `basin` does not already carry.
5. **Station names contradict their coordinates.** The five `basin == "Mekong"` stations are all
   tagged `country == "Vietnam"`, yet they span 14.3194–16.6476 °N and 102.8958–107.0847 °E:

   | station_id | station_name | latitude | longitude | country |
   |---|---|---|---|---|
   | `ASIA_0006` | Chiang Saen | 14.3194 | 103.0756 | Vietnam |
   | `ASIA_0007` | Luang Prabang | 16.1453 | 103.0213 | Vietnam |
   | `ASIA_0008` | Vientiane | 15.3964 | 107.0847 | Vietnam |
   | `ASIA_0009` | Phnom Penh | 15.3598 | 103.4985 | Vietnam |
   | `ASIA_0010` | Chau Doc | 16.6476 | 102.8958 | Vietnam |

   "Luang Prabang" at **16.1453 °N 103.0213 °E** lies in **Thailand**, not Laos, and none of
   these five towns sits in Vietnam at the stated coordinates. The `Helmand` group is likewise
   spread over **60.5478–65.1159 °E**. Names are decorative labels; the coordinates are what the
   file actually encodes.

**Columns to drop, each with its reason.** The list is the single source of truth in
`code/data.py` (`DROPPED`, 8 entries) and is printed by the script:

| dropped column | reason |
|---|---|
| `severity_level` | target leakage — a deterministic binning of `flood_risk_score` |
| `flood_risk_score` | target leakage / redundant — `risk > 7.0` reproduces the same-day label on 456,600/456,600 rows (ROC-AUC 1.0000); r = 0.934 with `soil_moisture_percent`, and a *weaker* next-day predictor (AUC 0.8088) |
| `data_quality_score` | noise — \|r\| ≤ 0.004 against every other column (re-derived max 0.0022, against `longitude`); next-day ROC-AUC 0.5072 |
| `station_name` | redundant — 1:1 with `station_id` (50 levels each) |
| `country` | redundant — a function of `basin` (exactly one country per basin) |
| `day` | noise — day-of-month, r = 0.0007 with the next-day target |
| `latitude` | constant within station — 1 unique value for each of the 50 stations |
| `longitude` | constant within station — 1 unique value for each of the 50 stations |

This leaves **6 numeric** inputs (`rainfall_mm`, `river_level_m`, `soil_moisture_percent`,
`temperature_celsius`, `day_of_year`, `monsoon_season`) and **2 categorical** ones (`basin`,
`season`) — 8 model features from 21 raw columns.

The two "noise" drops were re-derived rather than asserted. `data_quality_score` reaches a
maximum **|r| = 0.0022** against the 13 other numeric columns (its strongest partner is
`longitude`), and correlates **−0.0019** with the same-day label and **+0.0036** with the next-day
target — the `data.py` entry records the conservative bound **|r| ≤ 0.004**. Its next-day ROC-AUC
is **0.5072**, i.e. chance. `day` (day-of-month) correlates **0.0007** with the next-day target.
Both are correctly discarded.

**Leakage investigation.** This is the decisive finding of stage 1.

!!! warning "Two columns give the answer away"
    On **all 456,600 rows**, the same-day label is exactly the threshold rule

    ```
    flood_event_occurred  ==  (flood_risk_score > 7.0)
    ```

    with **max(risk | flood = 0) = 7.00** and **min(risk | flood = 1) = 7.01**: the classes are
    **linearly separable** on that one column, and its ROC-AUC against the same-day label is
    **1.0000**. `severity_level` is a second copy of the same information.

| `severity_level` | `flood_risk_score` min | max | rows | floods |
|---|---|---|---|---|
| Low | 1.00 | 2.99 | 234,226 | 0 |
| Moderate | 3.00 | 4.99 | 129,974 | 0 |
| High | 5.00 | 6.99 | 82,479 | 0 |
| Extreme | 7.00 | 10.00 | 9,921 | 9,889 |
| **all** | 1.00 | 10.00 | **456,600** | **9,889** |

`severity_level` is a 4-bin re-encoding of `flood_risk_score` with **left-closed, right-open**
intervals — `[1,3)` Low, `[3,5)` Moderate, `[5,7)` High, `[7,10]` Extreme. The interval
convention matters and is worth stating precisely, because the obvious one-liner is wrong:
`pd.cut(risk, bins=(1,3,5,7,10])` with pandas' default `right=True` reproduces `severity_level`
on only **450,602 / 456,600** rows (5,998 rows sit exactly on an edge); dropping the top edge with
`right=False` leaves the **31** rows at `risk == 10.00` as `NaN` and reproduces 456,569. The exact
reproduction is `pd.cut(risk, bins=(1, 3, 5, 7, 10.01], labels=[…], right=False)` —
**456,600 / 456,600**. The edge values are 1.00 (4,404 rows), 3.00 (711), 5.00 (851), 7.00 (32)
and 10.00 (31).

That `Extreme` bin holding 9,921 rows but only 9,889 floods is precisely the 32-row
contradiction from item 1 above: `risk == 7.00` is binned `Extreme` yet is not a flood.

**The honest nuance for a one-day-ahead target.** For the *same-day* label these columns are
leakage. For the project's target, `y(t) = flood_event_occurred(t+1)`, a value observed at time
*t* is not strictly leakage — it is merely *stale*. The columns are excluded anyway because they
are **redundant, not informative**:

- `flood_risk_score` correlates **r = 0.9337** with `soil_moisture_percent`, 0.6643 with
  `river_level_m` and 0.5949 with `rainfall_mm` — it is a composite of the measurements already in
  the feature set, not a new observation.
- As a single predictor of the **next-day** label it scores **ROC-AUC 0.8088**, *below*
  `soil_moisture_percent` (**0.8364**) and barely above `rainfall_mm` (0.8091); the other two
  measurements follow at 0.7956 (`temperature_celsius`) and 0.7948 (`river_level_m`).

So excluding `flood_risk_score` and `severity_level` costs nothing in next-day signal while
removing a generator artefact that would otherwise hand the model a shortcut. They are shown
**once**, explicitly labelled, as a leaky reference point — never as features.

### C. Target

The deliverable's label is the **one-day-ahead flood event**,

```
y(t) = flood_event_occurred(t + 1)        built with groupby("station_id")["flood_event_occurred"].shift(-1)
```

which is the question the project actually asks: *will this station flood tomorrow?* Shifting
drops the last day of every station — **50 rows** (50 stations × 2024-12-31) — leaving
**456,550 usable rows**.

| | count | share |
|---|---|---|
| `y = 0` (no flood tomorrow) | 446,661 | 97.834 % |
| `y = 1` (flood tomorrow) | 9,889 | **2.166 %** |
| imbalance (negatives : positives) | — | **45.2 : 1** |

The positive rate is **2.166028 %** — for reference the raw same-day label gives **2.165791 %**,
and the **positive count is identical, 9,889 either way**. That is not a coincidence: shifting
drops 2000-01-01 (which has **0** floods — January) at one end and adds 2024-12-31 (also **0**
floods) at the other, so the shifted window contains exactly the same flood days. The
majority-class accuracy is therefore **97.83 %**; that number is stated exactly once and
**accuracy is never used again** in this deliverable. The headline metric is **PR-AUC**, with
ROC-AUC alongside.

![Figure 1.2](figures/fig01_target_balance.png)

**Figure 1.2.** Class balance of the next-day label over the 456,550 usable station-days.
**(a)** Raw counts, with the 97.83 % majority-class accuracy annotated. **(b)** The same counts on
a log axis, annotated with the 45.17 : 1 ratio. **(c)** Class shares; the positive class is the
2.166 % red sliver. *Conclusion:* the problem is **severely imbalanced** — a trivial
"never floods" classifier is 97.83 % accurate, so accuracy is uninformative and every reported
metric must be read against the 2.166 % base rate.

**The seasonal gate is part of the target distribution.** The label is not merely imbalanced — it
is *structurally* absent for most of the year. **0 floods** occur in January–April and in
November–December, under *either* label definition.

| month | rows | floods (same-day) | rate (same-day) | floods (next-day) | rate (next-day) |
|---|---|---|---|---|---|
| Jan | 38,750 | 0 | 0.0000 % | 0 | 0.0000 % |
| Feb | 35,350 | 0 | 0.0000 % | 0 | 0.0000 % |
| Mar | 38,750 | 0 | 0.0000 % | 0 | 0.0000 % |
| Apr | 37,500 | 0 | 0.0000 % | 0 | 0.0000 % |
| May | 38,750 | 1 | 0.0026 % | 1 | 0.0026 % |
| Jun | 37,500 | 1 | 0.0027 % | 92 | 0.2453 % |
| **Jul** | 38,750 | 3,128 | **8.0723 %** | 3,149 | 8.1265 % |
| **Aug** | 38,750 | 3,141 | **8.1058 %** | 3,132 | 8.0826 % |
| **Sep** | 37,500 | 3,130 | **8.3467 %** | 3,045 | 8.1200 % |
| Oct | 38,750 | 488 | 1.2594 % | 470 | 1.2129 % |
| Nov | 37,500 | 0 | 0.0000 % | 0 | 0.0000 % |
| Dec | 38,750 | 0 | 0.0000 % | 0 | 0.0000 % |

**95.04 %** of all floods (9,399 of 9,889) fall in **July–September**, split near-perfectly evenly
across the three months (31.63 %, 31.76 %, 31.65 % of all floods); October adds **4.93 %**; the
entire rest of the year contributes **two days** (0.01 % in May and 0.01 % in June). By the
`monsoon_season` flag the contrast is starker still: a flood rate of **5.170196 %** when
`monsoon_season = 1` (9,888 of 191,250) against **0.000377 %** when it is 0 (**1** of 265,350) —
a factor of **13,717×**.

Under the *next-day* label the picture is the same with one honest wrinkle: June absorbs **92**
positives, because a flood on 1 July is predicted from 30 June. The July–September share becomes
**94.31 %** (9,326 of 9,889) and the monsoon rates are unchanged (5.170196 % inside,
0.000377 % outside). Whichever framing is used, the conclusion holds: **"never flood outside the
monsoon" is a free baseline, and any model must be judged inside the monsoon window**, where the
real problem is a ~5.2 % base rate rather than 2.2 %. A global AUC quoted as if it were skill
would be measuring the calendar.

### D. Train and test

**Temporal splitting is the appropriate choice; stratified-random is not.** Two properties of
this panel make a random split invalid:

1. **Rows one day apart are near-duplicates.** Pooled within-station lag-1 autocorrelation is
   **0.968** for `temperature_celsius`, **0.820** for `soil_moisture_percent`, **0.772** for
   `flood_risk_score`, 0.220 for `river_level_m` and 0.162 for `rainfall_mm`. A test row drawn at
   random is, for the two strongest features, a near-copy of a training row.
2. **A 45:1 label leaks across a random boundary.** Demonstrated on this panel with
   `train_test_split(test_size=0.20, random_state=42, stratify=…)`: **1,250 of 1,250**
   `(station, year)` blocks (100.0 %) end up on **both** sides, **145,898** adjacent-day pairs
   straddle the boundary, and **9,132 of 9,132** calendar dates appear on more than one side.
   Stratification controls the class ratio; it does nothing about any of this. The date-based
   split by contrast shares **0** dates, splits **0** of the 1,250 blocks, and has exactly **100**
   straddling pairs — the two year boundaries × 50 stations.

**The split adopted** (`SPLITS` in `code/data.py`) is chronological, **by date, identical dates
for every station**, with the seed fixed at **`random_state = SEED = 42`** for every downstream
stochastic step (sampling, PCA, t-SNE, UMAP, any later model).

| split | years | rows in the year window | **rows usable (next-day)** | floods | next-day rate | same-day rate | stations | dates (year window) |
|---|---|---|---|---|---|---|---|---|
| **train** | 2000–2016 | 310,500 | **310,500** | 6,566 | **2.1147 %** | 2.1147 % | 50 | 2000-01-01 → 2016-12-31 |
| **validation** | 2017–2020 | 73,050 | **73,050** | 1,620 | **2.2177 %** | 2.2177 % | 50 | 2017-01-01 → 2020-12-31 |
| **test** | 2021–2024 | 73,050 | **73,000** | 1,703 | **2.3329 %** | 2.3313 % | 50 | 2021-01-01 → 2024-12-31 |

The two row counts differ for test and only for test: the 50 dropped rows are all 2024-12-31,
which lies in the 2021–2024 window, so the year window holds **73,050** rows while the *usable*
next-day test set holds **73,000** and covers 2021-01-01 → **2024-12-30**. Train and validation
are unaffected, because the shifted label for their last day (2016-12-31, 2020-12-31) exists on
the following day. Consequently the test **flood rate under the next-day target is 2.3329 %**;
**2.3313 %** is the same-day rate over the 73,050-row window. 310,500 + 73,050 + 73,000 =
**456,550**, the usable frame. **All 50 stations appear in all three splits**, so no split is
starved of stations or of positives.

![Figure 1.3](figures/fig01_floods_by_year.png)

**Figure 1.3.** Flood days per year 2000–2024 (bars, left axis) and the annual flood rate (line,
right axis), with the train / validation / test windows shaded. *Conclusion:* flood counts are
**stable across all 25 years** at a minimum of 354, a maximum of 439 and a mean of 395.6 per year
— there is no regime change or structural break that a chronological split would turn into a
distribution-shift trap, and the three splits are therefore comparable.

There is a **mild upward drift** in the rate: the first year (2000) sits at 2.0820 % and the last
(2024) at 2.3716 %, and the split rates follow it from 2.1147 % (train) to 2.2177 % (validation)
to **2.3329 % (test)**. This is why the **test** rate is the one quoted whenever a test base rate
is needed: the model is scored on a slightly more flood-prone era than the one it is trained on.

!!! note "Fitting discipline from here on"
    **From this point on, every preprocessing statistic — scalers, imputers, station means and
    standard deviations, category vocabularies, any decision threshold — is fitted on the
    training split only.** Validation and test rows are transformed with those fitted values and
    never contribute to them. No statistic anywhere in this report is computed on train+test
    combined.

```python
--8<-- "docs/projects/eda/code/t1_inspection.py"
```

## 2. Univariate analysis

### A. Numerical

Scope: every numerical column of the raw panel — the four measurements, the two static
coordinates, the two outcome-derived columns, the four calendar columns and the monsoon
gate. All 13 are described on **all 456,600 rows** in Table 2A.1. Every column reads
`count = 456,600`, so nothing is estimated, imputed or skipped. Every number quoted below
is printed by `code/t2_numeric.py`; no model is fitted in this item, and the only
stochastic step anywhere is the 20,000-row KDE subsample in Figure 2A.3 (`SEED` = 42).

*Skewness* is the adjusted Fisher–Pearson coefficient G1 (`scipy.stats.skew(bias=False)`,
identical to pandas `.skew()`). *Kurtosis* is the **excess** kurtosis
(`scipy.stats.kurtosis(bias=False)`, identical to pandas `.kurt()`): 0 is the Gaussian
reference, a negative value means *flatter than Gaussian*. *Outside* counts the rows below
Q1 − 1.5·IQR or above Q3 + 1.5·IQR.

#### Table 2A.1 — descriptive statistics for all 13 numerical columns

| column | count | mean | std | min | 1 % | 25 % | median | 75 % | 99 % | max | skewness | kurtosis | n outside | % outside |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Measurements** (raw sensor columns) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `rainfall_mm` | 456,600 | 20.28 | 41.92 | 2.02 | 2.57 | 5.72 | 10.76 | 25.95 | 209.16 | 1,775.72 | 10.513 | 158.202 | 9,274 | 2.03 |
| `river_level_m` | 456,600 | 5.57 | 5.65 | 0.66 | 1.93 | 3.16 | 4.42 | 6.79 | 30.19 | 204.67 | 9.532 | 140.606 | 9,273 | 2.03 |
| `soil_moisture_percent` | 456,600 | 36.94 | 18.15 | 0.39 | 10.97 | 22.07 | 31.00 | 52.57 | 100.00 | 100.00 | 0.864 | 0.857 | 9,180 | 2.01 |
| `temperature_celsius` | 456,600 | 24.49 | 10.49 | -2.31 | 5.83 | 15.25 | 24.49 | 33.77 | 43.88 | 45.00 | 0.006 | -1.196 | 0 | 0.00 |
| **Static geography** (constant within station) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `latitude` | 456,600 | 23.0414 | 5.3662 | 12.7104 | 12.7104 | 18.6490 | 23.9696 | 27.0986 | 33.4752 | 33.4752 | -0.214 | -0.818 | 0 | 0.00 |
| `longitude` | 456,600 | 93.0482 | 15.9320 | 60.5478 | 60.5478 | 85.1654 | 96.9826 | 103.0756 | 115.8379 | 115.8379 | -0.561 | -0.691 | 0 | 0.00 |
| **Outcome-derived** (dropped — see the caveat below) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `flood_risk_score` | 456,600 | 3.44 | 1.67 | 1.00 | 1.01 | 2.09 | 2.92 | 4.75 | 8.90 | 10.00 | 0.849 | 0.534 | 5,973 | 1.31 |
| `data_quality_score` | 456,600 | 0.9349 | 0.0317 | 0.8800 | 0.8810 | 0.9070 | 0.9350 | 0.9620 | 0.9890 | 0.9900 | 0.003 | -1.199 | 0 | 0.00 |
| **Calendar** (deterministic re-encodings of `date`) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `year` | 456,600 | 2,012.00 | 7.21 | 2,000 | 2,000 | 2,006 | 2,012 | 2,018 | 2,024 | 2,024 | 0.000 | -1.204 | 0 | 0.00 |
| `month` | 456,600 | 6.52 | 3.45 | 1 | 1 | 4 | 7 | 10 | 12 | 12 | -0.009 | -1.208 | 0 | 0.00 |
| `day` | 456,600 | 15.73 | 8.80 | 1 | 1 | 8 | 16 | 23 | 31 | 31 | 0.007 | -1.194 | 0 | 0.00 |
| `day_of_year` | 456,600 | 183.14 | 105.45 | 1 | 4 | 92 | 183 | 274 | 362 | 366 | 0.000 | -1.200 | 0 | 0.00 |
| **Seasonal gate** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `monsoon_season` | 456,600 | 0.42 | 0.49 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0.329 | -1.892 | 0 | 0.00 |

#### Table 2A.2 — the 1.5×IQR fences these flags come from

| column | 25 % (Q1) | 75 % (Q3) | IQR | lower fence | upper fence | n outside | % outside |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `rainfall_mm` | 5.72 | 25.95 | 20.23 | -24.62 | 56.30 | 9,274 | 2.03 |
| `river_level_m` | 3.16 | 6.79 | 3.63 | -2.29 | 12.23 | 9,273 | 2.03 |
| `soil_moisture_percent` | 22.07 | 52.57 | 30.50 | -23.68 | 98.32 | 9,180 | 2.01 |
| `temperature_celsius` | 15.25 | 33.77 | 18.52 | -12.53 | 61.55 | 0 | 0.00 |
| `latitude` | 18.6490 | 27.0986 | 8.4496 | 5.9746 | 39.7730 | 0 | 0.00 |
| `longitude` | 85.1654 | 103.0756 | 17.9102 | 58.3001 | 129.9409 | 0 | 0.00 |
| `flood_risk_score` | 2.09 | 4.75 | 2.66 | -1.90 | 8.74 | 5,973 | 1.31 |
| `data_quality_score` | 0.9070 | 0.9620 | 0.0550 | 0.8245 | 1.0445 | 0 | 0.00 |
| `year` | 2,006 | 2,018 | 12 | 1,988 | 2,036 | 0 | 0.00 |
| `month` | 4 | 10 | 6 | -5 | 19 | 0 | 0.00 |
| `day` | 8 | 23 | 15 | -14 | 46 | 0 | 0.00 |
| `day_of_year` | 92 | 274 | 182 | -181 | 547 | 0 | 0.00 |
| `monsoon_season` | 0 | 1 | 1 | -2 | 2 | 0 | 0.00 |

Nine of the thirteen columns have **no** point outside their fence. The four that do are the
three skewed measurements plus the dropped `flood_risk_score` — so the whole of Figure 2A.2
is spent on the four measurements.

---

#### 1. `rainfall_mm` — severely right-skewed

mean 20.28 mm against a median of 10.76 mm: the mean sits at **1.88×** the median, and the
maximum of **1,775.72 mm** is **165.0×** the median. Half of the panel lies between
5.72 mm (Q1) and 25.95 mm (Q3) while the top 1 % runs out to 209.16 mm. Skewness **10.513**
and excess kurtosis **158.202** are the largest of any column in the dataset. This is the
textbook case for a variance-stabilising transform: `log1p` moves the skew to **0.69**,
the excess kurtosis to **1.08**, and the mean/median ratio from 1.88× to **1.06×** — the
distribution becomes symmetric and single-peaked (Figure 2A.1, bottom-left).

#### 2. `river_level_m` — the same failure, slightly milder

mean 5.57 m against a median of 4.42 m (**1.26×**), maximum **204.67 m** = **46.3×** the
median, skewness **9.532**, excess kurtosis **140.606**. `log1p` brings the skew to
**1.50** and the kurtosis to **5.65**, with the mean/median ratio at **1.04×**. A river
level is physically bounded below by the channel bed and unbounded above, which is exactly
the geometry that produces a log-normal-looking tail; the log transform is the natural
scale for it.

#### 3. `soil_moisture_percent` — moderately skewed **and censored**

Skewness 0.864 and excess kurtosis 0.857 make this the mildest of the three skewed
variables: `log1p` moves the skew only to **-0.10** (a slight over-correction), so the
transform is optional here — but the column carries a defect that no transform can repair.

!!! warning "`soil_moisture_percent` is right-censored at exactly 100.00 %"
    **9,153 rows (2.005 % of the panel)** hold the value 100.00. The next distinct value
    below it is 99.93, and 100.00 is also the 99th percentile — the whole top 1 % of the
    column is pinned at one number. The bin evidence is decisive: the 0.1-wide bin ending at
    100.00 holds 9,155 rows against a mean of 1.3 for the six bins below it, i.e. **68,648×**
    the count a single 0.01-wide slot would carry at the local density. This is a **ceiling
    artefact of the generator, not a measurement**. Consequence: any transform applied to
    this column must be **monotone** so that the 9,153 rows stay tied at one value — `log1p`
    does keep the tie (`log1p(100.00) = 4.62`, still 9,153 rows) but cannot remove the
    pile-up. A square-root or a quantile transform would be equally acceptable; a
    standardiser fitted on the train split is the only safe companion.

Of the 9,180 rows outside the fence at 98.32 %, 9,153 **are** the ceiling tie and only
**27** are genuine tail readings between 98.32 and 100.00.

#### 4. `temperature_celsius` — flat, not bell-shaped, and also censored

Skewness **0.006** and excess kurtosis **-1.196**: this is the one measurement that is not
skewed, and it is the one whose *shape* needs explaining. It runs from -2.31 °C to
45.00 °C with mean = median = 24.49 °C, and **0 rows** fall outside its fence (which sits
at 61.55 °C, 16.55 °C beyond the maximum). The negative excess kurtosis is the signature of
a **bounded, plateau-like density**, not of weather noise: the pooled distribution is the
projection of one smooth seasonal cycle plus daily noise, which spreads the mass across the
whole range instead of concentrating it near the centre (Figure 2A.3). The monthly means run
from 10.73 °C in month 3 to 38.28 °C in month 9 — a **27.55 °C** cycle — while the
within-month standard deviation is only **3.45–4.10 °C**. The cycle is therefore
**6.7–8.0×** the daily spread, which is exactly why the pooled histogram is flat/bimodal.

`temperature_celsius` is **also** right-censored, just far less often than soil moisture:
**1,788 rows (0.392 %)** sit at exactly 45.00 °C, and the 0.1-wide bin ending at 45.00
holds 1,967 rows against a mean of 224.3 for the six bins below it — **80×** the count a
single 0.01-wide slot would carry. Unlike soil moisture this censoring is **invisible to
the IQR rule** (0 rows flagged, the fence is 16.55 °C away), but it is equally a boundary
artefact and equally untransformable.

#### 5. `latitude` / `longitude` — flat quantiles by construction

Both columns have exactly **50 distinct values** and there are exactly **50 distinct
(latitude, longitude) pairs**: the coordinates are constants repeated on all 9,132 rows of
each station, not measurements repeated in time. Each coordinate value therefore carries
9,132 / 456,600 = **2.000 %** of the rows, which is more than 1 %, so the 1st percentile
cannot leave the smallest value and the 99th cannot leave the largest — the table shows
`1 % = min = 12.7104` and `99 % = max = 33.4752` (latitude; the same holds for longitude at
60.5478 and 115.8379). The flat quantiles are an artefact of the cardinality, nothing else.
Skew is mild and negative (-0.214 and -0.561) and no row falls outside either fence.

!!! note "`latitude` and `longitude` are not features"
    They are station identifiers in disguise (stage 1B, and `DROPPED` in `code/data.py`).
    They are described here because the univariate stage covers every numerical column, not
    because they enter the model.

#### 6. The two outcome-derived columns — described here, dropped from the feature set

`flood_risk_score` is a bounded index: [1.00, 10.00] on a **0.01 grid** (899 distinct
values, grid check passes), median 2.92, mean 3.44, skew 0.849, and 5,973 rows (1.31 %)
outside its fence at 8.74. `data_quality_score` is [0.8800, 0.9900] on a **0.001 grid**
(111 distinct values), near-uniform (skew **0.003**, excess kurtosis **-1.199**) with
**0** rows outside its fence.

!!! warning "Both columns are dropped — cross-reference to stage 1B, not re-derived here"
    `flood_risk_score` is the leaked target: `flood_event_occurred == (flood_risk_score > 7.0)`
    holds on **456,600 / 456,600** rows (ROC-AUC **1.0000**), and it is redundant even for the
    one-day-ahead target (r = **0.934** with `soil_moisture_percent`; next-day ROC-AUC
    **0.8088**, *below* `soil_moisture_percent` alone). `data_quality_score` is pure noise:
    |r| ≤ **0.004** against every other column, next-day ROC-AUC **0.5072**. Both bounds are
    quoted verbatim from the frozen `DROPPED` mapping in `code/data.py`, which is the single
    source of truth for the drop; the leakage rule was verified on all 456,600 rows in 1B and
    is **not** re-derived here. The two columns appear in Table 2A.1 only so that the
    univariate stage is complete.

#### 7. The calendar columns — and why their kurtosis is also telling

`year`, `month`, `day` and `day_of_year` reproduce `date` exactly (**0** mismatches against
`(year, month, day)` and **0** against `date.dt.dayofyear`; 350 rows on the leap day 366).
Their excess kurtosis is **-1.194 to -1.208** with |skew| ≤ 0.009 — the signature of a
**uniform** distribution, which is what a perfectly balanced panel must produce. It is the
same diagnostic that identifies `temperature_celsius`'s flatness, applied to a column where
flatness is expected rather than interesting. `monsoon_season` has mean 0.42 (the Jun–Oct
gate is on for 42 % of station-days) and no row outside its fence.

---

#### The figure budget: three figures, and why exactly these three

Stage 2 allows at most **3 figures per item**, and choosing what deserves one is part of
the analysis. Tables 2A.1 and 2A.2 already carry every statistic the rubric asks for, so a
figure is only justified where a picture proves something a number cannot:

1. **Figure 2A.1** — raw vs `log1p` for the four measurements. This is the evidence for the
   single most consequential preprocessing decision in the deliverable (stage 4A: `log1p`
   on `rainfall_mm` and `river_level_m`, nothing else). A skew of 10.513 is a number; the
   spike-and-flat-tail it describes, and its disappearance under `log1p`, is an argument.
2. **Figure 2A.2** — the four measurements as boxplots with the 1.5×IQR fences drawn. This
   is the evidence for the *opposite* decision: 9,274 + 9,273 + 9,180 rows sit outside those
   fences, and the tails run **31.5×** and **16.7×** past them. It shows what an "outlier
   removal" step would actually delete, and that the tails are continuous.
3. **Figure 2A.3** — `temperature_celsius`. It is the only measurement that is not skewed,
   and the only one whose shape is *generated* rather than observed: the flat pooled density
   is the shadow of one seasonal sinusoid. Without this figure the "temperature is fine,
   leave it alone" decision in 4A would be unsupported.
A fourth figure (per-station rainfall, a QQ plot, a violin grid) would restate one of these
three messages and cost a deduction. The static coordinates, the calendar columns and the
two dropped columns are numbers in a table; they need no picture.

#### Figure 2A.1 — raw vs `log1p`, the four measurements

![Figure 2A.1 — the four measurements, raw (top) vs log1p (bottom)](figures/fig10_measurements_hist_log.png)

**Figure 2A.1.** Columns = the four measurements; top row = raw values (mm, m, %, °C),
bottom row = `log1p`, with the dashed red line at the mean and the dotted orange line at
the median of the values actually plotted. Titles carry the skewness and excess kurtosis of
the plotted series. `log1p` is undefined for negative numbers, so the `temperature_celsius`
panel is shifted by its minimum (-2.31 °C) and labelled as such. *Interpretation:*
`rainfall_mm` and `river_level_m` collapse from an unreadable spike into a single-peaked,
near-symmetric distribution — the mean/median ratio falls from 1.88× to 1.06× and from
1.26× to 1.04×. `soil_moisture_percent` keeps its isolated bar at the ceiling
(`log1p(100.00) = 4.62`, still 9,153 tied rows), confirming that the artefact survives the
transform. `temperature_celsius` is the counter-example: it starts symmetric (skew 0.006)
and *shifting + log-compressing it manufactures a left skew of -0.57*. The transform is
not free, and this figure is why it is applied to exactly two of the four measurements.

#### Figure 2A.2 — the 1.5×IQR fences and the tails they flag

![Figure 2A.2 — boxplots of the four measurements with the 1.5×IQR fence](figures/fig10_measurements_box.png)

**Figure 2A.2.** Horizontal boxplots of the four measurements on their full data range
(units: mm, m, %, °C). The dashed red line is the upper 1.5×IQR fence, the dotted grey
lines are Q1 and Q3, and the inset in each panel re-draws the same box over the **bulk
only** (min → fence), because on the full range the box of `rainfall_mm` is a sliver against
a 1,775.72 mm tail. *Interpretation:* the fence for `rainfall_mm` sits at 56.30 mm and flags
9,274 station-days (2.03 %) while the tail continues to 1,775.72 mm — 31.5× beyond the fence;
`river_level_m` behaves identically (fence 12.23 m, 9,273 rows, max 204.67 m). The flagged
points form a **continuous, dense tail**, not a handful of impossible readings. The two
panels that differ are the informative ones: `soil_moisture_percent` is flagged for a
*single tied value* (9,153 of its 9,180 flagged rows are the 100.00 ceiling), and
`temperature_celsius` has **no** flagged row at all because its fence sits 16.55 °C beyond
its maximum. This figure is the direct input to the stage 4A outlier decision: the rows an
IQR filter would delete are the rows that make the column informative, so **no row is
removed and nothing is winsorised**.

#### Figure 2A.3 — `temperature_celsius`: one seasonal sinusoid behind a flat density

![Figure 2A.3 — temperature histogram with the monthly means, and the monthly cycle](figures/fig10_temperature_shape.png)

**Figure 2A.3.** Left: pooled probability density of `temperature_celsius` (°C) over all
456,600 station-days (120 bins) with a Gaussian KDE (20,000-row subsample, `SEED` = 42),
and the twelve monthly means marked as a rug on the axis. Right: monthly mean ± 1 sd versus
month, with the annual mean as a reference line. *Interpretation:* the pooled density is
flat with two soft modes, and its excess kurtosis of -1.196 is the *shape consequence of a
bounded sweep*, not evidence of a mixture. Moving to the right panel explains it: the mean
travels smoothly from 10.73 °C (month 3) to 38.28 °C (month 9), an amplitude of 27.55 °C,
while the within-month sd is only 3.45–4.10 °C — the cycle is **6.7–8.0×** the daily spread.
The pooled density is therefore the projection of that single sinusoid onto the temperature
axis, and the two soft modes are the two turning points where the cycle lingers. Consequence
for stage 4A: `temperature_celsius` needs **no** transform, and it must **not** be `log1p`-ed
(cf. Figure 2A.1). It is also, at 1,788 rows exactly at 45.00 °C, the dataset's second
boundary-censored measurement, which Figure 2A.3 annotates on the right-hand spike.

---

#### What this hands to stage 4A

| Decision | Motivated by |
|---|---|
| `log1p` on `rainfall_mm` and `river_level_m` **only** | skew 10.513 → 0.69 and 9.532 → 1.50; excess kurtosis 158.202 → 1.08 and 140.606 → 5.65 (Figure 2A.1) |
| **No** transform on `temperature_celsius` | already symmetric (skew 0.006, excess kurtosis -1.196); the shifted `log1p` manufactures a skew of -0.57 (Figure 2A.1) |
| Any `soil_moisture_percent` transform must be monotone | 9,153 rows tied at the 100.00 ceiling; `log1p` preserves the tie at 4.62 but cannot remove it |
| **No** row removal, **no** winsorising | 9,274 + 9,273 + 9,180 flagged rows form continuous tails that reach 165.0× and 46.3× the median (Figure 2A.2) |
| `StandardScaler` after `log1p`, fitted on train only | the four measurements span 0.39–1,775.72 in their native units |

#### Script

```python
--8<-- "docs/projects/eda/code/t2_numeric.py"
```

### B. Categorical

Seven columns carry categorical information: two station identifiers (`station_id`,
`station_name`), two geography labels (`basin`, `country`), the outcome-derived bin
(`severity_level`) and two calendar labels (`season`, `monsoon_season`). `monsoon_season` is a
0/1 flag and is listed under *numerical* features for the model, but for a frequency count it is
a two-level categorical — it belongs in this table.

Every count below is over the **full panel of 456,600 rows** (all timings of the 50 stations ×
9,132 days). Recomputing every level share on the 456,550-row one-day-ahead frame moves no share
by more than **0.0082 percentage points**, so the frequencies below are the frequencies the model
will see. 127 distinct levels exist across the seven columns and **0 of them hold less than 1 %
of the rows** — every column's "levels < 1 %" cell in Table 2B.1 is a zero.

**Table 2B.1 — cardinality and frequency of every categorical column** (n = 456,600 rows).

| column | cardinality | most frequent level — rows (share) | least frequent level — rows (share) | levels < 1 % of rows |
|---|---:|---|---|---:|
| `station_id` | **50** | all 50 tied — 9,132 (**2.0000 %**) | all 50 tied — 9,132 (**2.0000 %**) | 0 |
| `station_name` | **50** | all 50 tied — 9,132 (**2.0000 %**) | all 50 tied — 9,132 (**2.0000 %**) | 0 |
| `basin` | **10** | all 10 tied — 45,660 (**10.0000 %**) | all 10 tied — 45,660 (**10.0000 %**) | 0 |
| `country` | **7** | China / India / Myanmar — 91,320 (**20.0000 %**) | Afghanistan / Pakistan / Thailand / Vietnam — 45,660 (**10.0000 %**) | 0 |
| `severity_level` ⚠ *leaky* | **4** | Low — 234,226 (51.2979 %) | Extreme — 9,921 (**2.1728 %**) | 0 |
| `season` | **4** | Spring / Summer — 115,000 (25.1862 %) | Winter — 112,850 (**24.7153 %**) | 0 |
| `monsoon_season` | **2** | 0 (Nov-May) — 265,350 (58.1143 %) | 1 (Jun-Oct) — 191,250 (41.8857 %) | 0 |

Table 2B.2 gives the level-by-level breakdown for the five columns that are not station
identifiers (the two 50-level identifier columns are perfectly flat at 9,132 rows each and would
add 100 identical rows to the table).

**Table 2B.2 — every level of the five non-identifier categorical columns** (share of 456,600 rows).

| column | level | rows | share of rows |
|---|---|---:|---:|
| `basin` | Brahmaputra, Chao Phraya, Ganges, Helmand, Indus, Irrawaddy, Mekong, Pearl, Salween, Yangtze | 45,660 each | 10.0000 % each |
| `country` | China, India, Myanmar | 91,320 each | 20.0000 % each |
| `country` | Afghanistan, Pakistan, Thailand, Vietnam | 45,660 each | 10.0000 % each |
| `severity_level` | Low | 234,226 | 51.2979 % |
| `severity_level` | Moderate | 129,974 | 28.4656 % |
| `severity_level` | High | 82,479 | 18.0637 % |
| `severity_level` | Extreme | 9,921 | 2.1728 % |
| `season` | Spring | 115,000 | 25.1862 % |
| `season` | Summer | 115,000 | 25.1862 % |
| `season` | Autumn | 113,750 | 24.9124 % |
| `season` | Winter | 112,850 | 24.7153 % |
| `monsoon_season` | 0 (Nov-May) | 265,350 | 58.1143 % |
| `monsoon_season` | 1 (Jun-Oct) | 191,250 | 41.8857 % |

#### The finding: there is no rare category and no high-cardinality problem

This is the item's result, and it is a *negative* one, stated plainly:

- **The panel is balanced, so the level sizes are identical by construction.** Every one of the
  50 stations has exactly **9,132 rows (2.0000 %)** — the min, the max and the only distinct
  value of the per-station row count. Every one of the 10 basins has exactly **45,660 rows
  (10.0000 %)**, and every basin holds exactly 5 stations. `country` is 20.0000 % (91,320) or
  10.0000 % (45,660), nothing else.
- **The calendar columns are near-uniform**: `season` spans **24.7153 % (Winter) to 25.1862 %
  (Spring/Summer)**, a max/min ratio of just **1.02×**; `monsoon_season` splits **41.8857 % /
  58.1143 %** (1.39×) — the widest split of the two calendar columns, and still nowhere near a
  rarity threshold.
- **The smallest level share anywhere in the dataset is 2.0000 %** — the per-station share,
  9,132 rows (and identically the per-`station_name` share). Nothing is starved: the rarest level
  in the categorical columns is still **2×** the 1 % threshold that would trigger a "rare
  category" treatment.
- **The high-cardinality trigger fires twice and is harmless both times.** `station_id` and
  `station_name` have **k = 50**, above the usual k > 20 heuristic — but their levels are not
  *rare*, there are simply many *equal* ones. One-hot encoding both would add 50 + 50 = **100
  columns that carry no information**; one is redundant (below) and neither is a model input.
- **Do not confuse this with the class imbalance.** The imbalance belongs to the *target* —
  **9,889 positives in 456,550 usable rows (2.1660 %), 45.2 : 1** (item 1C) — while the
  categorical *levels* run from 2.0000 % to 58.1143 %. There is nothing to collapse and no long
  tail to bucket: with every level holding at least 2.0000 % of the rows, the
  `handle_unknown="ignore"` strategy prescribed in item 4A cannot silently zero out a meaningful
  share of the test set, and the one-hot width it protects is 14 columns.

![Figure 2B.1](figures/fig20_category_frequencies.png)

**Figure 2B.1.** Level frequency of every categorical column (456,600 rows, log count axis; each
bar is labelled with its count and its share of rows, and the two 50-bar identifier panels annotate
the single flat value instead; `severity_level` is hatched and flagged as the leaky column). The
three flat panels (`station_id`, `station_name`, `basin`) are the balance result: 50 × 9,132,
50 × 9,132 and 10 × 45,660, with no visible spread at all. The only panel with a skewed profile is
`severity_level`, and that column is dropped for leakage — its shape is a property of the
risk-score binning, not a signal to encode.

![Figure 2B.2](figures/fig20_cardinality.png)

**Figure 2B.2.** Cardinality (A) and the share of rows held by the least and most frequent level
(B), per column. Panel B is the rare-category screen in one picture: the red dashed line is the
1 % threshold, and **every dumbbell sits far to its right** — the closest approach is
`severity_level`'s 2.17 %. The marker pairs collapse onto a single point for `station_id`,
`station_name` and `basin` (max/min = 1.00×), which is the visual signature of a balanced panel;
the widest spread in the figure, `severity_level` at 23.61×, belongs to the leaky column.

#### The one exception, and why it is not a rare-category problem

`severity_level = "Extreme"` holds **9,921 rows (2.1728 %)** — the smallest share of any level
outside the two station-identity columns, and the only level anywhere near the 1 % mark. It is
**not** handled as a rare category, because the column is not a feature at all: `severity_level`
is a deterministic binning of `flood_risk_score`, and item 1B drops it outright for leakage (the
script prints the drop reason straight out of `data.DROPPED`). Its skewed profile — 51.2979 % Low
against 2.1728 % Extreme, a 23.61× max/min ratio — therefore never reaches the model. Excluding
`severity_level` as well, the smallest level share in the dataset is 10.0000 % (any basin).

#### Redundancy and re-encodings

!!! note "`station_name` is 1:1 with `station_id` — drop one"
    Both columns have exactly **50 levels**, and the pair `(station_id, station_name)` has exactly
    **50 distinct combinations** over 456,600 rows (max 1 name per id). The mapping is a bijection,
    so `station_name` adds zero information and is dropped; `station_id` is kept only as a grouping
    key (the *t+1* target shift and the per-station window features group by it) and is not a model
    input either.

!!! note "`country` is a function of `basin` — redundant"
    All **10 basins map to exactly one country** (max countries per basin = 1). The crosstab of
    `basin` × `country` is a permutation matrix with a single non-zero cell per row (45,660 rows
    in each): Brahmaputra and Ganges → India; Pearl and Yangtze → China; Irrawaddy and Salween →
    Myanmar; Chao Phraya → Thailand; Mekong → Vietnam; Indus → Pakistan; Helmand → Afghanistan.
    `country` therefore carries nothing that `basin` does not, and with 7 levels against 10 it is
    not even a compression. It is dropped.

!!! note "`season` and `monsoon_season` are deterministic re-encodings of `month`"
    `season` = Mar-May Spring, Jun-Aug Summer, Sep-Nov Autumn, Dec-Feb Winter — **0 mismatches**
    over 456,600 rows. `monsoon_season` = 1 for `month ∈ {6,7,8,9,10}`, else 0 — **0 mismatches**
    over 456,600 rows. The crosstabs prove both: `season` × `month` has exactly three non-zero
    columns per season row, and `month` × `monsoon_season` is a clean block (months 6-10 all
    monsoon, months 1-5 and 11-12 all zero).

    | `season` | months | rows per month |
    |---|---|---|
    | Spring | 3, 4, 5 | 38,750 / 37,500 / 38,750 |
    | Summer | 6, 7, 8 | 37,500 / 38,750 / 38,750 |
    | Autumn | 9, 10, 11 | 37,500 / 38,750 / 37,500 |
    | Winter | 12, 1, 2 | 38,750 / 38,750 / 35,350 |

    They are kept anyway — not as new information but as *explicit gates*: `month` is not a model
    input in `NUMERIC`, so these two columns are how the calendar reaches the model at all,
    in place of an inverted `sin/cos(day_of_year)` pair. `monsoon_season` is kept as the 0/1
    numerical flag; `season` is the one to be one-hot encoded.

!!! note "`latitude` and `longitude` are station constants, not measurements"
    Each of the 50 stations has exactly **1 distinct latitude and 1 distinct longitude**, and
    `(station_name, latitude, longitude)` has only **50 distinct triples over 456,600 rows**. They
    are static attributes repeated on every row — treated as identifiers, and dropped as features
    in item 1B.

#### What survives into the model

`CATEGORICAL = ["basin", "season"]` → **14 one-hot indicator columns** (basin 10 + season 4). They
are the only two columns that are both non-leaky and not implied by another column.

Everything else categorical is dropped, each for a reason established in item 1B rather than
re-derived here: `severity_level` (leakage — `cut(flood_risk_score, [1,3,5,7,10.01], right=False)` reproduces it on all 456,600 rows
on all 456,600 rows), `station_name` (1:1 with `station_id`), `country` (a function of `basin`),
`latitude`/`longitude` (constants per station). `station_id` survives as a grouping key only;
`monsoon_season` survives as a numerical 0/1 flag.

`basin` is kept: 10 indicator columns on a balanced 10-level column are cheap, and whether
geography earns its place is decided by item 3B's rates and item 4A's ablation — not by this
section, which only establishes that the levels are well populated.

```python
--8<-- "docs/projects/eda/code/t3_categorical.py"
```

## 3. Bivariate and multivariate analysis

Everything below is **descriptive statistics on the full usable panel** — 456,550 station-days
(50 stations × 9,131 days with a defined *t+1*), 9,889 positives, base rate **2.166 %**,
imbalance **45.17 : 1**. The label is `flood_next_day` = `flood_event_occurred` shifted one day
forward per station. **No model is trained**; `roc_auc_score` is applied to single raw columns,
which is a rank statistic, not a fit. No transformer is fitted on these numbers either — every
preprocessing statistic in stage 4 is fitted on train only.

### A. Numerical × numerical

#### Why Spearman is the primary method

The four measurements are not jointly elliptical, so the choice of coefficient is a decision
that has to be argued, not defaulted:

| measurement | skew | excess kurtosis | max |
|---|---|---|---|
| `rainfall_mm` | **10.5127** | **158.1862** | 1,775.72 mm/day |
| `river_level_m` | **9.5313** | **140.5934** | 204.67 m |
| `soil_moisture_percent` | 0.8642 | 0.8563 | 100.00 % (censored) |
| `temperature_celsius` | 0.0057 | −1.1964 | 45.00 °C |

Pearson is a *leverage* statistic: on a variable with skew 10.5 and kurtosis 158, a few extreme
monsoon days can supply most of the covariance. Spearman works on ranks, is invariant to any
monotone transform (`log1p` included), and answers the question the model actually cares about —
"do the two variables move together?" rather than "how well does one straight line through two
explosive tails fit?". **We therefore report Spearman as the primary coefficient and Pearson
beside it, and the gap between them is the evidence.**

**Table 3A.1 — the six pairs among the four measurements (n = 456,550).**

| pair | Pearson | Spearman | gap (P − S) |
|---|---|---|---|
| `rainfall_mm` ↔ `river_level_m` | **0.9604** | **0.8689** | +0.0915 |
| `rainfall_mm` ↔ `soil_moisture_percent` | 0.6103 | 0.8455 | −0.2353 |
| `rainfall_mm` ↔ `temperature_celsius` | 0.3082 | 0.6688 | −0.3607 |
| `river_level_m` ↔ `soil_moisture_percent` | 0.6652 | 0.8618 | −0.1966 |
| `river_level_m` ↔ `temperature_celsius` | 0.3427 | 0.6188 | −0.2761 |
| `soil_moisture_percent` ↔ `temperature_celsius` | 0.7197 | 0.7500 | −0.0303 |

Only one of the six pairs has Pearson *above* Spearman, and it is the one everybody quotes. For
the other five Pearson is lower by up to 0.36 — the heavy tail destroys the linear fit.

The decisive test is to remove the tail and watch what happens to each coefficient:

**Table 3A.2 — trimming test for `rainfall_mm` ↔ `river_level_m`.**

| subset | n | % of panel | Pearson | Pearson on `log1p` | Spearman |
|---|---|---|---|---|---|
| all rows | 456,550 | 100.00 % | 0.9604 | 0.9054 | 0.8689 |
| all rows, rainfall ≤ 150 mm | 449,748 | 98.51 % | 0.9098 | 0.8850 | 0.8628 |
| all rows, rainfall ≤ 100 mm | 447,547 | 98.03 % | 0.8927 | 0.8798 | 0.8608 |
| monsoon only | 191,250 | 41.89 % | **0.9632** | 0.9149 | **0.6110** |
| monsoon only, rainfall ≤ 150 mm | 184,448 | 40.40 % | 0.8380 | 0.7160 | 0.5663 |
| monsoon only, rainfall ≤ 100 mm | 182,247 | 39.92 % | **0.6072** | 0.5734 | **0.5504** |
| outside monsoon | 265,300 | 58.11 % | 0.5353 | 0.5351 | 0.5468 |

!!! note "Conclusion — Spearman is the justified method"
    Inside the monsoon, deleting the **4.71 %** of days with rainfall above 100 mm collapses
    Pearson from **0.9632 to 0.6072** — onto the Spearman value **0.6110** (Spearman itself
    moves only to 0.5504). The 0.96 was never a description of the joint distribution; it was
    the leverage of a few thousand extreme days. Outside the monsoon, where there is no such
    tail, Pearson (0.5353) and Spearman (0.5468) already agree. **Spearman is reported as the
    headline coefficient throughout; Pearson is kept only as the diagnostic that exposes the
    tail.**

#### The correlation matrix

![Figure 3A.1](figures/fig30_corr_heatmap.png)

*Figure 3A.1 — Correlation matrix of the numeric columns: Pearson above the diagonal, Spearman
below, diverging scale centred at 0, diagonal masked. n = 456,550 station-days.*

**Conclusion (Figure 3A.1): the matrix separates into one tight water block
(`rainfall_mm`, `river_level_m`, `soil_moisture_percent`, `flood_risk_score`), one calendar
block (`day_of_year` ↔ `month` = 0.9965) and a dead block (`latitude`, `longitude`,
`data_quality_score`, all |r| ≤ 0.49 but with no relation to the target).**

#### Redundant pairs

**Table 3A.3 — second-order redundancies and seasonality confounders (Pearson / Spearman).**

| pair | Pearson | Spearman | reading |
|---|---|---|---|
| `flood_risk_score` ↔ `soil_moisture_percent` | +0.9337 | +0.8781 | the risk score is redundant, not informative (stage 1B) |
| `flood_risk_score` ↔ `river_level_m` | +0.6643 | +0.9146 | same, by rank |
| `flood_risk_score` ↔ `rainfall_mm` | +0.5949 | +0.8908 | same, by rank |
| `month` ↔ `day_of_year` | +0.9965 | +0.9965 | one date re-encoded twice — keep one |
| `temperature_celsius` ↔ `day_of_year` | +0.7211 | +0.7167 | seasonality |
| `soil_moisture_percent` ↔ `day_of_year` | +0.4549 | +0.5261 | seasonality |
| `latitude` ↔ `longitude` | −0.4945 | −0.4478 | only 50 distinct coordinate pairs |
| `rainfall_mm` ↔ `day_of_year` | +0.1592 | +0.3722 | weak — the monsoon is a *gate*, not a gradient |
| `river_level_m` ↔ `day_of_year` | +0.1876 | +0.3804 | weak, same reason |
| `data_quality_score` ↔ everything | max \|r\| = **0.0036** | — | noise, dropped |
| `year` ↔ everything | max \|r\| = **0.0369** | — | no drift → the chronological split is fair |

!!! warning "The redundant pair — results-table row 7"
    **`rainfall_mm` ↔ `river_level_m`: Pearson 0.9604, Spearman 0.8689.** This is the most
    correlated numerical pair in the dataset and the answer to results-table row 7. The river
    level is, to a first approximation, a rescaled rainfall.

    Second-order: `soil_moisture_percent` ↔ `temperature_celsius` (Spearman **0.7500**) is a
    **seasonal artefact**, not a mechanism — both track `day_of_year` (r = **0.7211** for
    temperature, **0.4549** for soil moisture), so the pair is confounding. `flood_risk_score`
    ↔ `soil_moisture_percent` = **0.9337** is why the risk score is redundant rather than
    informative (stage 1B). `month` ↔ `day_of_year` = **0.9965** — keep one.

#### Why both members of the redundant pair stay in the feature set

Dropping a column on correlation alone is only correct if the two columns carry the same
information **about the label**. They do not:

| single-column predictor | ROC-AUC against the next-day label |
|---|---|
| `soil_moisture_percent` | **0.8364** |
| `rainfall_mm` | **0.8091** |
| `river_level_m` | **0.7948** |
| `temperature_celsius` | 0.7956 |
| `flood_risk_score` *(excluded, leaky/redundant)* | 0.8088 |
| `monsoon_season` | 0.7969 |
| `day_of_year` | 0.6313 |

The three water variables rank differently as next-day predictors than they correlate with each
other, and a river level physically lags rainfall (rain today, high water tomorrow). **Decision:
keep `rainfall_mm` and `river_level_m` both; let the ablation in the modelling phase decide.**
Correlation between features is not a reason to delete a feature — it is a reason to test it.

#### Point-biserial correlation with the next-day target

**Table 3A.4 — point-biserial r with `flood_next_day` for every numeric column (n = 456,550).**

| column | r | | column | r |
|---|---|---|---|---|
| `soil_moisture_percent` | **+0.1771** | | `flood_event_occurred` (t) | +0.0558 |
| `monsoon_season` | **+0.1752** | | `latitude` | −0.0101 |
| `flood_risk_score` *(dropped, leaky)* | +0.1623 | | `longitude` | +0.0049 |
| `temperature_celsius` | +0.1497 | | `year` | +0.0046 |
| `river_level_m` | +0.0946 | | `data_quality_score` | +0.0036 |
| `rainfall_mm` | +0.0850 | | `day` | +0.0007 |
| `month` | +0.0666 | | | |
| `day_of_year` | +0.0662 | | | |

!!! note "The most interesting sentence in this section: the ranking inverts"
    **Today's soil moisture (0.1771) and today's temperature (0.1497) matter more for tomorrow
    than today's rainfall (0.0850) does** — and today's rainfall outranks the day's own flood
    label at *t* (0.0558). Rain is a **same-day trigger**; soil moisture is the **standing
    precondition** that persists into tomorrow. A model built on "how much did it rain today"
    is looking at the wrong variable; the accumulated state is what carries over.

#### Rainfall × river level on the raw scale

![Figure 3A.2](figures/fig31_scatter_rain_river.png)

*Figure 3A.2 — `rainfall_mm` × `river_level_m`, coloured by the next-day label. All 9,889
positives are plotted; the 446,661 negatives are represented by a 30,000-row random sample
(6.7 %, `random_state=42`) for legibility → 39,889 points, alpha 0.25. The dotted line is the
stage 2A IQR fence at 56.30 mm.*

**Conclusion (Figure 3A.2): on the raw scale the two variables trace one straight line, but the
mass of the panel is collapsed into the bottom-left corner — 99 % of station-days are below
209 mm/day — and the two classes overlap almost completely in the region where the data
actually lives; the apparent line is drawn by the shared extreme tail.**

#### The same pair on `log1p` axes

![Figure 3A.3](figures/fig32_scatter_log1p.png)

*Figure 3A.3 — the same 39,889 points on `log1p` axes. The monsoon-band statistics in the box
are computed on the full panel, not on the sample.*

**Conclusion (Figure 3A.3): `log1p` does not turn the pair into a single linear band — it
reveals that the panel is a mixture of two disjoint regimes (dry season 2.02–41.39 mm, monsoon
16.72–1,775.72 mm) separated by a hole in which only 14 of 456,550 station-days sit, and that
within the monsoon the linear correlation lives entirely in the top ~5 % of days.**

Two structural facts behind that figure, both of which the modelling phase has to respect:

- **`rainfall_mm` is bimodal with a hole, not a tail.** Outside the monsoon the maximum is
  41.39 mm (99.9th percentile 26.32); inside it the 1st percentile is 17.48 mm and the maximum
  is 1,775.72 mm. Between 50 mm and 84 mm there are **14 rows out of 456,550 (0.0031 %)**.
  `log1p` compresses the tail but cannot remove a gap.
- **The next-day flood rate is not monotone in rainfall.** Deciles of the full panel:

| rainfall decile | n | floods | next-day flood rate |
|---|---|---|---|
| (10.76, 18.70] mm | 45,616 | 32 | 0.070 % |
| (18.70, 23.63] mm | 45,675 | 1,633 | **3.575 %** |
| (23.63, 28.52] mm | 45,668 | 3,491 | **7.644 %** ← peak |
| (28.52, 33.22] mm | 45,667 | 2,932 | 6.420 % |
| (33.22, 1775.72] mm | 45,589 | 1,800 | **3.948 %** ← *lower* than the 8th decile |

  The biggest storms are **not** the best predictor of a flood tomorrow: the top decile of
  rainfall carries a 3.95 % next-day flood rate against 7.64 % for the 8th. This is exactly why
  `rainfall_mm` has a point-biserial r of only 0.0850 despite a single-column ROC-AUC of 0.8091
  — a monotone coefficient cannot describe a non-monotone relation, and a rank statistic can.

### B. Categorical × target

Every rate below is computed against the **next-day** label `flood_next_day`, with the row count
`n` per level so that thin cells are visible. Overall mean: **2.1660 %** (9,889 / 456,550).

**Table 3B.1 — flood rate by month.**

| month | n | floods | rate (%) | share of all floods (%) |
|---|---|---|---|---|
| Jan | 38,750 | 0 | 0.000 | 0.00 |
| Feb | 35,350 | 0 | 0.000 | 0.00 |
| Mar | 38,750 | 0 | 0.000 | 0.00 |
| Apr | 37,500 | 0 | 0.000 | 0.00 |
| May | 38,750 | 1 | 0.003 | 0.01 |
| Jun | 37,500 | 92 | 0.245 | 0.93 |
| **Jul** | 38,750 | 3,149 | **8.126** | 31.84 |
| **Aug** | 38,750 | 3,132 | **8.083** | 31.67 |
| **Sep** | 37,500 | 3,045 | **8.120** | 30.79 |
| Oct | 38,750 | 470 | 1.213 | 4.75 |
| Nov | 37,500 | 0 | 0.000 | 0.00 |
| Dec | 38,700 | 0 | 0.000 | 0.00 |

**Table 3B.2 — flood rate by `monsoon_season` and by `season`.**

| grouping | level | n | floods | rate (%) |
|---|---|---|---|---|
| `monsoon_season` | 1 (Jun–Oct) | 191,250 | 9,888 | **5.1702** |
| `monsoon_season` | 0 | 265,300 | 1 | **0.00038** |
| `season` | Summer (Jun–Aug) | 115,000 | 6,373 | 5.5417 |
| `season` | Autumn (Sep–Nov) | 113,750 | 3,515 | 3.0901 |
| `season` | Spring (Mar–May) | 115,000 | 1 | 0.0009 |
| `season` | Winter (Dec–Feb) | 112,800 | 0 | 0.0000 |

**Table 3B.3 — flood rate by `basin` (rates differ by 1.27×; 5 stations / 45,655 rows each).**

| basin | country | n | floods | rate (%) |
|---|---|---|---|---|
| Chao Phraya | Thailand | 45,655 | 1,128 | **2.4707** |
| Mekong | Vietnam | 45,655 | 1,093 | 2.3940 |
| Pearl | China | 45,655 | 1,020 | 2.2341 |
| Irrawaddy | Myanmar | 45,655 | 1,013 | 2.2188 |
| Salween | Myanmar | 45,655 | 994 | 2.1772 |
| Ganges | India | 45,655 | 979 | 2.1443 |
| Indus | Pakistan | 45,655 | 929 | 2.0348 |
| Helmand | Afghanistan | 45,655 | 925 | 2.0261 |
| Yangtze | China | 45,655 | 919 | 2.0129 |
| Brahmaputra | India | 45,655 | 889 | **1.9472** |

**Table 3B.4 — flood rate by `country` (pooled basins).**

| country | n | floods | rate (%) |
|---|---|---|---|
| Thailand | 45,655 | 1,128 | 2.4707 |
| Vietnam | 45,655 | 1,093 | 2.3940 |
| Myanmar | 91,310 | 2,007 | 2.1980 |
| China | 91,310 | 1,939 | 2.1235 |
| India | 91,310 | 1,868 | 2.0458 |
| Pakistan | 45,655 | 929 | 2.0348 |
| Afghanistan | 45,655 | 925 | 2.0261 |

`country` adds nothing to `basin`: it is a function of it (stage 1B), and the pooled rates simply
reproduce the basin rates.

#### Flood rate by month

![Figure 3B.1](figures/fig33_flood_rate_month.png)

*Figure 3B.1 — next-day flood rate by month against the 2.166 % overall mean, monsoon window
(Jun–Oct) shaded; `n` and the number of positives per month are in the tick labels.*

**Conclusion (Figure 3B.1): the monsoon gate is the whole story — six months have exactly zero
next-day floods, May and June have one and 92, and Jul/Aug/Sep sit at 8.13 %, 8.08 % and 8.12 %
against a 2.166 % overall mean, holding 94.31 % of every flood in the dataset.**

#### Flood rate by basin

![Figure 3B.2](figures/fig34_flood_rate_basin.png)

*Figure 3B.2 — next-day flood rate by basin, sorted, with the overall mean as a reference line
and `n` inside each bar.*

**Conclusion (Figure 3B.2): basin rates span only 1.947 % (Brahmaputra) to 2.471 %
(Chao Phraya) — a 1.27× ratio around a 2.166 % mean, i.e. a null result. Geography is nearly
worthless as a predictor here; `basin` is worth keeping only as a cheap static context
variable.**

#### Flood rate by station

![Figure 3B.3](figures/fig35_flood_rate_station.png)

*Figure 3B.3 — next-day flood rate for all 50 stations, sorted, same reference line. The panel
is balanced: every station carries exactly 9,131 rows.*

**Conclusion (Figure 3B.3): the station spread is 1.632 % (Pandu) to 2.585 % (Bangkok) — a
1.58× ratio, std 0.207 pp across 50 levels — and because no cell is thin (n = 9,131 everywhere)
this spread is not a small-sample artefact but a genuinely small effect.**

??? note "Full station table (n and floods per station, next-day label)"
    | station | name | n | floods | rate (%) |
    |---|---|---|---|---|
    | ASIA_0049 | Bangkok | 9,131 | 236 | 2.5846 |
    | ASIA_0050 | Sing Buri | 9,131 | 232 | 2.5408 |
    | ASIA_0010 | Chau Doc | 9,131 | 229 | 2.5079 |
    | ASIA_0006 | Chiang Saen | 9,131 | 226 | 2.4751 |
    | ASIA_0048 | Ayutthaya | 9,131 | 224 | 2.4532 |
    | ASIA_0047 | Chai Nat | 9,131 | 223 | 2.4422 |
    | ASIA_0009 | Phnom Penh | 9,131 | 218 | 2.3875 |
    | ASIA_0026 | Mandalay | 9,131 | 217 | 2.3765 |
    | ASIA_0038 | Guangzhou | 9,131 | 216 | 2.3656 |
    | ASIA_0032 | Mawkmai | 9,131 | 215 | 2.3546 |
    | ASIA_0007 | Luang Prabang | 9,131 | 214 | 2.3437 |
    | ASIA_0046 | Nakhon Sawan | 9,131 | 213 | 2.3327 |
    | ASIA_0036 | Wuzhou | 9,131 | 212 | 2.3218 |
    | ASIA_0039 | Zhaoqing | 9,131 | 211 | 2.3108 |
    | ASIA_0005 | Farakka | 9,131 | 210 | 2.2999 |
    | ASIA_0019 | Hyderabad | 9,131 | 210 | 2.2999 |
    | ASIA_0028 | Bhamo | 9,131 | 208 | 2.2780 |
    | ASIA_0008 | Vientiane | 9,131 | 206 | 2.2561 |
    | ASIA_0001 | Varanasi | 9,131 | 203 | 2.2232 |
    | ASIA_0030 | Rangoon | 9,131 | 202 | 2.2122 |
    | ASIA_0041 | Kajaki | 9,131 | 202 | 2.2122 |
    | ASIA_0022 | Guwahati | 9,131 | 201 | 2.2013 |
    | ASIA_0031 | Mae Sam Laep | 9,131 | 199 | 2.1794 |
    | ASIA_0035 | Nyaunglebin | 9,131 | 199 | 2.1794 |
    | ASIA_0013 | Nanjing | 9,131 | 199 | 2.1794 |
    | ASIA_0029 | Prome | 9,131 | 199 | 2.1794 |
    | ASIA_0042 | Grishk | 9,131 | 198 | 2.1684 |
    | ASIA_0018 | Kotri | 9,131 | 197 | 2.1575 |
    | ASIA_0003 | Kanpur | 9,131 | 195 | 2.1356 |
    | ASIA_0033 | Pyinmana | 9,131 | 194 | 2.1246 |
    | ASIA_0040 | Guiping | 9,131 | 193 | 2.1137 |
    | ASIA_0004 | Allahabad | 9,131 | 193 | 2.1137 |
    | ASIA_0023 | Tezpur | 9,131 | 189 | 2.0699 |
    | ASIA_0037 | Nanning | 9,131 | 188 | 2.0589 |
    | ASIA_0034 | Taungoo | 9,131 | 187 | 2.0480 |
    | ASIA_0027 | Sagaing | 9,131 | 187 | 2.0480 |
    | ASIA_0015 | Chongqing | 9,131 | 186 | 2.0370 |
    | ASIA_0021 | Dhubri | 9,131 | 181 | 1.9823 |
    | ASIA_0011 | Yichang | 9,131 | 181 | 1.9823 |
    | ASIA_0002 | Patna | 9,131 | 178 | 1.9494 |
    | ASIA_0020 | Attock | 9,131 | 178 | 1.9494 |
    | ASIA_0045 | Kandahar | 9,131 | 178 | 1.9494 |
    | ASIA_0012 | Wuhan | 9,131 | 177 | 1.9385 |
    | ASIA_0014 | Shanghai | 9,131 | 176 | 1.9275 |
    | ASIA_0017 | Sukkur | 9,131 | 176 | 1.9275 |
    | ASIA_0043 | Gereshk | 9,131 | 174 | 1.9056 |
    | ASIA_0044 | Lashkargah | 9,131 | 173 | 1.8946 |
    | ASIA_0024 | Dibrugarh | 9,131 | 169 | 1.8508 |
    | ASIA_0016 | Tarbela | 9,131 | 168 | 1.8399 |
    | ASIA_0025 | Pandu | 9,131 | 149 | 1.6318 |

!!! warning "Honest headline for 3B"
    **Geography is almost worthless here.** Basin rates span 1.27× and station rates 1.58×
    against a 2.166 % mean; the vulnerability table (stage 1B) has almost no signal left to
    explain. **The only categorical that matters is the monsoon gate**: 5.1702 % inside Jun–Oct
    against 0.00038 % outside — 1 flood in 265,300 rows — a ratio of about 13,700×. A
    "never flood outside the monsoon" rule is free and must be part of every baseline in the
    modelling phase; every metric has to be reported **inside** the window, where the problem is
    a 5.2 % base rate, not 2.2 %.

### C. Numerical × categorical

#### Location *and* spread, by monsoon flag

**Table 3C.1 — the four measurements by `monsoon_season` (n = 265,300 outside, 191,250 inside).**

| measurement | mean outside | mean inside | location × | std outside | std inside | spread × |
|---|---|---|---|---|---|---|
| `rainfall_mm` | 7.0954 | **38.5753** | **5.44×** | 3.6109 | **60.0142** | **16.62×** |
| `river_level_m` | 3.4795 | 8.4719 | 2.43× | 0.9981 | 7.7725 | 7.79× |
| `soil_moisture_percent` | 23.6092 | 55.4311 | 2.35× | 6.5610 | 11.7580 | 1.79× |
| `temperature_celsius` | 18.3832 | 32.9699 | 1.79× | 8.2453 | 6.6739 | 0.81× |

#### Rainfall by monsoon, raw and `log1p`

![Figure 3C.1](figures/fig36_rainfall_monsoon.png)

*Figure 3C.1 — rainfall by `monsoon_season`, raw scale (left) and `log1p` (right), with the
mean and standard deviation annotated in both panels under the tick labels.*

**Conclusion (Figure 3C.1): the monsoon move is not only a shift — location moves 5.44×
(7.095 → 38.575 mm) but spread moves 16.62× (std 3.611 → 60.014 mm), so the raw scale puts a
tiny dry-season box beside a monsoon box whose tail reaches 1,775.72 mm; after `log1p` the
spread ratio falls to 1.24× (std 0.407 → 0.503) while the location shift survives (2.005 →
3.430, 1.71×).**

!!! note "Consequence for stage 4A — this is why `log1p` comes before `StandardScaler`"
    A plain `StandardScaler` fits one mean and one standard deviation per feature. Under a
    **16.62×** change in spread it would divide the entire dry season by a number determined by
    the monsoon tail, compressing the region where 58.11 % of the data lives into a sliver —
    and the destination is a neural network, where that kills the gradient signal.
    **`log1p` first, then `StandardScaler`**: the transform brings the two regimes within a
    1.24× spread ratio and *then* a single scaler is defensible.

#### Location *and* spread, by basin

**Table 3C.2 — the four measurements by `basin` (n = 45,655 each).**

| measurement | mean range | location ratio | std range | spread ratio |
|---|---|---|---|---|
| `rainfall_mm` | 19.3523 – 21.6050 mm/day | **1.12×** | 38.2112 – 45.7781 | 1.20× |
| `river_level_m` | 5.3154 – 5.9044 m | **1.11×** | 5.1884 – 6.1407 | 1.18× |
| `soil_moisture_percent` | 36.5508 – 37.5431 % | **1.03×** | 17.7253 – 18.7213 | 1.06× |
| `temperature_celsius` | 19.7557 – 29.3387 °C | 1.49× | 10.0192 – 10.0669 | 1.0048× |

Within-basin rainfall std (≈ 42 mm) dwarfs the entire across-basin spread of the basin
means (19.35 → 21.61 mm). A one-way decomposition of rainfall variance:

| grouping | η² = SS_between / SS_total | MS_between | MS_within | var of the group means vs mean within-group var |
|---|---|---|---|---|
| `station_id` (50 levels) | **0.0414 %** | 6,787.34 | 1,757.02 | 0.7285 mm² vs 1,757.02 mm² (**0.0415 %**) |
| `basin` (10 levels) | **0.0337 %** | 30,024.13 | 1,757.01 | 0.5919 mm² vs 1,757.01 mm² (0.0337 %) |

#### The four measurements by basin

![Figure 3C.2](figures/fig37_measurements_basin.png)

*Figure 3C.2 — the four measurements by basin, 2×2 grid; log scale for `rainfall_mm` and
`river_level_m`. Box = IQR, whiskers = 1.5 × IQR, dots = outliers, diamonds = means.
n = 456,550.*

**Conclusion (Figure 3C.2): for the three water variables the 10 basins are visually
indistinguishable in both location and spread — the boxes sit on top of each other, the standard
deviations differ by at most 1.20×, and the only measurable difference is
`temperature_celsius` (means 19.76 → 29.34 °C, 1.49×), which is a latitude effect and not a
hydrological one.**

#### River level by basin

![Figure 3C.3](figures/fig38_river_level_basin.png)

*Figure 3C.3 — river level by basin on a log scale, sorted by mean; per-basin mean and standard
deviation are in the tick labels.*

**Conclusion (Figure 3C.3): every basin spans the same range with the same IQR and the same
tail — location varies by only 1.11× (5.315 → 5.904 m) and spread by 1.18× (std 5.188 →
6.141 m) — so `basin` carries essentially no information about river level, exactly the null
result the rainfall decomposition predicts.**

!!! warning "Consequence — do not oversell station-level features"
    Between-station variance is **0.0414 %** of total rainfall variance. Station fixed effects
    and the per-station z-score anomalies planned for the modelling phase will therefore mostly
    encode **season**, not local character: a per-station mean is a 9,132-day average dominated
    by how many monsoon days that station has, and every station has the same 9,132 days. They
    are still cheap and harmless as context, but they must not be presented as capturing
    catchment-specific behaviour. The signal has to come from the **temporal** structure of the
    four measurements inside the monsoon window.

**Script.** Standalone and deterministic; regenerates the nine figures above and prints every
number cited in this section.

```python
--8<-- "docs/projects/eda/code/t4_bivariate.py"
```

## 4. Preprocessing

### A. Strategies

Four strategies, each chosen because of a finding in stages 1–3 — never because it is the
default. Every statistic below is fitted on the **training window only**: train 2000–2016
(310,500 rows), validation 2017–2020 (73,050), test 2021–2024 (73,000 usable). The split is
**chronological, by date, identical dates for every station**, because rows one day apart are
near-duplicates and the label is 45.2 : 1, so a random split would leak flood days across the
boundary (stage 1D). Validation and test are only ever *transformed*.

#### Table 4A.1 — the four strategies and the finding behind each

| Strategy | Decision | Finding that motivates it |
|---|---|---|
| **Missing values** | `SimpleImputer(strategy="median")` on the numeric branch and `SimpleImputer(strategy="most_frequent")` on the categorical branch. **0 cells are actually imputed.** | 1B: **0 missing values in all 21 columns (0.00 %)**; recounted here as **0 of 9,588,600 cells** |
| **Outliers** | **Never drop, never winsorise** (0 rows removed, 0 clipped). `log1p` compress the two heavy tails + an explicit `is_extreme_rainfall` flag. | **100.000 % of the 9,274 rows above the `rainfall_mm` 1.5×IQR fence (56.30 mm) are floods**, against **0.137 %** below it — an IQR filter would delete the entire positive class |
| **Categorical encoding** | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` on `basin` (10 levels) and `season` (4) → 14 columns. | 2B: **no rare category exists to collapse** — every station is exactly 2.0000 % of the panel, every basin exactly 10.0000 %, every season 24.7070–25.1889 % |
| **Scaling** | `log1p` **first** on `rainfall_mm` and `river_level_m`, then `StandardScaler` on all 6 numerics. The two hard ceilings are kept, never repaired. | 2A/3C: skew **10.5132** / **9.5317**, excess kurtosis **158.2017** / **140.6058**; monsoon rainfall spread is **16.62×** the off-monsoon spread; the four measurements span 0.39–1,775.72 across four units; **two of the four are right-censored** (`soil_moisture_percent` at 100.00 %, `temperature_celsius` at 45.00 °C); and the destination is a neural network |

---

#### A.1 Missing values — the honest finding is that there is none

!!! note "There is nothing to impute, and inventing a missingness problem would be dishonest"
    The raw panel holds **0 missing cells in 456,600 × 21 = 9,588,600 (0.00 %)**, and the
    feature matrices hold **0** in train and **0** in test. The stage-1B missing-value table is
    21 rows of `0 (0.00 %)`. `SimpleImputer` therefore changes **no cell at all** — it is in the
    pipeline as a **robustness measure for a partial future day** (a station posting rainfall
    before its soil-moisture reading lands), not as a repair for an observed defect. Any claim
    that this dataset needed imputation would be false.

The imputers are still *fitted*, on train, and their fitted values are reported so the claim
is checkable: the numeric medians are `rainfall_mm` **10.72 mm**, `river_level_m` **4.42 m**,
`soil_moisture_percent` **30.97 %**, `temperature_celsius` **24.38 °C**, `day_of_year` **183**,
`monsoon_season` **0**, `is_extreme_rainfall` **0** — note that the two logged columns store
their median in the compressed space (`log1p(10.72) = 2.461297`).

The categorical imputer is not decoration either, and this was **measured** rather than
assumed. Handing the fitted encoder a missing `basin` on its own:

| representation of a missing category | encoder alone, no imputer |
|---|---|
| unseen string level (`"Atlantis"`, dtype `str`) | accepted → all-zero one-hot block |
| `None` in an `object` column | accepted → all-zero one-hot block |
| all-missing `float64` column | **`TypeError: '<' not supported between instances of 'float' and 'str'`** |

Imputing first removes both failure modes: the crash, and the *silent* all-zero encoding that
would make a missing basin look like a brand-new basin. When the guard fires it fills
`basin` with `"Brahmaputra"` and `season` with `"Spring"` — and the honest caveat is that
`most_frequent` is a **10-way tie** on this panel (every basin is exactly 10.0000 % of the
rows), so the basin fill value is arbitrary; `season` does have a real mode (25.1889 %).
A future day is then transformed normally: **0 NaN** in the output row, and the imputed
rainfall lands at **-0.162490 σ** — not 0, because the imputer fills in the train *median*
while the scaler subtracts the train *mean*.

#### A.2 Outliers — the "outliers" are the positive class

!!! warning "A textbook 'IQR-filter the outliers' step would delete every flood in the dataset"
    The 1.5×IQR fence on `rainfall_mm` is **56.2950 mm** (Q1 5.72, Q3 25.95, IQR 20.23; the
    **56.30 mm** quoted in stage 2A) and flags **9,274 rows (2.031 %)**. **100.000 % of them are
    floods**; among the remaining 447,326 rows the flood rate is **0.137 %**. The
    `river_level_m` fence (12.2350 m) flags 9,273 rows (2.031 %), and the union of the two
    fences flags **9,280 rows (2.032 %)**, again 100.000 % floods against 0.136 % elsewhere.
    These are not measurement errors to be cleaned — they *are* the event being predicted.
    **Rows removed: 0. Rows winsorised or clipped: 0.**

Three decisions replace filtering:

1. **`log1p` on the right tail** (A.4) so the extreme days cannot dominate a neural network's
   gradients. The transform is strictly monotone — Spearman(raw `rainfall_mm`,
   `num__rainfall_mm`) = **1.000000** — so every extreme day stays present **and ordered**;
   only its leverage on the loss changes.
2. **An explicit `is_extreme_rainfall` indicator**, produced *inside* the numeric branch by a
   stateless transformer (`ExtremeRainfallFlag`) that appends `1.0` where the **raw** rainfall
   exceeds `EXTREME_RAINFALL_MM = 56.30 mm`. It is applied in `fit` **and** in `transform`, so
   new data cannot silently arrive without it. After scaling it still takes exactly **2
   distinct values** (`-0.1433` / `6.9805`) and flags **2.0110 %** of train — i.e. `log1p` did
   not touch the flag, only the measurement.
3. **No row removal, ever.** Rows are the only place the positive class exists.

The threshold is a **declared constant, not a fitted statistic**, which is the leakage-free
direction. Two checks back that up: the full-panel fence reproduces the constant to
**5.00e-03 mm**, and a fence fitted on **train only** would be **55.89 mm** — yet the declared
constant and the train-only fence **disagree on 0 of 456,600 rows**. Pinning the stage-2A
value therefore costs nothing and keeps one threshold for all three splits.

!!! note "The honest number for the deliverable's own label"
    100.000 % is the **same-day** result (stage 2A/2B). For the one-day-ahead target this
    report actually predicts, the 6,244 flagged train rows carry a flood rate of **8.456 %**
    against **1.985 %** below the fence — a **4.26× lift**, not a certainty. That is the
    expectation to hold: extreme rainfall is a strong *feature*, not a label.

#### A.3 Categorical encoding — 14 dense columns, and a defined answer for unseen levels

`OneHotEncoder(handle_unknown="ignore", sparse_output=False)` is fitted on `basin` (10 levels)
and `season` (4): **14 output columns**. The concrete case the statement asks about — *a
category present in test but not in train* — was executed, not assumed:

| probe row | basin one-hot block | season one-hot block | NaN |
|---|:--:|:--:|:--:|
| `basin = "New Basin 2099"`, `season = "Monsoon-X"` (both unseen) | 0.0 | 0.0 | 0 |
| unseen season only, known basin | 1.0 | 0.0 | 0 |
| unseen basin only, known season | 0.0 | 1.0 | 0 |
| both levels known | 1.0 | 1.0 | 0 |

A level absent from the training window maps to **all zeros** — the row is still transformed,
nothing raises, and the *known* level in the same row still encodes. That is exactly what
`handle_unknown="ignore"` buys, and with 10 + 4 levels the extra width is trivial.

There is **no rare category to collapse**, which is why no frequency threshold appears:
stations 50 levels at **2.0000 %–2.0000 %**, basins 10 levels at **10.0000 %–10.0000 %**,
seasons 4 levels at **24.7070 %–25.1889 %**. On a dataset that *did* have them the strategy
would differ — a level covering a fraction of a percent contributes a near-constant column
that a network cannot learn from, so the right move would be to group infrequent levels
(`min_frequency` / `infrequent_if_exist`) or to replace the high-cardinality column with a
frequency or target encoding. Here full one-hot is both correct and cheapest.

`sparse_output=False` is deliberate: the destination is a neural network fed dense mini-batches,
so a SciPy sparse matrix would only have to be densified downstream. Ordinal encoding is
rejected because basins have no order, and target encoding is rejected as a fitted *supervised*
statistic — a leakage risk this pipeline does not need to take.

#### A.4 Scaling — `log1p` first, then standardise

The four measurements carry four different units (mm/day, m, %, °C) across roughly three
orders of magnitude (2A): the raw numeric inputs run from **-2.3100** to **1,775.7200** mm, on
the full panel. `rainfall_mm` has skew **10.5132** and excess kurtosis **158.2017**;
`river_level_m` **9.5317** and **140.6058**; `soil_moisture_percent` is **censored at exactly
100.00** (**9,153 rows, 2.005 %**); `temperature_celsius` is already symmetric (skew
**0.0057**, kurtosis **-1.1963**) and is left alone — 2A showed a shifted `log1p` would
*manufacture* a skew of -0.57 in it — but it is **also censored**, at exactly 45.00 °C
(**1,788 rows, 0.392 %**). (Those figures are full-panel, as stage 2A reports them; the table
below repeats the two skewed columns on the 310,500 training rows, which is what the pipeline
actually fits.)

!!! note "Two of the four measurements have a hard ceiling — keep it, do not repair it"
    `soil_moisture_percent` stops at exactly 100.00 (**9,153 rows, 2.005 %** of the panel) and
    `temperature_celsius` at exactly 45.00 °C (**1,788 rows, 0.392 %**). In both columns the
    maximum **is** the ceiling — nothing exceeds it — and 2A measured the pile-up against the
    local density: the 0.1-wide bin ending at 45.00 °C holds 1,967 rows against a mean of 224.3
    for the six bins below it (**~80×**), while soil moisture's ceiling bin carries **~68,648×**
    its local expectation. By contrast `rainfall_mm` and `river_level_m` are **not** censored:
    their maxima (1,775.72 mm, 204.67 m) are ordinary single-row order statistics, which is why
    the tail strategy and the ceiling strategy are different problems.

    The handling is identical to the tail's: **keep the ceiling, do not clip further, do not
    impute it away, do not drop the tied rows**. The reason is mechanical — a tie is a
    property no monotone transform can repair: thousands of rows share one identical value, so
    `log1p` and `StandardScaler` cannot spread them and the column keeps an **isolated spike**
    in the standardised distribution. What the transform does preserve is the *order*: the
    ceiling remains the top of its column, at **3.4919 σ** (soil moisture) and **1.9674 σ**
    (temperature). The ceiling is therefore something a model must be **told about**, not a
    defect to be repaired — the same logic as the `is_extreme_rainfall` flag for rainfall. And
    `temperature_celsius` must not be `log1p`-ed at all: it contains negatives
    (min -2.31 °C), where `log1p` is undefined, and a shifted log would manufacture a skew of
    **-0.57** (2A).

`log1p` on the two heavy tails, fitted on train, then `StandardScaler` on all six numerics:

| column | skew raw (train) | skew after `log1p` | kurtosis raw | kurtosis after `log1p` | after the pipeline |
|---|:--:|:--:|:--:|:--:|:--:|
| `rainfall_mm` | 10.5802 | **0.6837** | 160.9068 | **1.0746** | -1.7912 → 5.8685 σ |
| `river_level_m` | 9.5798 | **1.4888** | 142.6101 | **5.6453** | -2.9015 → 8.1045 σ |
| `soil_moisture_percent` | 0.8683 | unchanged | 0.8858 | unchanged | -2.0134 → 3.4919 σ (**the 100.00 ceiling**) |
| `temperature_celsius` | 0.0064 | unchanged | -1.1953 | unchanged | -2.5445 → 1.9674 σ (**the 45.00 °C ceiling**) |

**Why `log1p` must come before the standardiser**, in one number: inside the monsoon,
`rainfall_mm` has mean **38.5753 mm** and sd **60.0142 mm** (n = 191,250); outside it, mean
**7.0954 mm** and sd **3.6108 mm** (n = 265,350). The **location moves 5.44× but the spread
moves 16.62×** (on train alone: 38.1861 / 59.3126 against 7.0653 / 3.6012 — **5.40×** and
**16.47×**). A `StandardScaler` fitted on the pooled column divides everything by a single sd
of 41.4337 mm that the monsoon tail owns, which would squash the dry-season bulk into a narrow
band; compressing the tail first makes one shared scale meaningful. Both ceilings survive the
whole branch, as they must: **6,598 unique** raw soil-moisture values map to **6,598 unique**
transformed values, and the two maxima in the table above *are* the ceiling ties — soil
moisture's pile-up at **3.4919 σ** and temperature's at **1.9674 σ**.

**Why standardise at all:** the destination is a neural network, and gradient descent is scale
sensitive — leaving `temperature_celsius` in °C next to `monsoon_season` in 0/1 and rainfall in
mm makes the first layer's gradients a function of the unit system rather than the signal.
After the pipeline all **21** features sit in one comparable range, **-2.9015 σ to 8.1045 σ**.

#### Figure 4A.1 — before and after the numeric branch

![Figure 4A.1 — the four measurements before and after the numeric branch](figures/fig40_before_after_scaling.png)

**Figure 4A.1.** Columns = the four measurements, on the 310,500 training station-days. Top row
= **raw** values with units (mm/day, m, %, °C) on a log count axis, so the thin tail is visible;
the dashed red line is the 1.5×IQR fence at 56.30 mm. Bottom row = the same four columns
**after** `log1p` + `StandardScaler`, x-axis in σ (train mean 0, train sd 1), dotted black line
at the mean. *Interpretation:* the top row is the reason for the strategy — `rainfall_mm` puts
essentially all of its mass in the first two bins and then runs to 1,775.72 mm, and 9,274 panel
days sit beyond the fence, 100.000 % of them same-day floods. The bottom row is the result: the
same columns are now comparable, single-peaked and inside roughly ±3 σ (one river reading
reaches 8.10 σ), with the tail *compressed but not removed* — which is the whole point, because
those tail days are the positive class. The two surviving visual defects are deliberate and
are the two censoring ceilings: `soil_moisture_percent` keeps an isolated bar at **3.4919 σ**
(its last bin holds 6,165 train days against 16 in the bin below), and `temperature_celsius`
keeps an inflated right edge at **1.9674 σ** (last bin 1,823 days against 1,137 in the bin
below, of which 1,074 are the exact 45.00 °C tie). No transform may break those ties, and this
figure is the evidence that neither `log1p` nor `StandardScaler` did.

#### All four strategies are fitted on train only — and that is measurable

`fit_preprocessor(X_train)` sees 310,500 training rows and nothing else; validation and test
pass through `transform` alone. To show the discipline is doing work rather than being
decorative, the identical preprocessor was also fitted on the **full usable frame** (456,550
rows, deliberately wrong) and the two were compared:

| quantity | value |
|---|---|
| largest &#124;Δ mean&#124; across the 7 numeric columns | **0.123038** (`temperature_celsius`) = **0.011734 σ** |
| largest &#124;Δ scale&#124; | **0.069436** (`soil_moisture_percent`) |
| largest shift inflicted on a training row (310,500 × 21 cells) | **0.048422 σ**, mean 0.001442 σ |
| for contrast: a scaler fitted on the test window alone | **0.043069 σ** |

The two fits are **not** identical, so the leak is real — but it is **small on this panel**,
because the panel is stationary (stage 3A: `year` correlates with nothing at |r| ≤ 0.037, and
the next-day flood rate only drifts **2.1147 %** on train → **2.3329 %** on test). The honest
reading is that here the train-only rule is a *correctness guarantee that happens to cost
~0.05 σ*, not a fix for a large distortion: it is cheap insurance against a distribution shift
this particular simulator does not have. A dataset with a trend would show exactly the same
mechanism with a much larger number.

#### What this hands to 4C

| Strategy | Handed over as |
|---|---|
| Missing values | `SimpleImputer(median)` + `SimpleImputer(most_frequent)`, medians/modes stored on the fitted object |
| Outliers | `SkewCompressor` (`log1p` on 2 columns) + `ExtremeRainfallFlag` (`is_extreme_rainfall`, threshold 56.30 mm) — **0 rows dropped** |
| Categorical encoding | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, 14 dense columns |
| Scaling | `StandardScaler` on the 7-column numeric block, after `log1p`; both ceiling ties (100.00 %, 45.00 °C) preserved and never repaired |

#### Script

```python
--8<-- "docs/projects/eda/code/pipeline.py"
```

### B. Dimensionality reduction

The features projected here are the **scaled training matrix** produced by the 4C pipeline —
`fit_preprocessor(X_train)` then `transform(pre, X_train)` — fitted on the **2000–2016 training
window only** and never on validation or test. The label enters in exactly one way: it colours
the points. **No model is trained in this item**; PCA, t-SNE and UMAP are projections.

Two different matrices are used, deliberately:

| projection | rows | why |
|---|---|---|
| **PCA** | **310,500** — the whole training set | an exact SVD of a 310,500 × 21 matrix costs **0.6–0.9 s**, so there is no reason to sample |
| **t-SNE, UMAP** | **30,000** — a stratified sample | the same fit costs minutes to hours at full size; see the sampling note below |

#### The matrix that is projected

| quantity | value |
|---|---|
| raw panel | 456,600 × 21 (station-days) |
| `X_train` (raw inputs, 2000–2016) | `(310500, 8)` = 6 numeric + 2 categorical |
| **`Z = transform(pre, X_train)`** | **(310500, 21)**, `float64` |
| `np.isnan(Z).sum()` | **0** |
| training positives | **6,566** of 310,500 = **2.1147 %** → **46.3 : 1** *inside the training window* (the whole usable panel is 2.166 % / 45.2 : 1) |

The pipeline turns 8 raw columns into **21 features**: 7 numeric
(`num__rainfall_mm`, `num__river_level_m`, `num__soil_moisture_percent`,
`num__temperature_celsius`, `num__day_of_year`, `num__monsoon_season`,
`num__is_extreme_rainfall`), 10 basin one-hots (`cat__basin_Brahmaputra` … `cat__basin_Yangtze`)
and 4 season one-hots (`cat__season_Autumn`, `cat__season_Spring`, `cat__season_Summer`,
`cat__season_Winter`). Every projection below runs on those 21 columns, all of them on one
scale after the 4C standardisation — which is what makes a variance-based method such as PCA
meaningful at all.

#### PCA — how much variance the plane carries

![Figure 4B.1 — PCA scree and cumulative explained variance, 310,500 × 21](figures/fig50_pca_variance.png)

**Figure 4B.1.** Panel (a): variance explained by each of the 21 components (the first ten
labelled). Panel (b): the cumulative curve, with the PC1 + PC2 level marked. Fitted on all
310,500 training rows, `svd_solver="full"`, **0.6–0.9 s**.

| component | variance % | cumulative % | | component | variance % | cumulative % |
|---|---|---|---|---|---|---|
| **PC1** | **55.3581** | **55.3581** | | PC6 | 2.0485 | 87.2088 |
| **PC2** | **14.8991** | **70.2572** | | PC7 | 1.4125 | 88.6213 |
| PC3 | 8.7895 | 79.0467 | | PC8 | 1.1601 | 89.7814 |
| PC4 | 3.4701 | 82.5168 | | PC9 | 1.1561 | 90.9374 |
| PC5 | 2.6435 | 85.1603 | | PC10 | 1.1561 | 92.0935 |

**Results-table row 9 — variance explained by PC1 + PC2: 70.2572 %** (PC1 **55.3581 %**,
PC2 **14.8991 %**). PC11–PC21 together carry the remaining **7.9065 %**, and **PC21 alone
carries 0.0000 %**: the 21 encoded columns are not linearly independent — each one-hot block
sums to 1 on every row and `monsoon_season` is exactly `Summer + Autumn` — so at least one
eigenvalue is structurally zero. Nine components reach 90 % and 13 reach 95 %.

**Conclusion (Figure 4B.1): two components carry 70.26 % of the variance of the scaled
training matrix, and the elbow is inside the first three — the 21 features are, for variance
purposes, a two- or three-dimensional object dominated by the seasonal water cycle.**

#### PCA — reading the loadings

![Figure 4B.2 — PCA loadings, first 8 components × 21 scaled features](figures/fig50_pca_loadings.png)

**Figure 4B.2.** Loadings (correlation of each scaled feature with each component), diverging
scale centred at 0, annotated. Components are ordered by variance; the share is given in the
row labels.

**Table 4B.1 — the eight largest |loading| of PC1 and PC2** (coefficients are in units of the
standardised features, so they are directly comparable).

| PC1 (55.3581 %) | loading | PC2 (14.8991 %) | loading |
|---|---|---|---|
| `num__soil_moisture_percent` | **+0.4432** | `num__is_extreme_rainfall` | **+0.5832** |
| `num__rainfall_mm` | **+0.4273** | `num__day_of_year` | **−0.5724** |
| `num__river_level_m` | **+0.4165** | `num__temperature_celsius` | **−0.4207** |
| `num__monsoon_season` | **+0.4027** | `num__river_level_m` | +0.2533 |
| `num__temperature_celsius` | **+0.3694** | `cat__season_Autumn` | −0.1770 |
| `num__day_of_year` | +0.2557 | `num__rainfall_mm` | +0.1687 |
| `num__is_extreme_rainfall` | +0.2158 | `cat__season_Spring` | +0.1261 |
| `cat__season_Summer` | +0.1025 | `num__soil_moisture_percent` | +0.0955 |

**What PC1 means.** Every water and season variable loads **positive and together** — soil
moisture, rainfall, river level, monsoon flag, temperature, day of year. PC1 is therefore a
single **wet-season axis**: no variable opposes another, so the component is not a contrast
but a depth-of-monsoon coordinate. High PC1 is a warm, wet day in the rainy half of the year
with wet soil and a high river; low PC1 is a dry-season day. The squared-loading share is
**96.3 % numeric, 3.7 % season one-hots and 0.0 % basin one-hots** — geography contributes
*nothing* to the dominant direction, which is the same null result stage 3B reached by a
different route.

**What PC2 means.** PC2 is a genuine contrast: the extreme-rainfall flag, river level and
rainfall load **positive**, while day of year, temperature and `Autumn` load **negative**.
Day of year and temperature move together through the year — stage 3A reports them as a
seasonal pair — and here they take the *same* sign, so PC2 is not a second seasonal axis. It
is an **event-vs-season axis**: high PC2 marks a day whose rainfall is extreme and whose river
is high *relative to what its position in the year would predict*; the smooth seasonal ramp is
subtracted out by the negative day-of-year/temperature pair. Its squared-loading share is again
**94.9 % numeric**. Basin one-hots stay at **0.1 %** on PC2 (and **0.0 %** on PC1); the
components in which they become visible at all are PC4–PC8, each carrying between **3.4701 %**
and **1.1601 %** of the variance. Geography is not a direction this dataset has.

**Conclusion (Figure 4B.2): PC1 is a wet-season axis (+0.44 soil moisture, +0.43 rainfall,
+0.42 river level, +0.40 monsoon, all the same sign) and PC2 is an event-vs-season axis
(+0.58 extreme-rainfall flag against −0.57 day of year and −0.42 temperature); both are
> 94 % numeric, and no basin indicator loads materially on either.**

#### PCA — do the classes separate?

![Figure 4B.3 — PC1–PC2 scatter of the training set coloured by the target, and the PC1 distribution of each class](figures/fig50_pca_scatter.png)

**Figure 4B.3.** Panel (a): all **6,566** positives drawn on top of a random **40,000** of the
303,934 negatives, with both class centroids. Panel (b): the **full** PC1 distribution of each
class on a log count scale, with the negatives' 99th percentile marked. The two panels use the
same 310,500 rows.

| | negatives (303,934) | positives (6,566) | shift | positives beyond the negatives' 99th percentile |
|---|---|---|---|---|
| PC1 | mean −0.0555, sd 2.1681 | mean +2.5689, sd 1.4534 | **+2.6244** | **4.23 %** (95.77 % inside) |
| PC2 | mean −0.0009, sd 1.1240 | mean +0.0409, sd 1.5689 | +0.0418 | 3.96 % (96.04 % inside) |

Panel (a) must be read with its own arithmetic: every positive is drawn but only **13.2 %** of
the negatives are, so the visible red:grey ratio overstates the true class ratio by **7.6×**.
The honest reading is panel (b), which uses every row: the positives are a *shifted but fully
overlapping* copy of the negatives on PC1 — a 2.62-unit shift of the mean, yet **95.77 %** of
the positives still sit inside the negatives' central 99 % range, and on PC2 the shift is
+0.0418, i.e. **nothing at all**.

**Conclusion (Figure 4B.3): the classes do not separate in the PCA plane.** The positive class
is over-represented in the wet-season tail of PC1 — the **4.23 %** of positives beyond the
negatives' 99th percentile are the only part of the class PC1 isolates at all — while **95.77 %**
of positives sit inside the range the negatives already occupy, and on PC2 the class means differ
by 0.0418. No linear cut on PC1 or PC2 can separate them.

#### t-SNE — three values of `perplexity`

!!! note "The 30,000-row sample is stratified, not random — and that inflates the red"
    310,500 rows is impractical for t-SNE and UMAP, so both were run on a **30,000-row
    sample**: **all 6,566 training positives** (100.000 % of them) plus **23,434 negatives
    drawn at random from the 303,934** (7.71 %), with `np.random.default_rng(42)` and no
    replacement. **The sample's positive rate is 21.8867 %, which is 10.35× the 2.1147 %
    training rate.** Every red point is a real training row, but the red *density* in
    Figures 4B.4–4B.7 is inflated by construction: the seven panels of Figure 4B.7 are
    comparable **with each other**, and are **not** estimates of the population. The
    alternative — a uniform 30,000-row sample — draws the same **9.7 %** of rows from each
    class (30,000 of 310,500), which would leave a positive class too small to see at all.
    PCA (Figures 4B.1–4B.3) uses the full set.

![Figure 4B.4 — t-SNE at perplexity 5, 30 and 50 on 30,000 rows](figures/fig51_tsne_perplexity.png)

**Figure 4B.4.** The same 30,000 rows under t-SNE at three perplexities
(`learning_rate="auto"`, `init="pca"`, `max_iter=1000`, `random_state=42`, `n_jobs=1`).
Negatives are drawn first, positives on top with larger markers and higher alpha. Axes are
embedding coordinates and carry **no unit and no scale** — the panel sizes say nothing.

| `perplexity` | runtime | KL divergence | what the panel shows |
|---|---|---|---|
| 5 | **257.7–406.7 s** | 1.9584 | one diffuse blob; positives concentrated on its right-hand side |
| 30 | **278.9–398.8 s** | 1.1470 | the positives break into dozens of compact clumps among looser negative clouds |
| 50 | **350.2–365.5 s** | 0.9393 | the same clumped structure, slightly smoother |

The KL divergence falls as perplexity rises (1.9584 → 1.1470 → 0.9393); that is arithmetic,
not quality — a larger perplexity makes the target distribution P smoother and easier to match.
The **runtime does not track perplexity in any usable way**: the same three fits took
257.7 / 278.9 / 350.2 s in the quieter of the two runs and 406.7 / 398.8 / 365.5 s in the
contended one (see [Runtime](#runtime) below). Only the KL values and the embeddings reproduce.

**Conclusion (Figure 4B.4): t-SNE shows that the positive class is not a diffuse minority
spread evenly through feature space — it collapses into many small, tight clumps, i.e. flood
days are near-duplicates of one another in the 21 scaled features — while the negatives form
looser, more extended clouds.**

#### UMAP — three values of `n_neighbors`

![Figure 4B.5 — UMAP at n_neighbors 5, 15 and 50 on the same 30,000 rows](figures/fig52_umap_nneighbors.png)

**Figure 4B.5.** The same 30,000 rows under UMAP (`min_dist=0.1`, `metric="euclidean"`,
`random_state=42`, `n_jobs=1`). A one-off numba JIT warm-up of **20.1–21.5 s** is excluded from
the timings.

| `n_neighbors` | runtime | what the panel shows |
|---|---|---|
| 5 | **44.1–46.9 s** | the visually *most* separated picture: positives in a dense arc above a negative field |
| 15 | **28.2–35.2 s** | that arc dissolves into dozens of small red blobs interleaved with grey ones |
| 50 | **41.6–46.4 s** | smaller, sharper blobs; the red and the grey are thoroughly interleaved |

**Conclusion (Figure 4B.5): UMAP reproduces the t-SNE finding — the positives form compact
local groups rather than one region — and it does so an order of magnitude cheaper
(**28.2–46.9 s** against t-SNE's **257.7–406.7 s**); but the apparent separation of the
`n_neighbors=5` panel does not survive the change to 15 or 50, which is the first sign that
the layout, not the data, is driving it.**

#### The comparison: what the nonlinear projections reveal that PCA does not

![Figure 4B.6 — PCA vs t-SNE (perplexity 30) vs UMAP (n_neighbors 15) on the same 30,000 rows](figures/fig53_comparison.png)

**Figure 4B.6.** The **same** 30,000 rows under all three projections, so the panels are
comparable with one another. The PCA panel is the full-training-set PCA evaluated at those
rows. Each panel carries the measured neighbourhood composition of its own embedding.

![Figure 4B.7 — neighbourhood label composition and local scale for all seven embeddings](figures/fig53_local_enrichment.png)

**Figure 4B.7.** Panel (a): mean positive share of the 25 nearest neighbours in each embedding,
for positive and for negative points, with a ratio above each pair and the sample's 21.89 %
positive rate as the no-structure reference. Panel (b): median distance to the 25th neighbour
as a share of the embedding's diagonal, i.e. how stretched the local neighbourhood is.
Neighbours are found with a `scipy` KD-tree; this measures what a projection preserved, it
does not predict anything.

| embedding (same 30,000 rows) | a positive's 25-NN are positive | a negative's are | **ratio** | 25-NN radius, positives | negatives |
|---|---|---|---|---|---|
| **PCA (PC1–PC2)** | 46.49 % | 14.97 % | **3.11** | 0.181 | 0.246 |
| t-SNE `perplexity=5` | 50.77 % | 13.85 % | 3.67 | 0.923 | 0.922 |
| **t-SNE `perplexity=30`** | 51.13 % | 13.79 % | **3.71** | 0.559 | 0.535 |
| t-SNE `perplexity=50` | 50.98 % | 13.86 % | 3.68 | 0.463 | 0.436 |
| UMAP `n_neighbors=5` | 50.48 % | 13.79 % | 3.66 | 0.396 | 0.347 |
| **UMAP `n_neighbors=15`** | 50.72 % | 13.97 % | **3.63** | 0.252 | 0.224 |
| UMAP `n_neighbors=50` | 50.43 % | 14.01 % | 3.60 | 0.184 | 0.155 |

Figure 4B.6 uses `perplexity=30` and `n_neighbors=15` because they are the balanced defaults
of the two sweeps; the diagnostic above shows the choice is immaterial — every one of the
seven embeddings lands between **3.60 and 3.71**, a spread of 3 %.

**What the nonlinear projections reveal that PCA does not.**

1. **Shape, not separability.** PCA compresses the matrix onto one wet-season axis and one
   event-vs-season axis, and in that plane the positive class is a single dense blob plus a
   sparse tail. t-SNE and UMAP resolve the same class into **dozens of compact, well-separated
   clumps**: in the 21 scaled features, flood days are near-duplicates of each other, so the
   class has a *granular* structure that a linear projection smears into one region. That is
   the real finding of this item, and it is invisible in Figure 4B.3.
2. **Local scale, and how little of it is readable.** Panel (b): in PCA the positives are
   locally *denser* than the negatives (radius **0.181** against **0.246** of the diagonal) —
   the flood class occupies a compact region. In every nonlinear embedding that asymmetry
   disappears (t-SNE `perplexity=5`: **0.923** vs **0.922**; UMAP `n_neighbors=15`: **0.252**
   vs **0.224**), because t-SNE fixes every point's effective neighbourhood size through its
   perplexity by construction. The radius also falls steadily as the parameter rises
   (t-SNE 0.923 → 0.559 → 0.463, UMAP 0.396 → 0.252 → 0.184), which is why the panels look
   progressively less clumpy. **A quantity that moves with the parameter is a property of the
   layout, not a measurement of the data** — which is the caveat below, made numeric.
3. **But the labels inside those clumps are still mixed.** A positive's 25 nearest neighbours
   are **51.13 %** positive under the best t-SNE and **50.72 %** under the best UMAP — i.e.
   about **half of the immediate neighbourhood of a flood day is not a flood day**. The
   enrichment over a negative point's neighbourhood is **3.71×** and **3.63×**; the linear PCA
   plane already achieves **3.11×**. The nonlinear methods therefore buy about **19 %** more
   local label structure than a two-component PCA — real, but nowhere near a separation.
4. **The layout is not evidence.** The `n_neighbors=5` panel of Figure 4B.5 looks by far the
   most separated and measures **3.66×** — statistically indistinguishable from the
   `n_neighbors=50` panel's **3.60×**, whose layout looks thoroughly mixed. Same data, same
   measurement, opposite visual impression: the gap in the picture is UMAP's, not the
   dataset's.

!!! warning "What t-SNE and UMAP do not license"
    In t-SNE and UMAP, **cluster sizes and the distances between clusters have no direct
    reading**. Only the neighbourhood structure is meaningful — *which* points end up near
    *which*. A tight clump is not a dense, homogeneous subpopulation; a wide gap is not
    dissimilarity; a big blob is not "more" of anything. This report therefore draws exactly
    one quantitative conclusion from those embeddings (the 25-neighbour label composition of
    Figure 4B.7, which is a property of the neighbourhood graph the methods optimise) and reads
    the panels themselves only qualitatively.

    **The honest headline is that the classes overlap.** With the positive class at
    **2.1147 %** of the training window, three projections and seven parameter settings all
    agree: there is a locally enriched structure — a flood day's neighbourhood is ~3.6× more
    flood-prone than a non-flood day's — and no global separation, in linear or in nonlinear
    coordinates, in any two-dimensional view of these 21 features. No model is trained in this
    deliverable, so this is a statement about the geometry of the feature space, not a
    performance estimate; it does say that a modeller should not expect a linear cut in two
    dimensions to carry the task.

**Conclusion (Figures 4B.6 and 4B.7): t-SNE and UMAP reveal that the positive class is
granular — flood days cluster tightly together in the scaled feature space — and that this
granularity raises the local label enrichment only from PCA's 3.11× to at most 3.71×, so the
nonlinear projections expose the shape of the class, not a way to separate it.**

#### Runtime

Wall clock, single-threaded: `OMP_NUM_THREADS = LOKY_MAX_CPU_COUNT = OPENBLAS_NUM_THREADS =
MKL_NUM_THREADS = NUMBA_NUM_THREADS = 1`, so the sandbox's named-pipe restriction on
`n_jobs > 1` never applies. Both columns are the **same script** run twice; every projection is
bit-identical between them (the printed checksums match to all six decimals) — only the clock
moves.

| step | run A | run B |
|---|---|---|
| PCA, full 310,500 × 21 training set, all 21 components | **0.6 s** | **0.9 s** |
| stratified 30,000-row sample construction | 0.01 s | 0.04 s |
| t-SNE, `perplexity=5` | **257.7 s** | **406.7 s** |
| t-SNE, `perplexity=30` | **278.9 s** | **398.8 s** |
| t-SNE, `perplexity=50` | **350.2 s** | **365.5 s** |
| UMAP, `n_neighbors=5` | **44.1 s** | **46.9 s** |
| UMAP, `n_neighbors=15` | **28.2 s** | **35.2 s** |
| UMAP, `n_neighbors=50` | **41.6 s** | **46.4 s** |
| 25-neighbour diagnostic, 7 embeddings | 1.44 s | 2.14 s |
| UMAP numba JIT warm-up (one-off, 500 rows) | 21.5 s | 20.1 s |
| **whole script** | **1043.4 s (17.4 min)** | **1351.2 s (22.5 min)** |

The **parameter sweeps are 96 % of the runtime** (886.7 s of t-SNE + 113.9 s of UMAP against
1043.4 s; 1171.0 s + 128.5 s against 1351.2 s). The practical lesson: the second and third
values of a parameter cost nothing to write and almost everything to run. **One t-SNE fit costs
an order of magnitude more than one UMAP fit** — 257.7–406.7 s against 28.2–46.9 s — and the
same arithmetic moved by 1.4× between the two runs, so a single t-SNE clock is dominated by
whatever else the machine is doing. UMAP's cost does **not** grow monotonically with
`n_neighbors`, and it does so in neither run (44.1 / 28.2 / 41.6 s and 46.9 / 35.2 / 46.4 s),
because the neighbour graph and the layout optimisation scale differently. PCA on the full
training set is **0.6–0.9 s** — three orders of magnitude below the nonlinear methods and, as
Figures 4B.6 and 4B.7 show, it gives up remarkably little.

!!! note "What is reproducible here, exactly"
    Every **plotted and printed quantity** is deterministic: `random_state=42` everywhere,
    `n_jobs=1` for UMAP, `svd_solver="full"` for PCA, and the two runs above agree on all of
    them — the embedding checksums, the neighbourhood percentages and the loadings included.
    The only numbers that differ between runs are wall clocks: this table's two columns and the
    "runtime" lines of the script's output. **No runtime is baked into any figure** — the panel
    titles in Figures 4B.4–4B.6 carry the parameter values only, so the committed PNGs are
    byte-reproducible. A runtime is a property of the machine, not of the projection.

```python
--8<-- "docs/projects/eda/code/t6_reduction.py"
```

### C. Pipeline

The four strategies of 4A are assembled into **one importable object**, in the shape the
statement asks for: a `ColumnTransformer` with a numeric branch and a categorical branch, each
a `Pipeline`. The module is `docs/projects/eda/code/pipeline.py`; `pipeline.py` holds no
plotting, no training and no global state, and it **fits nothing outside `fit_preprocessor`**.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocess = ColumnTransformer([
    ("num", Pipeline([...]), numeric_features),   # 6 raw numerics -> 7 columns
    ("cat", Pipeline([...]), categorical_features),  # 2 categoricals -> 14 columns
])

X_train_t = preprocess.fit_transform(X_train)   # fit on train only
X_test_t  = preprocess.transform(X_test)        # test is only transformed
```

#### What the branches actually contain

| branch | steps | in → out |
|---|---|---|
| `num` | `ExtremeRainfallFlag` → `SkewCompressor` → `SimpleImputer(median)` → `StandardScaler` | 6 → **7** |
| `cat` | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` | 2 → **14** |

`remainder="drop"` and `sparse_threshold=0.0` are explicit: the eight raw feature columns are
the only inputs, and the output is always one **dense `float64`** array — never a SciPy sparse
matrix — because the consumer is a neural network reading mini-batches.

#### Fitted on train, transformed on test — the reported numbers

`fit_preprocessor(X_train)` is called once, on the **310,500 training rows (2000–2016)**; the
test frame is only passed to `transform`. Nothing in the pipeline is re-fitted, and no
statistic of any kind is computed on validation or test (the leak check in 4A quantifies what
fitting outside train would have moved: up to **0.048422 σ** on a training row).

#### Table 4C.1 — the pipeline's output (results-summary rows for the report)

| quantity | value |
|---|---|
| `X_train` shape (raw inputs, 2000–2016) | `(310500, 8)` = 6 numeric + 2 categorical |
| `X_test` shape (raw inputs, 2021–2024) | `(73000, 8)` |
| **`X_train_t.shape`** | **`(310500, 21)`**, dtype `float64` |
| **`X_test_t.shape`** | **`(73000, 21)`**, dtype `float64` |
| columns before → after | **8 → 21** |
| block widths | numeric **7**, `basin` one-hot **10**, `season` one-hot **4** |
| **`int(np.isnan(X_train_t).sum())`** | **0** |
| **`int(np.isnan(X_test_t).sum())`** | **0** |
| all values finite | `True` (train and test) |
| transformed range, all 21 features | **-2.9015 σ to 8.1045 σ** |

The test figure is **73,000**, not 73,050: 73,050 rows sit in the 2021–2024 window, and the
one-day `shift(-1)` drops the last day of each of the 50 stations (2024-12-31, inside that
window), leaving 73,000 usable rows. Train 310,500 + validation 73,050 + test 73,000 =
**456,550** usable rows, exactly the panel minus its 50 unlabelable last days.

`transform()` additionally **raises** if the result contains a non-finite value, so "no NaN" is
a guarantee of the seam rather than an observation about one run.

#### The 21 feature names, from `get_feature_names_out()`

```text
 1. num__rainfall_mm            8. cat__basin_Brahmaputra   15. cat__basin_Pearl
 2. num__river_level_m          9. cat__basin_Chao Phraya   16. cat__basin_Salween
 3. num__soil_moisture_percent 10. cat__basin_Ganges        17. cat__basin_Yangtze
 4. num__temperature_celsius   11. cat__basin_Helmand       18. cat__season_Autumn
 5. num__day_of_year           12. cat__basin_Indus         19. cat__season_Spring
 6. num__monsoon_season        13. cat__basin_Irrawaddy     20. cat__season_Summer
 7. num__is_extreme_rainfall   14. cat__basin_Mekong        21. cat__season_Winter
```

The block above is laid out in three columns for width, but it is one flat list: read the
indices **1 → 21** in order (down the first column, then the second, then the third). Features
**1–7** are the numeric branch — the six raw numerics in `NUMERIC_RAW` order with
`is_extreme_rainfall` appended as the seventh; **8–17** are the ten `basin` indicators in
alphabetical order (`Brahmaputra` … `Yangtze`); **18–21** are the four `season` indicators
(`Autumn`, `Spring`, `Summer`, `Winter`). All 21 names are unique, and none is typed by hand
anywhere: the list is produced by the fitted object itself, and `transform()` returns the
columns in exactly this order.

#### How `is_extreme_rainfall` is produced, and why it lives inside the pipeline

It is **step 1 of the numeric branch**, a small stateless transformer
(`ExtremeRainfallFlag`) that appends `1.0` where the **raw** `rainfall_mm` exceeds the declared
constant `EXTREME_RAINFALL_MM = 56.30 mm`, and it implements `get_feature_names_out` so the
derived column is named like any other. Abridged — the full source is embedded above:

```python
class ExtremeRainfallFlag(TransformerMixin, BaseEstimator):
    def transform(self, X):
        arr = ...                             # the 6 numeric columns, in NUMERIC_RAW order
        raw = arr[:, self.column_index]       # raw rainfall_mm; NaN -> not flagged
        flag = (raw > self.threshold).astype(float).reshape(-1, 1)
        return np.hstack([arr, flag])
```

**Why not compute it on the raw frame before the pipeline:** a flag added outside would be
present when the pipeline is fitted and *silently absent* whenever `transform` is called on a
new frame — the column count would still match, the model would read a shifted feature matrix,
and nothing would raise. Inside the pipeline it is applied by `fit` **and** by `transform`, it
travels with the fitted object, and it stays importable for 4B and for the Classification
deliverable. The threshold is a module constant, not a fitted statistic, so it is identical for
train, validation and test (4A shows the train-only fence agrees on 0 of 456,600 rows).

The companion step `SkewCompressor` applies `log1p` to `rainfall_mm` and `river_level_m` **by
name**, leaving `soil_moisture_percent` (censored at 100.00 %) and `temperature_celsius`
(censored at 45.00 °C and partly negative, where `log1p` is undefined, so 2A's measured skew
of -0.57 for a shifted log applies) untouched, along with the freshly added flag (binary,
where `log1p(1.0) = 0.693` would destroy the encoding). The two censoring ties therefore reach
the model intact, as the isolated spikes at **3.4919 σ** and **1.9674 σ** in Figure 4C.1.

#### Importable and reused downstream

```python
from data import train_test_frames
from pipeline import NUMERIC_RAW, CATEGORICAL_RAW, EXTREME_FLAG_COL, EXTREME_RAINFALL_MM
from pipeline import build_preprocessor, fit_preprocessor, transform, feature_names

X_train, X_test, y_train, y_test = train_test_frames()
preprocess = fit_preprocessor(X_train)      # fit on TRAIN ONLY
X_train_t  = transform(preprocess, X_train)  # (310500, 21)
X_test_t   = transform(preprocess, X_test)   # ( 73000, 21)
names      = feature_names(preprocess)       # 21 names
```

The module is imported by `t5_pipeline.py` (this item), by **`t6_reduction.py`** (stage 4B,
which projects these scaled features with PCA / t-SNE / UMAP), and by the **Classification
deliverable**, which reuses it unchanged so the EDA and the model see byte-identical features.
`build_preprocessor()` returns an **unfitted** `ColumnTransformer`; `fit_preprocessor(X_train)`
returns a fitted one; `transform(preprocess, X)` accepts train, validation, test or genuinely
new data. Running the module directly (`python pipeline.py`, cwd `docs/projects/eda/code`)
prints the shapes, the NaN counts and the 21 names as a self-report.

#### Dropped columns — cross-reference to stage 1B

The pipeline never sees `severity_level` or `flood_risk_score` (**leakage**: `risk > 7.0`
reproduces the same-day label on 456,600/456,600 rows, ROC-AUC 1.0000), `data_quality_score`
(**noise**: |r| ≤ 0.002 with everything, next-day ROC-AUC 0.5072), `station_name`,
`country`, `latitude`, `longitude` (redundant or constant per station) and `day` (noise,
r = 0.0007 with the target). The rules live in the frozen `DROPPED` mapping in `code/data.py`
and were derived in stage 1B — they are **not** re-derived here.

#### Figure 4C.1 — one comparable scale out of eight incomparable inputs

![Figure 4C.1 — raw numeric inputs vs the 21 transformed features](figures/fig40_transformed_feature_scales.png)

**Figure 4C.1.** Panel A: boxplots of the 6 raw numeric inputs on the 310,500 training rows, in
their native units (mm/day, m, %, °C, day-of-year, 0/1) — one shared linear axis, 1.5×IQR
whiskers, fliers shown. Panel B: boxplots of all **21** pipeline outputs on the same rows, in σ
(standardised units), fliers hidden for legibility (whiskers still 1.5×IQR); dotted line at the
train mean, 0 σ. Orange = median, green triangle = mean. *Interpretation:* panel A is the
problem — on one shared linear axis `rainfall_mm` runs from 2.02 to 1,775.72 mm while
`soil_moisture_percent` (0.46–100.00), `temperature_celsius` (-2.31–45.00),
`day_of_year`, `monsoon_season` (0/1) and the rainfall bulk are all flat lines near the
bottom; no single learning rate is defensible across that. Panel B is the delivered fix: all
21 features now share one scale — the bulk sits inside ±3 σ and nothing exceeds **8.1045 σ** —
so the binary flag and the one-hot indicators sit next to the continuous measurements instead
of being dwarfed by them. The two features that reach furthest are the river-level extreme at
**8.1045 σ** and the two censoring ceilings — soil moisture at **3.4919 σ** and temperature at
**1.9674 σ**: the extreme days the outlier strategy refused to delete, and the ties no
transform may break. All three are still there.

#### Script

```python
--8<-- "docs/projects/eda/code/t5_pipeline.py"
```

## 5. Synthesis

### Main findings

**F1 — The panel is complete, and the quality problem is not missing data.**
50 stations × 9,132 days = 456,600 rows, 2000-01-01 → 2024-12-31, with **0 missing values across
all 21 columns (0 of 9,588,600 cells)**, **0 duplicated rows**, **0 duplicated `(station_id, date)`**
and **0 date gaps** ([Figure 1.1](figures/fig01_panel_completeness.png), Table 1B.1).
There is nothing to impute, so the imputation strategy in 4A is a *robustness* guarantee for
future partial days, not a repair. Saying otherwise would invent a problem the data does not have.

**F2 — Two columns are the label in disguise, and the rule is exact.**
On **all 456,600 rows**, `flood_event_occurred == (flood_risk_score > 7.0)` — ROC-AUC **1.0000** on
that one column, with `max(risk | no flood) = 7.00` and `min(risk | flood) = 7.01`. `severity_level`
is the same information again, rebinned (Table 1B.2, Table 1B.3). Both are excluded from the
feature set. The one-day-ahead nuance is stated in 1B: for *next-day* prediction the risk score is
not strictly leakage, but it is **redundant** — r = 0.934 with `soil_moisture_percent`, and a
*weaker* next-day predictor (**0.8088**) than soil moisture alone (**0.8364**). Excluding it costs
nothing. Also flagged: **32 rows with `risk == 7.00`** are `Extreme` yet not floods — the dataset's
one genuine internal contradiction.

**F3 — The task is rare, imbalanced, and gated by the calendar.**
9,889 positives of 446,661 = **2.166 %**, an imbalance of **45.2 : 1**; the majority class scores
**97.83 %** accuracy, so accuracy is never used again ([Figure 1.2](figures/fig01_target_balance.png)).
Floods effectively do not exist outside the monsoon: **≈ 13,700× more likely** inside it
(**5.1702 %** vs **0.00038 %** — one flood in 265,300 off-season rows), with **0 floods** in
Jan–Apr, Nov and Dec (Table 3B.1, Table 3B.2,
[Figure 3B.1](figures/fig33_flood_rate_month.png)). Under the **next-day** target, Jul–Sep holds
**94.31 %** of all floods (under the raw same-day label, 95.04 %). Every metric in the modelling
phase must therefore be reported **inside the monsoon window**, where the base rate is 5.17 %, not
the 2.17 % headline.

**F4 — The "outliers" are the positive class. This is the decision that matters most.**
The 1.5×IQR fence on `rainfall_mm` is **56.30 mm** and flags **9,274 rows (2.031 %)** — and
**100.000 % of the rows above it are floods**, against **0.137 %** below it (Figure 2A.2,
[Figure 3C.1](figures/fig36_rainfall_monsoon.png)). A textbook "remove the outliers" step would
delete the entire positive class. The strategy is therefore **0 rows removed, 0 winsorised**:
`log1p` compresses the tail while preserving order, and an explicit `is_extreme_rainfall` indicator
restores the magnitude (4A, Table 4A.1).

**F5 — Spearman is the honest correlation method here, and one pair is redundant.**
All four measurements are severely right-skewed (`rainfall_mm` skew **10.513**, kurtosis **158.202**;
`river_level_m` **9.532** / **140.606**), so Pearson is dominated by a few extreme monsoon days
([Figure 3A.1](figures/fig30_corr_heatmap.png)). The trimming test makes this quantitative: inside
the monsoon, deleting the 4.71 % of days above 100 mm collapses Pearson from **0.9632 → 0.6072**,
landing on the Spearman value (**0.6110**) while Spearman itself barely moves (Table 3A.2). The
redundant pair is **`rainfall_mm` ↔ `river_level_m`: Pearson 0.9604, Spearman 0.8689**. Both stay in
the feature set — they diverge for the next-day target (rainfall AUC 0.8091, river level 0.7948) and
the ablation decides. A second, weaker redundancy is `soil_moisture_percent` ↔ `temperature_celsius`
(Spearman 0.7500), which is a **seasonal artefact**: both track `day_of_year` (Table 3A.3).

**F6 — Geography and vulnerability explain almost nothing; the season explains almost everything.**
Basin flood rates span **1.947 % – 2.470 %** (1.27×) and station rates **1.632 % – 2.585 %** (1.58×)
against a 2.166 % mean (Table 3B.3, [Figure 3B.2](figures/fig34_flood_rate_basin.png),
[Figure 3B.3](figures/fig35_flood_rate_station.png)). Between-station variance of rainfall is
**0.0414 %** of the total, and by basin the four measurements differ by only **1.03×–1.49×** in
location ([Figure 3C.2](figures/fig37_measurements_basin.png)). This is a **null result reported as
such**: the per-station z-score anomalies planned for the modelling phase will largely re-encode
*season*, not local character, and `vulnerability_data.csv` has almost no signal to explain. Basin
is kept only as small static context; no conclusion should rest on it.

**F7 — For tomorrow's flood, today's *soil moisture* beats today's *rainfall*.**
Point-biserial correlation with the next-day target inverts the same-day picture: soil moisture
**0.1771**, monsoon flag 0.1752, temperature **0.1497**, river level 0.0946, rainfall **0.0850**,
today's flood 0.0558 (Table 3A.4). The mechanism is visible in the data: rainfall's next-day flood
rate is **non-monotone**, peaking at **7.644 %** in the 8th decile and *falling* to 3.948 % in the
top decile — a single very wet day is often the *event itself*, not a precursor. **Soil moisture is
the standing precondition; rainfall is the same-day trigger.** This is the strongest argument for
the accumulated-rain and rising-river window features in the modelling phase, and against leaning
on any single day's rainfall.

**F8 — Two measurements are censored, not merely skewed.**
`soil_moisture_percent` has **9,153 rows at exactly 100.00** (2.0046 %) and `temperature_celsius`
has **1,788 rows at exactly 45.00 °C** (0.392 %); nothing exceeds either ceiling. The
temperature ceiling carries far more mass than the local 0.1 °C bins below it and the soil ceiling far more than the 0.01 % bins below it (2A, 4A). `log1p`
and `StandardScaler` cannot spread a tie mass — they only preserve its order — so both ceilings are
**kept as-is, never clipped further**, and documented as generator artefacts a model must be told
about rather than defects to repair. `temperature_celsius` must **not** be `log1p`-transformed: it
contains negatives (min −2.31 °C) and a shifted log manufactures skew.

**F9 — The pipeline turns 8 raw columns into 21 features, with no missing values.**
`X_train_t.shape = (310500, 21)`, `X_test_t.shape = (73000, 21)`, `int(np.isnan(...).sum()) = 0` for
both, 8 raw → 21 = 7 numeric (6 raw + `is_extreme_rainfall`) + 10 basin dummies + 4 season dummies
(4C, [Figure 4C.1](figures/fig40_transformed_feature_scales.png)). Fitting on train only is not
cosmetic: a scaler fitted on the full usable frame differs, with a largest |Δ mean| of **0.123038**
(**0.011734** train σ) and a largest |Δ scale| of **0.069436**. The leak is *small here* because the
panel is stationary — but it is a correctness guarantee, and the honest framing is that it costs
about 0.05 σ rather than fixing a large distortion.

**F10 — Three projections agree: the classes overlap, but the positive class is *granular*.**
PC1 + PC2 carry **70.26 %** of the variance of the 310,500 × 21 scaled training matrix
(PC1 55.36 %, PC2 14.90 %). PC1 is a **depth-of-monsoon axis**, not a flood axis: every water and
season variable loads positive together (soil moisture +0.44, rainfall +0.43, river level, monsoon
flag, temperature, day of year), and the basin dummies contribute **0.0 %** — the same null result
as F6 ([Figure 4B.1](figures/fig50_pca_variance.png), [Figure 4B.2](figures/fig50_pca_loadings.png)).
The positive class sits **+2.62 units** along PC1, and **95.77 %** of it lies inside the negatives'
range, so a linear cut in the PCA plane cannot isolate it
([Figure 4B.3](figures/fig50_pca_scatter.png)). t-SNE and UMAP add one genuine finding and no more:
flood days are near-duplicates of one another, so the class resolves into **dozens of compact
clumps** that PCA smears into a single blob ([Figure 4B.6](figures/fig53_comparison.png)). Measured
on the same 30,000 rows, a flood day's 25 nearest neighbours are **51.13 %** flood days under the
best t-SNE and **50.72 %** under the best UMAP — against PCA's **46.49 %** — an enrichment of
**3.71×** and **3.63×** versus PCA's **3.11×** ([Figure 4B.7](figures/fig53_local_enrichment.png)).
So the nonlinear projections buy about **19 %** more local label structure: they expose the *shape*
of the class, not a way to separate it. The parameter sweeps also make the standard caveat numeric —
across seven settings the enrichment moves only 3 %, while the 25-neighbour radius falls from
**0.923** to **0.463** (t-SNE) and **0.396** to **0.184** (UMAP), and the visually *most*-separated
panel (`n_neighbors=5`, 3.66×) is statistically indistinguishable from the visually mixed
`n_neighbors=50` panel (3.60×). **A quantity that moves with the parameter is a property of the
layout, not of the data** — cluster sizes and inter-cluster distances in t-SNE and UMAP have no
direct reading (4B).

### Risks for modelling, and how each is handled

| # | Risk | Evidence | Handling |
|---|---|---|---|
| R1 | **Two columns give the answer away** | `risk > 7.0` ⇒ flood on 456,600/456,600 rows | Drop `severity_level` and `flood_risk_score`; show the risk score once, labelled as a leaky baseline |
| R2 | **Extreme values are the positive class** | 100.000 % of rows above the 56.30 mm fence are floods | Never filter or winsorise; `log1p` + explicit `is_extreme_rainfall` flag (0 rows removed) |
| R3 | **45.2 : 1 imbalance** | 2.166 % positives; majority class = 97.83 % accuracy | PR-AUC as the headline; class weights / positive oversampling; threshold chosen on validation for a stated recall |
| R4 | **Floods exist only inside the monsoon** | 1 flood in 265,300 off-monsoon rows (next-day); 94.31 % of floods in Jul–Sep | Always report metrics **inside the monsoon**; never quote a global AUC as if it were skill |
| R5 | **Almost no day-to-day label persistence** | 8,448 of 9,135 episodes are one day; P(flood tomorrow \| flood today) = 7.6 % | Lean on accumulated-rain and rising-river windows, not on the lagged label |
| R6 | **Daily rainfall is nearly white noise** | within-station lag-1 autocorrelation 0.162 | Single-day rainfall will not carry a window model; use multi-day sums |
| R7 | **Geography and vulnerability are ~useless** | basin rates 1.27×, station rates 1.58×; between-station rainfall variance 0.0414 % | Keep as small static context, report the null result, build nothing on them |
| R8 | **Two measurements are censored** | 9,153 rows at 100.00 %; 1,788 at 45.00 °C | Keep the ceilings; do not clip, impute or log-shift; document the tie mass |
| R9 | **`data_quality_score` is noise** | \|r\| ≤ 0.004 against every other column (exact max 0.0022); next-day AUC 0.5072 | Drop the column; document the check |
| R10 | **The dataset is synthetic and self-inconsistent** | 32 `risk == 7.00` rows are `Extreme` but not floods; station names contradict coordinates | Disclose it in the data dictionary; never frame a result as real flood risk |
| R11 | **Six calendar columns encode one date** | `month` ↔ `day_of_year` r = 0.9965; 0 mismatches on every consistency check | Keep `monsoon_season` as a numeric flag; the delivered pipeline feeds raw `day_of_year` and defers any `sin`/`cos` encoding to the modelling phase; `year`, `month` and `day` are dropped |
| R12 | **Rows one day apart are near-duplicates** | a random 80/20 split would share 9,132 / 9,132 dates and all 1,250 `(station, year)` blocks | Chronological split by date, identical dates for every station; test touched once, at the end |

### Results summary

| # | Results summary | Value |
|:--:|---|---|
| 1 | Dataset, task and target | **ASIA-FLOOD: 25-Year Flood Risk Atlas** (Kaggle, CC BY-SA 4.0, synthetic) · **binary classification** · `flood_event_occurred` on day *t+1*, same station |
| 2 | Instances × features (numerical / categorical) | **456,600 × 21** raw → **456,550** usable after the one-day shift · raw dtypes **14 numerical / 6 categorical + 1 datetime** → **6 numerical + 2 categorical** enter the pipeline → **21 features** out |
| 3 | Column with the most missing values and its percentage | **none — 0 missing in all 21 columns (0.00 %, 0 of 9,588,600 cells)** |
| 4 | Dropped columns and the reason | `severity_level`, `flood_risk_score` (leakage: `risk > 7.0` ⇒ flood, ROC-AUC 1.0000 on 456,600/456,600 rows) · `data_quality_score` (noise, \|r\| ≤ 0.004) · `station_name` (1:1 with `station_id`) · `country` (a function of `basin`) · `day` (noise, r = 0.0007) · `latitude`, `longitude` (constant within station) |
| 5 | Minority class (%) | **2.166 %** (9,889 / 456,550) — imbalance **45.2 : 1**; majority-class accuracy 97.83 % |
| 6 | Size of the training and test sets | train **310,500** (2000–2016) · validation **73,050** (2017–2020) · test **73,000** (2021–2024) — 73,050 in the test *year window*, minus the 50 last-day rows the one-day shift removes |
| 7 | Most correlated numerical pair and its value | `rainfall_mm` ↔ `river_level_m`: **Pearson 0.9604**, **Spearman 0.8689** |
| 8 | Rows affected by the outlier strategy | **0 rows removed, 0 winsorised** (by design) · **9,274 rows (2.031 %)** flagged above the 56.30 mm rainfall fence, of which **100.000 %** are floods |
| 9 | Variance explained by PC1 + PC2 | **70.26 %** (PC1 55.36 %, PC2 14.90 %) on the 310,500 × 21 scaled training matrix |
| 10 | `shape` of train and test after the pipeline | **train `(310500, 21)`** · **test `(73000, 21)`** · `np.isnan(...).sum() = 0` on both · 8 raw columns → 21 |

### Closing note for the modelling phase

The dataset is clean, stationary and easy to split — and almost all of its signal is **seasonal
persistence plus accumulated water**, not geography. Three consequences carry into the
Classification deliverable:

1. **Judge everything inside the monsoon window.** A global AUC of 0.87 is mostly the calendar;
   the honest baseline is the 5.17 % monsoon base rate.
2. **The window features are the bet.** F7 says a single wet day is often the event, not its
   precursor, so accumulated rainfall, rising river level and soil-moisture *anomaly* are where the
   remaining signal lives.
3. **Report PR-AUC, and set the threshold for a stated recall.** With a 45.2 : 1 imbalance and
   isolated single-day events, a warning system is judged on the floods it misses.

Every number in this report is printed by the script named in its section, and the scripts run from
a clean checkout with the dataset in place — see [Submission](#submission) for the exact commands.
**No model was trained in this deliverable.**

---

## Submission

- Report at `docs/projects/eda/index.md`, headings in the order of the statement.
- Code in `docs/projects/eda/code/` — one script per stage, run offline; figures committed
  under `docs/projects/eda/figures/`.
- Every number quoted in the prose is printed by the script named in the same section.

### Reproducing this report

From the repository root, with the dataset at `docs/projects/data/ASIA_FLOOD_25YEAR_DATASET/`:

```bash
python -m pip install -r requirements.txt          # pandas, numpy, scikit-learn, matplotlib, seaborn, umap-learn
cd docs/projects/eda/code
python data.py            # panel + target + splits
python t1_inspection.py   # stage 1
python t2_numeric.py      # stage 2A
python t3_categorical.py  # stage 2B
python t4_bivariate.py    # stage 3
python pipeline.py        # stage 4C
python t5_pipeline.py     # stage 4A + 4C
python t6_reduction.py    # stage 4B
```

Every script is deterministic (`random_state=42`) and writes its figures into
`../figures/`. The originals are committed, so the report renders without running anything.

---

## Rubric

| Item | Points | Where it is answered |
| --- | :---: | --- |
| **1A.** Data dictionary | 0.5 | Stage 1A |
| **1B.** Quality | 0.75 | Stage 1B |
| **1C.** Target | 0.25 | Stage 1C |
| **1D.** Train and test | 0.5 | Stage 1D |
| **2A.** Numerical | 0.75 | Stage 2A |
| **2B.** Categorical | 0.75 | Stage 2B |
| **3A.** Numerical × numerical | 0.75 | Stage 3A |
| **3B.** Categorical × target | 0.75 | Stage 3B |
| **3C.** Numerical × categorical | 0.5 | Stage 3C |
| **4A.** Strategies | 1.5 | Stage 4A |
| **4B.** Dimensionality reduction | 1.0 | Stage 4B |
| **4C.** Pipeline | 1.0 | Stage 4C |
| **5.** Synthesis | 1.0 | Stage 5 |
| | **10** | |
