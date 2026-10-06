# EDA Deliverable — What the Statement Asks, and What We Do

Spec source: <https://insper.github.io/ann-dl/2026.2/projects/eda/#rubric>
(2026.2, English edition — fetched and transcribed; the 2026.2 page is **not** vendored under
`specs/`, only the 2025.2 and template versions are, so this fetch is the authoritative copy.)

Companion note: [`01-dataset-findings.md`](01-dataset-findings.md).

---

## 0. The deliverable at a glance

| | |
|---|---|
| **Deadline** | 08 Oct 2026 (Thu), commits until 23:59 |
| **Weight** | 20 % of the project grade |
| **Team** | up to 3 people |
| **Task type** | **classification** (the plan fixes it: one-day-ahead flood, yes/no) |
| **Total** | 10 points, scored in full / half / zero per item |
| **Prerequisite** | an **approved proposal**. Without it, the EDA is not graded |
| **Hard rule** | **No model is trained in this deliverable.** |

### Where it lives

```
Neural-Networks-Deep-Learning/
└── docs/projects/eda/
    ├── index.md          # THE report — front matter + headings in statement order
    ├── code/             # standalone .py scripts, one per stage, run offline
    └── figures/          # committed PNGs: fig1.png, fig2.png, ...
```

`docs/projects/eda/` **does not exist yet** — it must be created, and `mkdocs.yml` must get a
nav entry (`Projects` is currently a single page pointing at `./projects/index.md`).

Front matter — mandatory, exact keys (per `PROJECT_STRUCTURE.md` and the statement):

```yaml
---
project: eda
task: classification          # or regression
dataset: https://www.kaggle.com/datasets/aliahmadmphil/asia-flood-25-year-flood-risk-atlas
team:
  - Luigi Lopes
ai_use: "AI-assisted: analysis scripts drafted with an AI agent; every number verified by the team"
---
```

> `PROJECT_STRUCTURE.md` lists the project front matter as `project` + `ai_use`; the statement
> adds `task`, `dataset` and `team`. **Include all five** — the statement is the stricter spec,
> and `ai_use` must honestly disclose AI collaboration either way.

### Mechanics of the repo (from `PROJECT_STRUCTURE.md`)

- Code lives in `code/*.py` and is **embedded into the report** with the snippets extension:
  ```` ```python\n--8<-- "docs/projects/eda/code/<file>.py"\n``` ````
- Figures are **committed PNGs** under `figures/`, referenced as
  `![Figure N](figures/figN.png)`. mkdocs never executes anything; scripts are run locally.
- `mkdocs build --strict` must pass before the work is called done.

---

## 1. Non-negotiable rules (they cost points directly)

| # | Rule | Where it bites us |
|---|---|---|
| G1 | Python with `pandas`, `numpy`, `matplotlib`/`seaborn`, `scikit-learn` and **`umap-learn`** | `umap-learn` was missing from `.venv`; now installed (0.5.12) |
| G2 | **Fixed seed, `random_state=42`** everywhere | split, PCA, t-SNE, UMAP, any sampling |
| G3 | **At most 3 figures per item in stages 2 and 3** | 5 items (2A, 2B, 3A, 3B, 3C) → **≤ 15 figures total** there. Choosing *what deserves a figure* is itself graded |
| G4 | Figures **numbered**, with **title, labelled axes and legend** | every axis needs a unit: mm, m, %, °C |
| G5 | **Every number asked for appears in the text**, not only in the output | a number that only exists in a printout scores half |
| G6 | The **same dataset** carries over to the Classification deliverable | it does — this dataset, this target |
| G7 | Stage 1D split happens **before any preprocessing decision** | no statistic may be computed on train+test combined |
| G8 | Report headings **in the order of the statement's page** | stage 0→5, then Submission, then Rubric |

**"At most 3 figures per item" is a budget, not a target.** The rubric rewards figures
"chosen with a justification and interpreted". A 4th figure in 2A is a *deduction*; a
well-chosen 2 is full marks.

---

## 2. Stage-by-stage: what full marks requires, and our answer

### Stage 0 — Proposal · *prerequisite*

Submit the dataset via the Google Form: URL, size, target, counts of numerical and categorical
features, 2–3 sentence motivation, first risk. Requirements: tabular and public, ≥ 1,000
instances and ≥ 5 features, numerical **and** categorical present, target in the file, and
**not a classical dataset**.

**Our dataset:** ASIA-FLOOD 25-Year Flood Risk Atlas, Kaggle, CC BY-SA 4.0.
`station_daily_data_full.csv` — **456,600 × 21** (456,600 ≥ 1,000 ✓, 21 ≥ 5 ✓, numerical ✓,
categorical ✓, target in file ✓, not classical ✓).

**Action:** confirm the submission is recorded. It is the gate on the whole grade.

---

### Stage 1 — Initial inspection · **2.0 pts**

#### 1A. Data dictionary — 0.5
*Full marks: source, what each row is, and meaning, type and unit of each feature.*

- **Source:** Kaggle, by aliahmadmphil, CC BY-SA 4.0, `documentation/metadata.json` v3.0.
  **State plainly that it is synthetic** ("Generated on: 2026-07-23") so no result reads as a
  claim about real Asian hydrology.
- **What a row is:** *one monitoring station on one calendar day.* 50 stations × 9,132 days
  (2000-01-01 → 2024-12-31) = 456,600 rows, a **balanced daily panel**.
- **A 21-row table**: column · meaning · dtype · unit · role (id / static / calendar /
  measurement / outcome-derived / label). Units: `rainfall_mm` mm·day⁻¹, `river_level_m` m,
  `soil_moisture_percent` %, `temperature_celsius` °C, `latitude`/`longitude` decimal degrees.
- Name the three **outcome-derived** columns and commit to the drop (see 3A/R1 below).
- Also document the two companion files (`vulnerability_data.csv` 10 × 18, per basin;
  `asia_flood_geospatial.geojson` 10 basin polygons) and `station_daily_data.csv`
  (a 100,000-row **random sample** — say it is *not* used and why).
- **Correct the GeoJSON description**: `docs/projects/index.md` calls it "50 station
  coordinates"; it is **10 basin polygons**.

#### 1B. Quality — 0.75
*Full marks: missing values as count and %, inconsistencies quantified, leakage columns investigated.*

- **Missing values:** a full 21-row table of count and % — every entry **0 (0.00 %)**.
  That *is* the finding; do not manufacture imputation needs from it.
- **Duplicates:** 0 full-row duplicates, 0 duplicated `(station_id, date)`. Panel completeness:
  9,132 rows for all 50 stations, no date gaps.
- **Impossible / inconsistent values, quantified:**
  - **32 rows** where `flood_risk_score == 7.00` → `severity_level == "Extreme"` but
    `flood_event_occurred == 0`. The dataset's one genuine internal contradiction.
  - `soil_moisture_percent` **censored at exactly 100.00**.
  - `latitude`/`longitude` constant within station (1 unique value each); station names
    contradict coordinates (a "Luang Prabang" in Thailand; Mekong stations tagged Vietnam).
  - `basin → country` is a function (1 country per basin) → `country` is redundant.
- **Columns to drop, each with a reason:** `severity_level` and `flood_risk_score` (leakage),
  `data_quality_score` (noise, |r| ≤ 0.004 with every other numeric column), `station_name` (1:1 with
  `station_id`), `country` (function of `basin`), `day` (noise), `latitude`/`longitude`
  (constants per station).
- **Leakage investigation — the centrepiece.** Report the exact rule, verified on all 456,600
  rows: `flood_event_occurred == (flood_risk_score > 7.0)`, ROC-AUC **1.0000**; and
  `severity_level == cut(flood_risk_score, (1,3,5,7,10])`. Then the *honest nuance*: for a
  one-day-ahead target the risk score is not strictly leakage, but it is **redundant**
  (r = 0.934 with `soil_moisture_percent`) and as a next-day predictor scores **0.8088 AUC**,
  *below* soil moisture's 0.8364 — so excluding it costs nothing.

#### 1C. Target — 0.25
*Full marks: target distribution analysed — imbalance or skewness.*

- **Reformulate the target for the project:** `y(t) = flood_event_occurred(t+1)`, computed with
  `groupby(station_id).shift(-1)`. 456,550 usable rows (the last day of each station has no *t+1*).
- 9,889 positives / 446,661 negatives → **2.166 %**, imbalance **45.2 : 1**.
  Majority-class accuracy is **97.83 %** — state it once, then never use accuracy.
- Show the raw same-day label too (2.1658 %), and note they are numerically identical.
- **The seasonal gate is part of the target's distribution:** 0 floods in Jan–Apr/Nov/Dec,
  **95.0 % of floods in Jul–Sep**, flood rate 5.170 % inside the monsoon vs 0.00038 % outside.
  Any honest metric must be reported **inside the monsoon window**.

#### 1D. Train and test — 0.5
*Full marks: fixed seed, stratified (or temporal) when appropriate, done before any preprocessing decision.*

- **Temporal is the appropriate choice and must be argued** — not stratified-random. Rows one
  day apart are near-duplicates, and with a 45:1 label a random split leaks flood days across
  the boundary. `random_state=42` fixed; split is by date, **identical dates for all stations**.

| Split | Years | Rows in window | Usable rows (next-day) | Floods | Next-day rate |
|---|---|---|---|---|---|
| train | 2000–2016 | 310,500 | **310,500** | 6,566 | 2.1147 % |
| validation | 2017–2020 | 73,050 | **73,050** | 1,620 | 2.2177 % |
| test | 2021–2024 | 73,050 | **73,000** | 1,703 | 2.3329 % |

- ⚠️ **The test split loses 50 rows**: `shift(-1)` drops the last day of each station
  (2024-12-31), and all 50 fall in the test window. Quote **73,000** as the test set size and
  explain the difference. 310,500 + 73,050 + 73,000 = 456,550.
- Rates are **stable across splits** (2.11 → 2.33 %) and all 50 stations appear in each, so no
  split is starved. The mild upward drift is why the test rate is quoted.
- **From here on, every preprocessing statistic is fitted on train only.** Say it explicitly
  in 1D and honour it in stage 4.

---

### Stage 2 — Univariate analysis · **1.5 pts**

#### 2A. Numerical — 0.75 (*≤ 3 figures*)
*Full marks: statistics for all numerical features and figures chosen with a justification and interpreted.*

- **Descriptive table for all 8 numerical columns**: count, mean, std, min, quartiles, max,
  **plus skewness and kurtosis** (the shape is what the stage is about).
- Figures (justify the choice, then interpret):
  1. **Histograms of the four measurements, raw vs `log1p`** — proves the skew fix
     (`rainfall` skew 10.51 → ≈0), which motivates the 4A transform.
  2. **Boxplots of the four measurements** — shows the IQR fence sitting at 56.30 mm with the
     tail reaching 1,775.72 mm.
  3. **`temperature_celsius` distribution** — the one variable that is *not* skewed
     (skew 0.006, kurtosis −1.20), flat/bimodal from a seasonal sinusoid.
- State in text: `rainfall_mm` skew **10.51** kurt **158.20**; `river_level_m` skew **9.53**
  kurt **140.61**; `soil_moisture_percent` skew 0.86, censored at 100.00;
  `temperature_celsius` skew 0.006. `latitude`/`longitude`/`data_quality_score` are near-uniform.

#### 2B. Categorical — 0.75 (*≤ 3 figures*)
*Full marks: frequencies and cardinality for all, rare and high-cardinality categories pointed out.*

- **Frequency + cardinality table for all 7 categorical columns**: `station_id` (50),
  `station_name` (50), `basin` (10), `country` (7), `severity_level` (4, leakage — flagged),
  `season` (4), `monsoon_season` (2).
- **Point out that there are *no* rare categories.** Every station is exactly 2.00 %, every
  basin exactly 10.00 %, every country 10 % or 20 %. The only sub-1 % level is
  `severity_level = "Extreme"` at 2.173 % — and that is the leaky column. This is the honest
  finding, and it is what makes the "unseen category in the test set" strategy in 4A cheap.
- Figures: (1) bar chart of basin/station flood counts, (2) bar chart of the calendar
  categories (`season`, `monsoon_season`), (3) cardinality bar. Keep to ≤ 3.

---

### Stage 3 — Bivariate and multivariate analysis · **2.0 pts**

#### 3A. Numerical × numerical — 0.75 (*≤ 3 figures*)
*Full marks: correlation with a justified method and redundant pairs pointed out.*

- **Justify Spearman.** All four measurements are heavily right-skewed (skew 9.5–10.5), so
  Pearson is dominated by a handful of extreme monsoon days. **Report Pearson side by side**
  to show the gap — that *is* the justification.
- **The redundant pair:** `rainfall_mm` ↔ `river_level_m`, **Pearson 0.9604**, Spearman 0.8688.
  Second: `soil_moisture_percent` ↔ `temperature_celsius` Spearman 0.7500 (a seasonal artefact,
  both track `day_of_year`).
  Also flag `flood_risk_score` ↔ `soil_moisture_percent` **0.934** and `month` ↔ `day_of_year` **0.997**.
- Argue **why both of the pair stay**: they diverge for the next-day target (soil moisture
  0.8364 vs rainfall 0.8091 vs river level 0.7948 AUC) and the ablation decides.
- Figures: (1) Spearman heatmap (lower) vs Pearson (upper), (2) scatter rainfall × river level
  coloured by the next-day target, (3) scatter rainfall × river level on `log1p` axes to show
  the linearity is tail-driven.

#### 3B. Categorical × target — 0.75 (*≤ 3 figures*)
*Full marks: relation to the target shown and each figure with an explicit conclusion.*

- **Flood rate per category**, for `basin` (1.947 %–2.470 %), `country`, `season`, `monsoon_season`
  (5.170 % vs 0.00038 %), `station_id` (1.63 %–2.58 %), `month`.
- **Every figure ends with one explicit conclusion** — a sentence, in bold or as an admonition,
  stating what the figure shows. This is graded.
- The honest headline: **geography is almost worthless here.** Basin rates span 1.27×, station
  rates 1.58×, against a 2.17 % mean. The only categorical that matters is the **monsoon gate**.
- Figures: (1) flood rate by month with the monsoon window shaded, (2) flood rate by basin with
  the overall mean as a reference line, (3) flood rate by station.

#### 3C. Numerical × categorical — 0.5 (*≤ 3 figures*)
*Full marks: grouped boxplots interpreted in **location** and **spread**.*

- Grouped boxplots of each measurement **by `monsoon_season`** and **by `basin`**.
- Interpret **both** axes explicitly: monsoon shifts rainfall **location** from ~7.1 mm to
  38.6 mm **and spreads it** (within-monsoon std 60.0 vs 3.6 — a 16× change in spread).
  That spread change is the reason a plain standardiser is not enough and `log1p` comes first.
- By basin: **location and spread are essentially identical across all 10** — between-station
  rainfall variance is **0.0423 %** of the total. Report the null result.
- Figures: (1) rainfall by monsoon (raw + log1p), (2) the four measurements by basin,
  (3) river level by basin.

---

### Stage 4 — Preprocessing · **3.5 pts** (the heaviest stage)

*Every choice must point at the finding that motivates it: "we standardised because the features
range from 0–1 to 0–10⁵ (Table 3)", never a bare "we standardised".*

#### 4A. Strategies — 1.5
*Full marks: all four strategies chosen, each tied to the finding that motivates it and fitted on train.*

| Strategy | Our decision | Finding that motivates it |
|---|---|---|
| **Missing values** | `SimpleImputer(strategy="median")` in the numeric branch, so the pipeline cannot crash on a partial future day. **No imputation is needed on this data — 0 missing values in 456,600 × 21.** | 1B missing-value table: 0 (0.00 %) |
| **Outliers** | **Do not drop, do not winsorise.** `log1p` + an explicit `is_extreme_rainfall` flag. | **100.000 % of rows above the rainfall IQR fence (56.30 mm) are floods**; 9,274 rows. Removing them deletes the entire positive class |
| **Categorical encoding** | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for `basin` (10 levels) and `season` (4). **Justify `handle_unknown="ignore"` by naming the concrete case:** a basin absent from train maps to all-zeros instead of raising — and with 10 basins the added width is trivial. Say that no rare category exists to collapse (every station 2.00 %, every basin 10.00 %). | 2B frequency table |
| **Scaling** | `log1p` on `rainfall_mm` + `river_level_m` **first**, then `StandardScaler` on everything. **The destination is a neural network**, so features must be on comparable scales for gradient descent. | 2A skew 10.51/9.53; 3C monsoon spread ×16; measurements span 0.39–1,775.72 |

Also address: `latitude`/`longitude` are constants per station (not features) and
`data_quality_score` is noise (drop). **All fitted on train only.**

#### 4B. Dimensionality reduction — 1.0
*Full marks: PCA with variance **and loadings**; t-SNE and UMAP with **2 parameter values each**;
comparison of the three.*

- Three 2D projections of the **scaled training features**, **coloured by the target**.
- **PCA:** cumulative explained-variance curve **and a reading of the `loadings`** — which
  original features load on PC1/PC2. Report **PC1 + PC2 variance** (results-table row 9).
- **t-SNE and UMAP, ≥ 2 values each** — e.g. t-SNE `perplexity ∈ {5, 30, 50}`,
  UMAP `n_neighbors ∈ {5, 15, 50}`. **On 310,500 training rows, sample** — a stratified
  sample (e.g. 20,000–30,000 rows, all positives) with `random_state=42`, and say the sample
  was taken. Budget the runtime: UMAP on 30k rows is minutes, not seconds.
- **The required comparison:** what do the nonlinear projections reveal that PCA does not?
  Then the mandatory caveat: **in t-SNE and UMAP, cluster sizes and inter-cluster distances
  have no direct reading.** With a 2 % positive class expect the classes to overlap heavily —
  that is the honest finding and it is consistent with PR-AUC ≈ 0.08.
- **`umap-learn` is required by the statement.** Installed at 0.5.12; use `n_jobs=1`.

#### 4C. Pipeline — 1.0
*Full marks: fitted on train only, importable, no `NaN`, `shape` and feature names reported.*

- An **importable** module under `code/` exposing a fitted `Pipeline`/`ColumnTransformer`,
  in the statement's skeleton shape:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocess = ColumnTransformer([
    ("num", Pipeline([...]), numeric_features),
    ("cat", Pipeline([...]), categorical_features),
])

X_train_t = preprocess.fit_transform(X_train)   # fit on train only
X_test_t  = preprocess.transform(X_test)        # test is only transformed
```

- **Report:** `X_train_t.shape`, `X_test_t.shape`, `np.isnan(X_train_t).sum() == 0`, and the
  **feature names** after the transform (`get_feature_names_out`).
- The seam: `code/pipeline.py` is imported by 4B's script and by the Classification deliverable
  later — so it takes a raw dataframe and returns transformed arrays plus names, and it must
  **not** be re-fitted outside `fit`.

---

### Stage 5 — Synthesis · **1.0 pt**

*Full marks: findings backed by figures, risks with a plan, code that runs from a clean clone,
and a complete summary.*

- **Main findings**, each with the figure or table that supports it — a short list, no new analysis.
- **Risks for modelling with a handling plan** — reuse R1–R11 from
  [`01-dataset-findings.md`](01-dataset-findings.md) §11, trimmed to the ones that matter.
- **Code runs from a clean clone with the dataset in place.** State the exact command and the
  expected paths.
- **The results-summary table, filled in** (below).

---

## 3. Results summary table — the index of numbers the report must already contain

| # | Results summary | Value |
|:--:|---|---|
| 1 | Dataset, task and target | ASIA-FLOOD 25-Year (Kaggle, CC BY-SA 4.0), synthetic · binary classification · `flood_event_occurred(t+1)` per station |
| 2 | Instances × features (numerical / categorical) | 456,600 × 21 raw → 456,550 usable · **8 numerical / 7 categorical** → **6 numerical / 10 categorical** after the drop |
| 3 | Column with the most missing values and its percentage | **none — 0 missing in all 21 columns (0.00 %)** |
| 4 | Dropped columns and the reason | `severity_level`, `flood_risk_score` (leakage: `risk > 7.0` ⇒ flood, AUC 1.0000) · `data_quality_score` (noise, \|r\| ≤ 0.004) · `station_name` (1:1 with `station_id`) · `country` (function of `basin`) · `day` (noise) · `latitude`, `longitude` (constant per station) |
| 5 | Minority class (%) | **2.166 %** (9,889 / 456,550) — imbalance **45.2 : 1** |
| 6 | Size of the training and test sets | train 310,500 (2000–2016) · validation 73,050 (2017–2020) · test **73,000** (2021–2024) — 73,050 in the test year window, minus the 50 last-day rows the one-day shift removes |
| 7 | Most correlated numerical pair and its value | `rainfall_mm` ↔ `river_level_m`: **Pearson 0.9604**, Spearman 0.8688 |
| 8 | Rows affected by the outlier strategy | **0 rows removed** (by design) · 9,274 rows (2.03 %) flagged as extreme rainfall — **100 % of them are floods** |
| 9 | Variance explained by PC1 + PC2 | *from stage 4B* |
| 10 | `shape` of train and test after the pipeline | *from stage 4C* |

---

## 4. Figure budget

| Stage | Item | Max figures | Planned |
|---|---|:--:|:--:|
| 2 | 2A numerical | 3 | 3 |
| 2 | 2B categorical | 3 | 2 |
| 3 | 3A num × num | 3 | 3 |
| 3 | 3B cat × target | 3 | 3 |
| 3 | 3C num × cat | 3 | 3 |
| 4 | 4B PCA / t-SNE / UMAP | — | 6–8 |
| | **Total** | | **≈ 22** |

Stage 1 and stage 4 figures are not capped by G3, but keep them purposeful. Every figure:
numbered, titled, labelled axes with units, legend, and one explicit conclusion in the text.

---

## 5. Task decomposition for the fan-out

Every task owns **one script** in `docs/projects/eda/code/`, **its own figures**, and **one
markdown fragment** in `docs/projects/eda/sections/`. Fragments are concatenated into
`index.md` at the end; no agent edits `index.md` directly.

All fragment paths and figure names are fixed by the assembly step, so agents never collide.

| Task | Stage | Rubric | Script | Figures | Fragment |
|---|---|:--:|---|---|---|
| **T1** Initial inspection | 1A–1D | 2.0 | `t1_inspection.py` | `fig01_*` | `s1-inspection.md` |
| **T2** Univariate numerical | 2A | 0.75 | `t2_numeric.py` | `fig10_*`–`fig12_*` | `s2a-numeric.md` |
| **T3** Univariate categorical | 2B | 0.75 | `t3_categorical.py` | `fig20_*`–`fig21_*` | `s2b-categorical.md` |
| **T4** Bivariate / multivariate | 3A–3C | 2.0 | `t4_bivariate.py` | `fig30_*`–`fig38_*` | `s3-bivariate.md` |
| **T5** Strategies + pipeline | 4A, 4C | 2.5 | `t5_pipeline.py` + `pipeline.py` | `fig40_*` | `s4a-strategies.md`, `s4c-pipeline.md` |
| **T6** Dimensionality reduction | 4B | 1.0 | `t6_reduction.py` | `fig50_*`–`fig57_*` | `s4b-reduction.md` |
| **T7** Synthesis + assembly | 5 | 1.0 | — | — | `index.md` |
| **V1** Verification | all | — | — | — | `verification-report.md` |

**Shared seam (written once, before the fan-out):** `code/data.py` — `load_raw()`,
`add_next_day_target()`, `SPLITS` (year boundaries), `numeric_features`, `categorical_features`,
`DROPPED`, `set_seed()`. T5's `pipeline.py` imports from it. This removes the only real
coordination risk between agents.

**Dependencies:**
- T1 is independent; T2, T3, T4 are independent of each other and of T1.
- **T5 depends on T1's split contract** (fixed in `data.py`, so no blocking).
- **T6 depends on T5's `pipeline.py`** — sequence it after T5, or stub the import.
- T7 depends on all fragments.
- V1 runs last, against the assembled report **and** the scripts' outputs.

---

## 6. Definition of done — the self-check before submission

1. `docs/projects/eda/index.md` exists with the exact front matter and headings **in statement
   order** (0 → 1 → 2 → 3 → 4 → 5 → Submission → Rubric).
2. Every script in `code/` runs from a clean clone with the dataset in place, seed 42, and
   regenerates every committed figure byte-identically.
3. `np.isnan(X_train_t).sum() == 0`; `shape` and feature names reported in the text.
4. No statistic anywhere was fitted on validation or test.
5. Every figure: numbered, titled, labelled axes **with units**, legend, and one explicit
   conclusion in the prose.
6. Every number in the results table appears in the report's **prose**, not only in a printout.
7. ≤ 3 figures per item in stages 2 and 3.
8. The leakage section proves the rule on all 456,600 rows and states the one-day-ahead nuance.
9. The monsoon gate and the 45:1 imbalance are stated wherever a metric could mislead.
10. **No model is trained.** No classifier, no regressor, no fitted estimator other than
    preprocessing transformers and the PCA/t-SNE/UMAP projections.
11. `mkdocs build --strict` passes; nav updated.
12. `ai_use` in the front matter honestly describes the AI collaboration.

---

## 7. Scoring reality check

| Item | Pts | Our confidence | Why |
|---|:--:|---|---|
| 1A Data dictionary | 0.5 | high | 21-column table, units, synthetic disclosed |
| 1B Quality | 0.75 | **high** | the leakage rule is exact and verified on every row |
| 1C Target | 0.25 | high | both labels + imbalance + seasonal gate |
| 1D Train and test | 0.5 | high | temporal, argued, stable rates |
| 2A Numerical | 0.75 | high | full stats + skew/kurtosis, 3 justified figures |
| 2B Categorical | 0.75 | medium | must frame "no rare categories" as the finding |
| 3A Num × num | 0.75 | high | justified Spearman + Pearson gap + named redundant pair |
| 3B Cat × target | 0.75 | **medium** | risk of a null result reading as thin — must be argued |
| 3C Num × cat | 0.5 | medium | location **and spread** required, by basin it is a null |
| 4A Strategies | 1.5 | **high** | the outlier story (100 % of flagged rows are floods) is exceptional |
| 4B Dim. reduction | 1.0 | medium | runtime on 310k rows; needs 2 values each for t-SNE **and** UMAP |
| 4C Pipeline | 1.0 | high | importable, train-only, no NaN, shapes + names |
| 5 Synthesis | 1.0 | medium | depends on every number matching its script |
| | **10** | | |
