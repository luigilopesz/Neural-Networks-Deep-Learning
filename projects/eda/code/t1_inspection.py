"""T1 - EDA stage 1 "Initial inspection" for the ASIA-FLOOD 25-Year panel.

Covers rubric items 1A (data dictionary), 1B (quality + the leakage investigation),
1C (target) and 1D (train/validation/test). Prints every number the report fragment
``sections/s1-inspection.md`` cites, and writes the three ``fig01_*`` figures.

Run it from this directory with the repo-root .venv:

    cd docs/projects/eda/code
    ..\\..\\..\\..\\.venv\\Scripts\\python.exe t1_inspection.py

Deterministic: no sampling, no estimator other than ``roc_auc_score`` on single raw
columns (a rank statistic, not a fit). NO MODEL IS TRAINED in this deliverable.
"""

from __future__ import annotations

import json
import os

# The DSH file sandbox forbids named pipes, so any joblib/OMP thread pool dies with
# PermissionError. Pin everything to one thread *before* numpy/sklearn are imported.
for _var in ("OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from data import (  # noqa: E402
    CATEGORICAL,
    DATA_DIR,
    DROPPED,
    FIGDIR,
    NUMERIC,
    RANDOM_STATE,
    SEED,
    SAMPLE_CSV,
    SPLITS,
    TARGET,
    TARGET_NEXT,
    add_next_day_target,
    class_balance,
    configure_env,
    finish,
    label_split,
    load_geojson,
    load_raw,
    load_vulnerability,
    savefig,
)

configure_env()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def head(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def kv(label: str, value) -> None:
    print(f"  {label:<52s} {value}")


# ======================================================================================
# Load
# ======================================================================================
head("0. Load")
df = load_raw()
N_ROWS, N_COLS = df.shape
kv("rows x cols", f"{N_ROWS:,} x {N_COLS}")
kv("dtypes", dict(df.dtypes.astype(str)))
kv("SEED / RANDOM_STATE", f"{SEED} / {RANDOM_STATE}")
kv("TARGET / TARGET_NEXT", f"{TARGET} / {TARGET_NEXT}")
kv("SPLITS", dict(SPLITS))

# ======================================================================================
# 1A. Data dictionary
# ======================================================================================
head("1A. Source and provenance (read from the dataset's own files, not hard-coded)")
_meta = json.loads((DATA_DIR.parent / "documentation" / "metadata.json").read_text(encoding="utf-8"))
for _k, _v in _meta.items():
    kv(f"metadata.json: {_k}", _v)
for _line in (DATA_DIR.parent / "README.md").read_text(encoding="utf-8").splitlines():
    if "Generated on:" in _line or "CC BY-SA" in _line:
        kv("README.md:", _line.strip())
kv("=> the panel is SYNTHETIC, not an observation of real hydrology", True)

head("1A. Data dictionary")

#: (column, meaning, unit, role).  Roles: identifier / static-per-station / calendar /
#: measurement / outcome-derived / label.
DICTIONARY = [
    ("station_id", "Monitoring-station key, ASIA_0001..ASIA_0050", "-", "identifier"),
    ("station_name", "Place name of the station; 1:1 with station_id", "-", "identifier"),
    ("basin", "River basin the station belongs to", "-", "static-per-station"),
    ("country", "Country label attached to the basin", "-", "static-per-station"),
    ("latitude", "Station latitude, repeated on all of the station's rows",
     "decimal degrees", "static-per-station"),
    ("longitude", "Station longitude, repeated on all of the station's rows",
     "decimal degrees", "static-per-station"),
    ("date", "Calendar day of the observation", "-", "calendar"),
    ("year", "Calendar year", "year", "calendar"),
    ("month", "Calendar month", "month (1-12)", "calendar"),
    ("day", "Day of month", "day (1-31)", "calendar"),
    ("day_of_year", "Ordinal day inside the year", "day-of-year (1-366)", "calendar"),
    ("rainfall_mm", "Daily rainfall accumulation at the station", "mm/day", "measurement"),
    ("river_level_m", "River stage at the station gauge", "m", "measurement"),
    ("soil_moisture_percent", "Volumetric soil moisture, censored at 100.00", "%", "measurement"),
    ("temperature_celsius", "Daily mean air temperature", "degC", "measurement"),
    ("flood_risk_score", "Synthesised composite risk index on a 0.01 grid",
     "index 1-10 (dimensionless)", "outcome-derived"),
    ("flood_event_occurred", "Same-day flood label: 1 if the station flooded that day",
     "0/1", "label"),
    ("severity_level", "4-level binning of flood_risk_score",
     "-", "outcome-derived"),
    ("season", "Meteorological season, deterministic from month",
     "-", "calendar"),
    ("monsoon_season", "1 for Jun-Oct, else 0; deterministic from month",
     "0/1", "calendar"),
    ("data_quality_score", "Per-row quality score; uncorrelated with every other column",
     "index 0.88-0.99", "measurement"),
]
assert len(DICTIONARY) == N_COLS, "dictionary must cover every column"
assert [c for c, *_ in DICTIONARY] == list(df.columns), "dictionary column order mismatch"


def levels_or_range(s: pd.Series) -> str:
    """Low-cardinality -> the levels; otherwise the observed min..max."""
    if s.nunique() <= 12:
        return "levels: " + ", ".join(str(v) for v in sorted(s.unique().tolist()))
    if pd.api.types.is_datetime64_any_dtype(s):
        return f"range: {s.min():%Y-%m-%d} .. {s.max():%Y-%m-%d}"
    if pd.api.types.is_numeric_dtype(s):
        return f"range: {s.min():g} .. {s.max():g}"
    return f"{s.nunique()} distinct values"


print(f"{'column':<22}{'dtype':<16}{'n_uniq':>7}  {'unit':<28}{'role':<20}observed")
for col, _meaning, unit, role in DICTIONARY:
    s = df[col]
    dt = str(df.dtypes[col])
    print(f"{col:<22}{dt:<16}{s.nunique():>7}  {unit:<28}{role:<20}{levels_or_range(s)}")

print("\nmeanings:")
for col, meaning, _unit, _role in DICTIONARY:
    print(f"  {col:<22} {meaning}")

head("1A. Companion files")
vuln = load_vulnerability()
kv("vulnerability_data.csv shape", f"{vuln.shape[0]} rows x {vuln.shape[1]} cols")
kv("vulnerability_data.csv basins", f"{vuln['basin'].nunique()} == {df['basin'].nunique()} in the panel")
kv("vulnerability_data.csv columns", list(vuln.columns))
kv("geojson feature count", len(load_geojson()["features"]))
kv("geojson geometry types", sorted({f["geometry"]["type"] for f in load_geojson()["features"]}))
kv("geojson properties", list(load_geojson()["features"][0]["properties"].keys()))

sample = pd.read_csv(SAMPLE_CSV, parse_dates=["date"])
kv("station_daily_data.csv shape", f"{sample.shape[0]:,} x {sample.shape[1]}")
kv("station_daily_data.csv share of full", f"{len(sample) / N_ROWS * 100:.2f} %")
kv("station_daily_data.csv stations", sample["station_id"].nunique())
kv("station_daily_data.csv rows per station", f"{sample.groupby('station_id').size().min()} .. "
                                              f"{sample.groupby('station_id').size().max()}")
_full_keys = set(map(tuple, df[["station_id", "date"]].astype(str).to_numpy()))
_sample_keys = set(map(tuple, sample[["station_id", "date"]].astype(str).to_numpy()))
kv("sample (station_id, date) pairs found in full panel",
   f"{len(_sample_keys & _full_keys):,} / {len(_sample_keys):,}")
kv("sample columns identical to full", list(sample.columns) == list(df.columns))
del sample

# ======================================================================================
# 1B. Quality
# ======================================================================================
head("1B. Missing values - all 21 columns")
miss = df.isna().sum()
miss_tbl = pd.DataFrame({"count": miss, "pct": (miss / N_ROWS * 100).round(4)})
print(miss_tbl.to_string())
kv("total missing cells", f"{int(miss.sum()):,} of {N_ROWS * N_COLS:,} "
                          f"({miss.sum() / (N_ROWS * N_COLS) * 100:.4f} %)")
kv("columns with any missing value", int((miss > 0).sum()))

head("1B. Duplicates and panel completeness")
kv("fully duplicated rows", int(df.duplicated().sum()))
kv("duplicated (station_id, date) pairs", int(df.duplicated(["station_id", "date"]).sum()))
rows_per_station = df.groupby("station_id").size()
kv("rows per station - unique values", rows_per_station.unique().tolist())
kv("rows per station - min / max", f"{rows_per_station.min()} / {rows_per_station.max()}")
expected_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
kv("date range", f"{df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")
kv("distinct dates present", df["date"].nunique())
kv("dates expected in that range", len(expected_days))
kv("dates missing from the range", len(expected_days.difference(pd.DatetimeIndex(df["date"].unique()))))
kv("stations", df["station_id"].nunique())
kv("50 stations x 9,132 days", f"{50 * 9132:,} == {N_ROWS:,}")
kv("every station has the same date set",
   bool(df.groupby("station_id")["date"].nunique().nunique() == 1))
kv("dates per station", int(df.groupby("station_id")["date"].nunique().iloc[0]))

head("1B. Calendar consistency (deterministic re-encodings of `date`)")
ymd = pd.to_datetime(df[["year", "month", "day"]].rename(columns={"day": "day"}))
kv("date != (year, month, day)", int((ymd != df["date"]).sum()))
kv("date.dt.dayofyear != day_of_year", int((df["date"].dt.dayofyear != df["day_of_year"]).sum()))
season_map = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer",
              7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"}
kv("season != f(month)", int((df["month"].map(season_map) != df["season"]).sum()))
kv("monsoon_season != month in Jun..Oct",
   int((df["month"].isin([6, 7, 8, 9, 10]).astype(int) != df["monsoon_season"]).sum()))

head("1B. Inconsistencies, quantified")
r7 = df[df["flood_risk_score"] == 7.0]
kv("rows with flood_risk_score == 7.00", len(r7))
kv("  ... their severity_level", sorted(r7["severity_level"].unique().tolist()))
kv("  ... their flood_event_occurred", sorted(r7[TARGET].unique().tolist()))
kv("  => rows 'Extreme' but not a flood (the internal contradiction)", len(r7))
censored = int((df["soil_moisture_percent"] == 100.0).sum())
kv("soil_moisture_percent == 100.00 (censored ceiling)",
   f"{censored:,} ({censored / N_ROWS * 100:.4f} %)")
kv("soil_moisture_percent max", df["soil_moisture_percent"].max())
for col in ("latitude", "longitude", "station_name", "basin", "country"):
    nu = df.groupby("station_id")[col].nunique()
    kv(f"{col} unique values per station - max", int(nu.max()))
kv("distinct (latitude, longitude) pairs", df[["latitude", "longitude"]].drop_duplicates().shape[0])
kv("duplicated (station_name, latitude, longitude) triples",
   int(df.duplicated(["station_name", "latitude", "longitude"]).sum()))
basin_country = df.groupby("basin")["country"].nunique()
kv("countries per basin - unique values", basin_country.unique().tolist())
kv("stations per basin - unique values", df.groupby("basin")["station_id"].nunique().unique().tolist())
kv("rows per basin - unique values", df.groupby("basin").size().unique().tolist())
kv("country levels / basin levels", f"{df['country'].nunique()} / {df['basin'].nunique()}")
kv("country counts", df["country"].value_counts().to_dict())
mekong = (df.loc[df["basin"] == "Mekong", ["station_id", "station_name", "latitude",
                                           "longitude", "country"]].drop_duplicates()
          .sort_values("station_id"))
print("\nMekong group - names vs coordinates vs country label:")
print(mekong.to_string(index=False))
kv("Mekong latitude span", f"{mekong['latitude'].min():.4f} .. {mekong['latitude'].max():.4f}")
kv("Mekong longitude span", f"{mekong['longitude'].min():.4f} .. {mekong['longitude'].max():.4f}")
luang = df.loc[df["station_name"] == "Luang Prabang",
               ["station_id", "station_name", "latitude", "longitude", "country"]].drop_duplicates()
print(luang.to_string(index=False))
helmand = (df.loc[df["basin"] == "Helmand", ["station_name", "latitude", "longitude"]]
           .drop_duplicates().sort_values("longitude"))
print("\nHelmand group - coordinates:")
print(helmand.to_string(index=False))
kv("Helmand longitude span", f"{helmand['longitude'].min():.4f} .. {helmand['longitude'].max():.4f}")

head("1B. Columns dropped, with the reason (from data.py:DROPPED)")
print(json.dumps(DROPPED, indent=2))
kv("dropped columns", len(DROPPED))
kv("kept NUMERIC", NUMERIC)
kv("kept CATEGORICAL", CATEGORICAL)
kv("model feature count after the drop", len(NUMERIC) + len(CATEGORICAL))

head("1B. Are the dropped columns really noise? (re-derived, not asserted)")
labelled = add_next_day_target(df)
usable = labelled.dropna(subset=[TARGET_NEXT])
y = usable[TARGET_NEXT].astype(int)
_num_others = [c for c in df.select_dtypes("number").columns if c != "data_quality_score"]
_dqs = df[_num_others].corrwith(df["data_quality_score"]).abs()
kv("numeric columns compared against data_quality_score", len(_num_others))
kv("  max |Pearson r|", f"{_dqs.max():.4f} (against {_dqs.idxmax()})")
kv("  Pearson r(data_quality_score, flood_event_occurred) [same-day label]",
   f"{df['data_quality_score'].corr(df[TARGET]):.4f}")
kv("  Pearson r(data_quality_score, next-day target)",
   f"{usable['data_quality_score'].corr(y):.4f}")
kv("  DROPPED records the conservative bound", "|r| <= 0.004")
kv("Pearson r(day_of_month, next-day target)", f"{usable['day'].corr(y):.4f}")
kv("Pearson r(year, next-day target)", f"{usable['year'].corr(y):.4f}")
kv("ROC-AUC(data_quality_score) for the next-day target",
   f"{roc_auc_score(y, usable['data_quality_score']):.4f}")

head("1B. LEAKAGE - is the same-day label recoverable from flood_risk_score?")
rule = (df["flood_risk_score"] > 7.0).astype(int)
kv("rows where flood_event_occurred == (flood_risk_score > 7.0)",
   f"{int((rule == df[TARGET]).sum()):,} / {N_ROWS:,}  -> exact")
kv("max(flood_risk_score | flood == 0)", df.loc[df[TARGET] == 0, "flood_risk_score"].max())
kv("min(flood_risk_score | flood == 1)", df.loc[df[TARGET] == 1, "flood_risk_score"].min())
kv("ROC-AUC(flood_risk_score -> flood_event_occurred)",
   f"{roc_auc_score(df[TARGET], df['flood_risk_score']):.4f}")
print("\nseverity_level x flood_event_occurred:")
print(pd.crosstab(df["severity_level"], df[TARGET], margins=True).to_string())
sev_agg = df.groupby("severity_level")["flood_risk_score"].agg(["min", "max", "size"])
sev_agg["floods"] = df.groupby("severity_level")[TARGET].sum()
print("\nseverity_level vs flood_risk_score:")
print(sev_agg.to_string())

labels = ["Low", "Moderate", "High", "Extreme"]
sev_dataset = df["severity_level"].astype(str)
sev_literal = pd.cut(df["flood_risk_score"], bins=(1, 3, 5, 7, 10), labels=labels).astype(str)
sev_open = pd.cut(df["flood_risk_score"], bins=(1, 3, 5, 7, 10), labels=labels,
                  right=False).astype(str)
sev_leftclosed = pd.cut(df["flood_risk_score"], bins=(1, 3, 5, 7, 10.01), labels=labels,
                        right=False).astype(str)
kv("pd.cut(risk, (1,3,5,7,10]) default right=True reproduces severity_level",
   f"{int((sev_literal == sev_dataset).sum()):,} / {N_ROWS:,} "
   f"({N_ROWS - int((sev_literal == sev_dataset).sum()):,} boundary rows disagree)")
kv("pd.cut(risk, [1,3,5,7,10], right=False) reproduces severity_level",
   f"{int((sev_open == sev_dataset).sum()):,} / {N_ROWS:,} "
   f"({N_ROWS - int((sev_open == sev_dataset).sum()):,} rows fall off the open top edge, "
   f"risk == 10.00 -> NaN)")
kv("pd.cut(risk, (1,3,5,7,10.01], right=False) reproduces severity_level",
   f"{int((sev_leftclosed == sev_dataset).sum()):,} / {N_ROWS:,} -> exact "
   f"(left-closed [1,3) [3,5) [5,7) [7,10])")
print("\nwhere the interval conventions differ (rows sitting exactly on a bin edge):")
for edge in (1.0, 3.0, 5.0, 7.0, 10.0):
    n = int((df["flood_risk_score"] == edge).sum())
    kv(f"  risk == {edge:.2f}", f"{n:,} rows")
kv("  total", f"{sum(int((df['flood_risk_score'] == e).sum()) for e in (1.0, 3.0, 5.0, 7.0, 10.0)):,}")

risk = df["flood_risk_score"]
kv("Pearson r(flood_risk_score, soil_moisture_percent)", f"{risk.corr(df['soil_moisture_percent']):.4f}")
kv("Pearson r(flood_risk_score, river_level_m)", f"{risk.corr(df['river_level_m']):.4f}")
kv("Pearson r(flood_risk_score, rainfall_mm)", f"{risk.corr(df['rainfall_mm']):.4f}")

# ======================================================================================
# 1C. Target
# ======================================================================================
head("1C. Target")
# `labelled`, `usable` and `y` were built at the end of 1B and are reused here.
kv("rows in the panel", f"{N_ROWS:,}")
kv("rows with a t+1 label", f"{len(usable):,}")
kv("rows dropped (last day of each station)", f"{N_ROWS - len(usable)} = "
                                               f"{df['station_id'].nunique()} stations x 1 day")
kv("date of the dropped rows", f"{df.loc[labelled[TARGET_NEXT].isna(), 'date'].min():%Y-%m-%d}")
print("\nnext-day target y(t) = flood_event_occurred(t+1):")
print(class_balance(y).to_string())
kv("positives", f"{int(y.sum()):,}")
kv("negatives", f"{int((1 - y).sum()):,}")
kv("positive rate", f"{y.mean() * 100:.6f} %  -> {y.mean() * 100:.3f} %")
kv("imbalance (negatives : positives)", f"{(1 - y.mean()) / y.mean():.3f} : 1  -> 45.2 : 1")
kv("majority-class accuracy", f"{(1 - y.mean()) * 100:.4f} %  -> {(1 - y.mean()) * 100:.2f} %")
kv("same-day label rate (raw flood_event_occurred)", f"{df[TARGET].mean() * 100:.6f} %")
kv("positives, same-day label", f"{int(df[TARGET].sum()):,}")
kv("floods on 2000-01-01 (dropped by the shift)",
   int(df.loc[df["date"] == "2000-01-01", TARGET].sum()))
kv("floods on 2024-12-31 (the day with no t+1)",
   int(df.loc[df["date"] == "2024-12-31", TARGET].sum()))

head("1C. The seasonal gate - the label is not uniform over the year")

monthly = pd.DataFrame({
    "n": df.groupby("month").size(),
    "floods_same_day": df.groupby("month")[TARGET].sum(),
    "rate_same_day_pct": (df.groupby("month")[TARGET].mean() * 100).round(4),
    "floods_next_day": usable.groupby("month")[TARGET_NEXT].sum().astype(int),
    "rate_next_day_pct": (usable.groupby("month")[TARGET_NEXT].mean() * 100).round(4),
})
print(monthly.to_string())

dead = [1, 2, 3, 4, 11, 12]
jul_sep = [7, 8, 9]
kv("floods in Jan-Apr and Nov-Dec (same-day)", int(df.loc[df["month"].isin(dead), TARGET].sum()))
kv("floods in Jan-Apr and Nov-Dec (next-day)",
   int(usable.loc[usable["month"].isin(dead), TARGET_NEXT].sum()))
same_js = int(df.loc[df["month"].isin(jul_sep), TARGET].sum())
kv("floods in Jul-Sep (same-day)", f"{same_js:,} = {same_js / df[TARGET].sum() * 100:.2f} % of all floods")
next_js = int(usable.loc[usable["month"].isin(jul_sep), TARGET_NEXT].sum())
kv("floods in Jul-Sep (next-day)",
   f"{next_js:,} = {next_js / y.sum() * 100:.2f} % of all floods")
for flag in (1, 0):
    sub = df[df["monsoon_season"] == flag]
    kv(f"flood rate, monsoon_season = {flag}", f"{sub[TARGET].mean() * 100:.6f} % "
                                               f"({int(sub[TARGET].sum()):,} of {len(sub):,})")
_in_rate = df.loc[df["monsoon_season"] == 1, TARGET].mean()
_out_rate = df.loc[df["monsoon_season"] == 0, TARGET].mean()
kv("monsoon : outside flood-rate ratio", f"{_in_rate / _out_rate:.1f} x")
kv("share of all floods by month (same-day)",
   (df.groupby("month")[TARGET].sum() / df[TARGET].sum() * 100).round(2).to_dict())

head("1C. Single-column rank signal for the next-day target (no estimator fitted)")
for col in ["soil_moisture_percent", "rainfall_mm", "flood_risk_score", "temperature_celsius",
            "river_level_m"]:
    kv(f"ROC-AUC({col})", f"{roc_auc_score(y, usable[col]):.4f}")

# ======================================================================================
# 1D. Train and test
# ======================================================================================
head("1D. Why temporal, not stratified-random")


def lag1_within_station(frame: pd.DataFrame, col: str) -> float:
    """Pooled within-station lag-1 autocorrelation (vectorised, no groupby.apply)."""
    s = frame.sort_values(["station_id", "date"])
    x = s[col]
    xl = s.groupby("station_id")[col].shift(1)
    m = xl.notna()
    g = s.loc[m, "station_id"]
    dx = x[m] - g.map(x[m].groupby(g).mean())
    dl = xl[m] - g.map(xl[m].groupby(g).mean())
    return float((dx * dl).sum() / np.sqrt((dx**2).sum() * (dl**2).sum()))


print("pooled within-station lag-1 autocorrelation (rows one day apart are near-duplicates):")
for col in ["temperature_celsius", "soil_moisture_percent", "flood_risk_score", "river_level_m",
            "rainfall_mm"]:
    kv(f"  lag-1 autocorr, {col}", f"{lag1_within_station(df, col):.3f}")


def compare_splits(side: np.ndarray, name: str) -> dict:
    """Quantify how a proposed split cuts the panel apart."""
    s = df.assign(_side=side).sort_values(["station_id", "date"])
    blocks = s.groupby(["station_id", "year"])["_side"].nunique()
    prev = s.groupby("station_id")["_side"].shift(1)
    straddle = int(((s["_side"] != prev) & prev.notna()).sum())
    side_dates = {v: set(s.loc[s["_side"] == v, "date"].unique()) for v in s["_side"].unique()}
    shared = 0
    vals = sorted(side_dates)
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            shared += len(side_dates[vals[i]] & side_dates[vals[j]])
    return {"name": name, "n_sides": len(vals), "blocks_total": len(blocks),
            "blocks_split": int((blocks > 1).sum()), "pairs_straddling": straddle,
            "dates_shared": shared}


_idx = np.arange(N_ROWS)
_rand_side = np.full(N_ROWS, "train", dtype=object)
_rand_side[train_test_split(_idx, test_size=0.20, random_state=SEED, stratify=df[TARGET])[1]] = "test"
_temporal_side = label_split(df["year"]).astype(str).to_numpy()
for res in (compare_splits(_rand_side, "stratified random 80/20, seed 42"),
            compare_splits(_temporal_side, "temporal by date (adopted)")):
    print(f"\n{res['name']}:")
    kv("  (station, year) blocks", f"{res['blocks_total']:,}")
    kv("  blocks split across the boundary", f"{res['blocks_split']:,} "
                                              f"({res['blocks_split'] / res['blocks_total'] * 100:.1f} %)")
    kv("  adjacent-day pairs straddling the boundary", f"{res['pairs_straddling']:,}")
    kv("  calendar dates appearing on more than one side", f"{res['dates_shared']:,}")

head("1D. The adopted split - chronological, identical dates for every station")
same = df.assign(split=label_split(df["year"]))
nextd = usable.assign(split=label_split(usable["year"]))
ORDER = ["train", "validation", "test"]
tbl = pd.DataFrame({
    "rows_same_day": same.groupby("split").size(),
    "floods_same_day": same.groupby("split")[TARGET].sum(),
    "rate_same_day_pct": (same.groupby("split")[TARGET].mean() * 100).round(4),
    "rows_next_day": nextd.groupby("split").size(),
    "floods_next_day": nextd.groupby("split")[TARGET_NEXT].sum().astype(int),
    "rate_next_day_pct": (nextd.groupby("split")[TARGET_NEXT].mean() * 100).round(4),
    "stations": same.groupby("split")["station_id"].nunique(),
    "first_date": same.groupby("split")["date"].min().dt.strftime("%Y-%m-%d"),
    "last_date": same.groupby("split")["date"].max().dt.strftime("%Y-%m-%d"),
}).reindex(ORDER)
# inserted after reindexing so the label cannot be mis-aligned with the row
tbl.insert(0, "years", pd.Series([f"{lo}-{hi}" for lo, hi in SPLITS.values()], index=ORDER))
print(tbl.to_string())
kv("test rows lost to the one-day shift",
   f"{int(tbl.loc['test', 'rows_same_day'] - tbl.loc['test', 'rows_next_day'])} "
   f"(2024-12-31 has no t+1)")
_te_next = nextd[nextd["split"] == "test"]
kv("test window, next-day usable",
   f"{_te_next['date'].min():%Y-%m-%d} .. {_te_next['date'].max():%Y-%m-%d}")
kv("rate drift train -> test (next-day)",
   f"{tbl.loc['train', 'rate_next_day_pct']:.4f} % -> {tbl.loc['test', 'rate_next_day_pct']:.4f} %")
kv("rows per split sum to the usable panel",
   int(tbl["rows_next_day"].sum()) == len(usable))
kv("every station in every split", bool((tbl["stations"] == df["station_id"].nunique()).all()))

head("1D. Flood counts per year (drives Figure 1.3)")
per_year = df.groupby("year")[TARGET].agg(station_days="size", floods="sum")
per_year["flood_rate_pct"] = (per_year["floods"] / per_year["station_days"] * 100).round(4)
print(per_year.to_string())
kv("floods per year - min / max / mean",
   f"{int(per_year['floods'].min())} / {int(per_year['floods'].max())} / "
   f"{per_year['floods'].mean():.1f}")
kv("first-year vs last-year flood rate",
   f"{per_year['flood_rate_pct'].iloc[0]:.4f} % (2000) -> "
   f"{per_year['flood_rate_pct'].iloc[-1]:.4f} % (2024)")

# ======================================================================================
# Figures
# ======================================================================================
head("Figures")
figdir_before = {p.name for p in FIGDIR.glob("fig01_*.png")}

# ---- Figure 1.1 - balanced panel ------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
stations = rows_per_station.index.tolist()
ypos = np.arange(len(stations))
ax1.barh(ypos, rows_per_station.to_numpy(), height=0.8, color="#3b6ea5",
         label=f"rows per station (min = max = {int(rows_per_station.iloc[0]):,})")
ax1.axvline(int(rows_per_station.iloc[0]), color="crimson", ls="--", lw=1.5,
            label=f"expected 9,132 days x 50 stations")
ax1.set_yticks(ypos[::5])
ax1.set_yticklabels([s for s in stations[::5]], fontsize=8)
ax1.set_xlim(0, 11_000)
ax1.text(9_400, 25, f"all 50 stations:\n{int(rows_per_station.iloc[0]):,} rows\n"
                    f"= 50 x 9,132\n= {N_ROWS:,}",
         fontsize=9, va="center", bbox=dict(boxstyle="round", fc="#fdf3d0", ec="grey"))
finish(ax1, title="(a) Rows per station - the panel is perfectly balanced",
       xlabel="Rows per station (count)", ylabel="Station id (ASIA_nnnn)")

dmin, dmax = df["date"].min(), df["date"].max()
ax2.hlines(ypos, dmin, dmax, color="#3b6ea5", lw=3.0,
           label=f"observed coverage, 1 line per station ({len(stations)} lines)")
for lo, hi in SPLITS.values():
    ax2.axvline(pd.Timestamp(f"{hi}-12-31"), color="grey", ls=":", lw=1.0)
for label, yr in (("train\n2000-2016", 2008), ("validation\n2017-2020", 2018.5),
                  ("test\n2021-2024", 2022.5)):
    ax2.text(pd.Timestamp(f"{int(yr)}-01-01"), 52, label, fontsize=9, ha="center", va="bottom")
ax2.set_ylim(-2, 62)
ax2.text(dmin, 46, f"{dmin:%Y-%m-%d} -> {dmax:%Y-%m-%d}\n9,132 days x 50 stations, 0 gaps",
         fontsize=9, bbox=dict(boxstyle="round", fc="#e6f2e6", ec="grey"))
finish(ax2, title="(b) Date coverage - every station spans the full 25 years",
       xlabel="Date (calendar day)", ylabel="Station index (1-50)")
fig.tight_layout()
savefig(fig, "fig01_panel_completeness")

# ---- Figure 1.2 - class balance -------------------------------------------------------
counts = [int((1 - y).sum()), int(y.sum())]
shares = [counts[0] / len(y) * 100, counts[1] / len(y) * 100]
names = ["no flood (y = 0)", "flood (y = 1)"]
colours = ["#8fa8c8", "#c0392b"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

axes[0].bar(names, counts, color=colours, label="station-days")
for i, c in enumerate(counts):
    axes[0].text(i, c, f"{c:,}\n({shares[i]:.3f} %)", ha="center", va="bottom", fontsize=10)
axes[0].set_ylim(0, counts[0] * 1.22)
axes[0].yaxis.set_major_formatter("{x:,.0f}")
axes[0].text(0.02, 0.60,
             f"majority-class accuracy\n= {shares[0]:.2f} %\n(never used again)",
             transform=axes[0].transAxes, fontsize=10,
             bbox=dict(boxstyle="round", fc="#fdf3d0", ec="grey"))
finish(axes[0], title="(a) Raw counts - the majority class dominates",
       xlabel="Next-day flood label y(t) = flood(t+1)", ylabel="Station-days (count)")

axes[1].bar(names, counts, color=colours, label="station-days (log scale)")
axes[1].set_yscale("log")
for i, c in enumerate(counts):
    axes[1].text(i, c * 1.35, f"{c:,}", ha="center", va="bottom", fontsize=10)
axes[1].set_ylim(1e3, counts[0] * 6)
axes[1].text(0.02, 0.06, f"imbalance {counts[0] / counts[1]:.2f} : 1\n-> 45.2 : 1",
             transform=axes[1].transAxes, fontsize=10,
             bbox=dict(boxstyle="round", fc="#e6f2e6", ec="grey"))
finish(axes[1], title="(b) Same counts on a log axis - 45x fewer positives",
       xlabel="Next-day flood label y(t) = flood(t+1)", ylabel="Station-days (count, log scale)")

left = 0.0
for share, colour, name in zip(shares, colours, names):
    axes[2].barh([0], [share], left=left, color=colour, height=0.45, label=f"{name}: {share:.3f} %")
    left += share
axes[2].set_xlim(0, 100)
axes[2].set_ylim(-0.6, 0.6)
axes[2].set_yticks([])
axes[2].text(50, 0, f"negative class {shares[0]:.3f} %\n(accuracy floor)", ha="center",
             va="center", fontsize=10, color="white", fontweight="bold")
axes[2].annotate(f"{shares[1]:.3f} %\npositives\n({counts[1]:,} days)", xy=(98, -0.26),
                 xytext=(5, -0.48), fontsize=10, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="black", connectionstyle="arc3,rad=-0.2"),
                 bbox=dict(boxstyle="round", fc="#fdf3d0", ec="grey"))
finish(axes[2], title="(c) Class shares - 2.166 % positives, 97.834 % negatives",
       xlabel="Share of the 456,550 usable station-days (%)", ylabel="")
fig.tight_layout()
savefig(fig, "fig01_target_balance")

# ---- Figure 1.3 - floods per year with the split boundaries ---------------------------
fig, ax = plt.subplots(figsize=(13, 6))
years = per_year.index.to_numpy()
bars = ax.bar(years, per_year["floods"].to_numpy(), color="#3b6ea5", width=0.72,
              label="flood days per year (station-days, count)")
span_colours = {"train": "#dce9f5", "validation": "#fdf3d0", "test": "#f6dcd8"}
for name, (lo, hi) in SPLITS.items():
    ax.axvspan(lo - 0.5, hi + 0.5, color=span_colours[name], zorder=0,
               label=f"{name}: {lo}-{hi}")
ax2 = ax.twinx()
ax2.plot(years, per_year["flood_rate_pct"].to_numpy(), color="#7a1f1f", marker="o", ms=4,
         lw=1.6, label="flood rate (% of station-days)")
ax2.set_ylabel("Flood rate (% of station-days per year)")
ax2.set_ylim(0, per_year["flood_rate_pct"].max() * 1.45)
ax.set_ylim(0, per_year["floods"].max() * 1.50)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", ncols=2, fontsize=8.5, frameon=True,
          title=f"flood days per year: min {int(per_year['floods'].min())}, "
                f"max {int(per_year['floods'].max())}, mean {per_year['floods'].mean():.1f}")
for name, (lo, hi) in SPLITS.items():
    ax.text((lo + hi) / 2, per_year["floods"].max() * 1.18, f"{name}\n{lo}-{hi}",
            ha="center", va="top", fontsize=9, fontweight="bold")
finish(ax, title="Figure 1.3  Floods per year, 2000-2024, with the temporal split boundaries",
       xlabel="Year", ylabel="Flood days per year (station-days, count)", legend=False)
fig.tight_layout()
savefig(fig, "fig01_floods_by_year")

figdir_after = {p.name for p in FIGDIR.glob("fig01_*.png")}
print(f"\nfig01_* files written by this script: {sorted(figdir_after - figdir_before)}")
for p in sorted(FIGDIR.glob("fig01_*.png")):
    kv(f"  {p.name}", f"{p.stat().st_size / 1024:.1f} KB")

# ======================================================================================
# Self-check - fails loudly if the panel, the label or the split ever drifts.
# This is the report's contract: every number in s1-inspection.md is anchored here.
# ======================================================================================
head("Self-check")
assert (N_ROWS, N_COLS) == (456_600, 21), "panel shape changed"
assert int(miss.sum()) == 0, "missing values appeared"
assert int(df.duplicated().sum()) == 0, "duplicate rows appeared"
assert rows_per_station.nunique() == 1 and int(rows_per_station.iloc[0]) == 9_132, "panel unbalanced"
assert int((rule == df[TARGET]).sum()) == N_ROWS, "risk > 7.0 no longer reproduces the label"
assert int((sev_leftclosed == sev_dataset).sum()) == N_ROWS, "severity binning changed"
assert len(r7) == 32, "the 32 contradictory rows moved"
assert len(usable) == 456_550 and int(y.sum()) == 9_889, "target definition changed"
assert round(y.mean() * 100, 3) == 2.166, "positive rate changed"
assert int(tbl.loc["test", "rows_next_day"]) == 73_000, "test split size changed"
assert int(tbl.loc["test", "floods_next_day"]) == 1_703, "test flood count changed"
assert bool((tbl["stations"] == 50).all()), "a split lost stations"
print("all headline invariants hold (shape, quality, leakage rule, target, split)")
