"""T5 - stage 4A (strategies) and 4C (pipeline): evidence generator.

Produces every number quoted in ``sections/s4a-strategies.md`` and
``sections/s4c-pipeline.md``, plus the two stage-4 figures.  **No model is trained.**
Only preprocessing transformers are fitted, and only on the training window.

    python t5_pipeline.py        # cwd = docs/projects/eda/code
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pipeline as pl
from data import (
    SEED,
    TARGET,
    TARGET_NEXT,
    add_next_day_target,
    configure_env,
    finish,
    load_raw,
    savefig,
    split_frames,
    train_test_frames,
)

configure_env()
np.random.seed(SEED)

MEASUREMENTS = ["rainfall_mm", "river_level_m", "soil_moisture_percent", "temperature_celsius"]
UNITS = {
    "rainfall_mm": "mm/day",
    "river_level_m": "m",
    "soil_moisture_percent": "%",
    "temperature_celsius": "degC",
}
TRAIN_COLOUR, PIPE_COLOUR, FENCE_COLOUR = "#4C72B0", "#DD8452", "#C44E52"


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def stats_block(x: np.ndarray | pd.Series) -> dict[str, float]:
    """min / max / mean / std / skew / excess kurtosis, the shape statistics stage 2A uses."""
    s = pd.Series(np.asarray(x, dtype=float)).dropna()
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "skew": float(s.skew()),
        "kurt": float(s.kurt()),
    }


def table(rows: list[tuple], header: tuple) -> None:
    """Print a fixed-width table so the fragment can cite it verbatim."""
    widths = [max(len(str(header[i])), *(len(str(r[i])) for r in rows)) for i in range(len(header))]
    line = "  ".join(str(h).ljust(w) for h, w in zip(header, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def scaler_of(preprocess) -> tuple[np.ndarray, np.ndarray]:
    """The fitted ``StandardScaler`` mean/scale of the numeric branch - the leak evidence."""
    step = preprocess.named_transformers_["num"].named_steps["scale"]
    return np.asarray(step.mean_), np.asarray(step.scale_)


# ======================================================================================
print("=" * 88)
print("T5 - STAGE 4A STRATEGIES / 4C PIPELINE")
print("=" * 88)

raw = add_next_day_target(load_raw())
frame = split_frames(raw)                      # usable rows + split label
X_train, X_test, y_train, y_test = train_test_frames(raw)

print("\n[1] Data and split contract")
print(f"    raw panel            : {raw.shape[0]:,} rows x {raw.shape[1] - 1} cols (+ {TARGET_NEXT})")
print(f"    usable rows          : {len(frame):,}  ({int(frame[TARGET_NEXT].sum()):,} positives, "
      f"{frame[TARGET_NEXT].mean() * 100:.4f} %, {int((frame[TARGET_NEXT] == 0).sum()) / int(frame[TARGET_NEXT].sum()):.1f} : 1)")
for name in ("train", "validation", "test"):
    sub = frame[frame["split"] == name]
    print(f"    {name:<20}: {len(sub):>7,} rows, {int(sub[TARGET_NEXT].sum()):>5,} positives, "
          f"{sub[TARGET_NEXT].mean() * 100:.4f} %")
print(f"    X_train              : {X_train.shape}   (raw feature matrix, 6 numeric + 2 categorical)")
print(f"    X_test               : {X_test.shape}")
print("    (73,050 rows sit in the 2021-2024 window; 50 are dropped by the t+1 shift - "
      "2024-12-31 for each station - so the usable test frame is 73,000)")

# --------------------------------------------------------------------------------------
print("\n[2] 4A.2 OUTLIERS - the IQR fence, on all 456,600 raw rows")
rain = raw["rainfall_mm"]
q1, q3 = float(rain.quantile(0.25)), float(rain.quantile(0.75))
iqr = q3 - q1
fence_rain = q3 + 1.5 * iqr
river = raw["river_level_m"]
rq1, rq3 = float(river.quantile(0.25)), float(river.quantile(0.75))
fence_river = rq3 + 1.5 * (rq3 - rq1)

flag_rain = rain > fence_rain
flag_river = river > fence_river
print(f"    rainfall_mm   Q1 = {q1:.4f} mm, Q3 = {q3:.4f} mm, IQR = {iqr:.4f} mm "
      f"-> 1.5 x IQR fence = {fence_rain:.4f} mm")
print(f"    river_level_m Q1 = {rq1:.4f} m,  Q3 = {rq3:.4f} m,  IQR = {rq3 - rq1:.4f} m "
      f"-> 1.5 x IQR fence = {fence_river:.4f} m")
print(f"    declared constant EXTREME_RAINFALL_MM = {pl.EXTREME_RAINFALL_MM:.2f} mm "
      f"(stage 2A value; reproduces the full-panel fence to {abs(fence_rain - pl.EXTREME_RAINFALL_MM):.2e} mm)")
flood = raw[TARGET].astype(bool)
rows = []
for label, mask in (
    ("rainfall_mm > %.2f mm" % fence_rain, flag_rain),
    ("river_level_m > %.2f m" % fence_river, flag_river),
    ("either fence", flag_rain | flag_river),
):
    n_flag = int(mask.sum())
    rows.append(
        (
            label,
            f"{n_flag:,}",
            f"{n_flag / len(raw) * 100:.3f} %",
            f"{int((flood & mask).sum()):,}",
            f"{(flood & mask).sum() / n_flag * 100:.3f} %",
            f"{flood[~mask].mean() * 100:.3f} %",
        )
    )
table(rows, ("fence", "rows flagged", "share", "of which floods", "flood rate | flagged", "flood rate | unflagged"))
print(f"    rows removed by the outlier strategy : 0")
print(f"    rows winsorised / clipped            : 0")

# train-only fence, to show the declared constant is not tuned on test
tr_rain = frame.loc[frame["split"] == "train", "rainfall_mm"]
tq1, tq3 = float(tr_rain.quantile(0.25)), float(tr_rain.quantile(0.75))
fence_train = tq3 + 1.5 * (tq3 - tq1)
flag_train_only = raw["rainfall_mm"] > fence_train
print(f"    train-only fence (2000-2016)         : {fence_train:.4f} mm "
      f"-> flags {int(flag_train_only.sum()):,} rows; "
      f"disagrees with the declared constant on {int((flag_train_only != (rain > pl.EXTREME_RAINFALL_MM)).sum()):,} of {len(raw):,} rows")
train_extreme = (X_train[pl.LOG1P_COLS[0]] > pl.EXTREME_RAINFALL_MM).to_numpy()
train_flood_next = y_train.to_numpy().astype(bool)                                   # flood_next_day
train_flood_same = frame.loc[frame["split"] == "train", TARGET].to_numpy().astype(bool)
print(f"    train rows above the fence           : {int(train_extreme.sum()):,} "
      f"({train_extreme.mean() * 100:.4f} % of train)")
print(f"      same-day floods among them {train_flood_same[train_extreme].mean() * 100:.3f} % "
      f"(below the fence {train_flood_same[~train_extreme].mean() * 100:.3f} %)   <- stage 2A finding, same-day label")
print(f"      next-day floods among them {train_flood_next[train_extreme].mean() * 100:.3f} % "
      f"(below the fence {train_flood_next[~train_extreme].mean() * 100:.3f} %)   <- the deliverable's label, t+1")

# --------------------------------------------------------------------------------------
print("\n[3] 4C. THE PIPELINE - fit on train, transform train and test")
preprocess = pl.fit_preprocessor(X_train)            # <-- fit on TRAIN ONLY
X_train_t = pl.transform(preprocess, X_train)
X_test_t = pl.transform(preprocess, X_test)
names = pl.feature_names(preprocess)

print(f"    columns before         : {X_train.shape[1]}  ({len(pl.NUMERIC_RAW)} numeric + {len(pl.CATEGORICAL_RAW)} categorical)")
print(f"    columns after          : {X_train_t.shape[1]}")
print(f"    X_train_t.shape        : {X_train_t.shape}   dtype {X_train_t.dtype}")
print(f"    X_test_t.shape         : {X_test_t.shape}   dtype {X_test_t.dtype}")
print(f"    int(np.isnan(X_train_t).sum()) : {int(np.isnan(X_train_t).sum())}")
print(f"    int(np.isnan(X_test_t).sum())  : {int(np.isnan(X_test_t).sum())}")
print(f"    finite values everywhere      : {bool(np.isfinite(X_train_t).all() and np.isfinite(X_test_t).all())}")
blocks = {
    "num__": sum(n.startswith("num__") for n in names),
    "cat__basin_": sum(n.startswith("cat__basin_") for n in names),
    "cat__season_": sum(n.startswith("cat__season_") for n in names),
}
print(f"    block widths           : numeric {blocks['num__']} (6 raw + 1 flag), "
      f"basin one-hot {blocks['cat__basin_']}, season one-hot {blocks['cat__season_']}")
print("    feature names (get_feature_names_out):")
for i, n in enumerate(names, start=1):
    print(f"      {i:>2}. {n}")

# --------------------------------------------------------------------------------------
print("\n[4] 4A.1 MISSING VALUES - nothing to impute on this panel")
n_cells = raw.shape[0] * (raw.shape[1] - 1)
n_missing = int(raw.drop(columns=[TARGET_NEXT]).isna().sum().sum())
print(f"    missing cells, all {raw.shape[1] - 1} raw columns x {raw.shape[0]:,} rows : {n_missing} "
      f"({n_missing / n_cells * 100:.2f} %) of {n_cells:,}")
print(f"    missing cells in X_train / X_test                  : {int(X_train.isna().sum().sum())} / {int(X_test.isna().sum().sum())}")
num_imp = preprocess.named_transformers_["num"].named_steps["impute"]
cat_imp = preprocess.named_transformers_["cat"].named_steps["impute"]
print("    fitted numeric imputer (SimpleImputer(median), train only), in raw units:")
for col, val in zip(pl.NUMERIC_BRANCH_IN, num_imp.statistics_):
    shown = f"{np.expm1(val):.4f}" if col in pl.LOG1P_COLS else f"{val:.4f}"
    print(f"      {col:<26} median(raw) = {shown:<12} (internal value {val:.6f})")
print(f"    fitted categorical imputer (most_frequent, train only) : {list(cat_imp.statistics_)}")
print(f"      note: every basin holds exactly 10.0000 % of the panel, so most_frequent is a 10-way")
print(f"      tie and falls back to the first level observed in train ({cat_imp.statistics_[0]!r}); "
      f"season has a real mode ({cat_imp.statistics_[1]!r}).")
print("    cells actually imputed : 0 - the imputers are a robustness measure for a partial")
print("    future day, NOT a fix for observed missingness (there is none to fix)")

# --------------------------------------------------------------------------------------
print("\n[5] 4A.4 SCALING - log1p first, then StandardScaler")
print("    right-censored measurements (a hard ceiling: nothing in the column exceeds it):")
for col, ceiling in (("soil_moisture_percent", 100.0), ("temperature_celsius", 45.0)):
    n_full = int((raw[col] == ceiling).sum())
    n_tr = int((X_train[col] == ceiling).sum())
    print(f"      {col:<24} exactly {ceiling:>6.2f} : {n_full:,} rows full panel ({n_full / len(raw) * 100:.3f} %), "
          f"{n_tr:,} in train ({n_tr / len(X_train) * 100:.3f} %); column max {raw[col].max():.2f}")
for col in ("soil_moisture_percent", "temperature_celsius"):
    counts, edges = np.histogram(X_train[col], bins=80)
    print(f"      {col:<24} 80-bin figure: last bin {counts[-1]:,} days vs {counts[-2]:,} in the bin below "
          f"(width {edges[1] - edges[0]:.3f})")
for col in MEASUREMENTS:
    full_stats = stats_block(raw[col])
    raw_stats = stats_block(X_train[col])
    log_stats = stats_block(np.log1p(X_train[col])) if col in pl.LOG1P_COLS else raw_stats
    t_stats = stats_block(X_train_t[:, names.index(f"num__{col}")])
    print(f"    {col}")
    print(f"      full panel  min {full_stats['min']:>11.4f}  max {full_stats['max']:>11.4f}  "
          f"mean {full_stats['mean']:>9.4f}  std {full_stats['std']:>9.4f}  "
          f"skew {full_stats['skew']:>8.4f}  kurt {full_stats['kurt']:>10.4f}   <- stage 2A, 456,600 rows")
    print(f"      train       min {raw_stats['min']:>11.4f}  max {raw_stats['max']:>11.4f}  "
          f"mean {raw_stats['mean']:>9.4f}  std {raw_stats['std']:>9.4f}  "
          f"skew {raw_stats['skew']:>8.4f}  kurt {raw_stats['kurt']:>10.4f}   <- 310,500 rows")
    if col in pl.LOG1P_COLS:
        print(f"      log1p       min {log_stats['min']:>11.4f}  max {log_stats['max']:>11.4f}  "
              f"mean {log_stats['mean']:>9.4f}  std {log_stats['std']:>9.4f}  "
              f"skew {log_stats['skew']:>8.4f}  kurt {log_stats['kurt']:>10.4f}   <- compression")
    print(f"      pipeline    min {t_stats['min']:>11.4f}  max {t_stats['max']:>11.4f}  "
          f"mean {t_stats['mean']:>9.4f}  std {t_stats['std']:>9.4f}  "
          f"skew {t_stats['skew']:>8.4f}  kurt {t_stats['kurt']:>10.4f}   <- after StandardScaler")

mon = raw[raw["monsoon_season"] == 1]["rainfall_mm"]
off = raw[raw["monsoon_season"] == 0]["rainfall_mm"]
print(f"    rainfall inside  monsoon (n={len(mon):,}) : mean {mon.mean():.4f} mm, std {mon.std():.4f} mm")
print(f"    rainfall outside monsoon (n={len(off):,}) : mean {off.mean():.4f} mm, std {off.std():.4f} mm")
print(f"    location moves {mon.mean() / off.mean():.2f}x, spread moves {mon.std() / off.std():.2f}x")
trm = frame[(frame["split"] == "train") & (frame["monsoon_season"] == 1)]["rainfall_mm"]
tro = frame[(frame["split"] == "train") & (frame["monsoon_season"] == 0)]["rainfall_mm"]
print(f"    same on train only                        : mean {trm.mean():.4f} / {tro.mean():.4f} mm, "
      f"std {trm.std():.4f} / {tro.std():.4f} mm "
      f"(location {trm.mean() / tro.mean():.2f}x, spread {trm.std() / tro.std():.2f}x)")
flag_t = X_train_t[:, names.index(f"num__{pl.EXTREME_FLAG_COL}")]
uniq_raw = int(X_train["soil_moisture_percent"].nunique())
uniq_t = int(np.unique(X_train_t[:, names.index("num__soil_moisture_percent")]).size)
rho = float(pd.Series(X_train["rainfall_mm"].to_numpy()).corr(
    pd.Series(X_train_t[:, names.index("num__rainfall_mm")]), method="spearman"))
print(f"    is_extreme_rainfall after scaling : {np.unique(flag_t).size} distinct values "
      f"{np.round(np.unique(flag_t), 4).tolist()}, rate {float((X_train[pl.LOG1P_COLS[0]] > pl.EXTREME_RAINFALL_MM).mean()) * 100:.4f} % of train")
print(f"    Spearman(raw rainfall, num__rainfall_mm) = {rho:.6f}  -> strictly monotone, rank order kept")
print(f"    soil moisture ties kept: {uniq_raw:,} unique raw values -> {uniq_t:,} unique transformed values")

# --------------------------------------------------------------------------------------
print("\n[6] 4A.3 CATEGORICAL ENCODING - no rare level exists to collapse")
shares = {}
for col in ("station_id", "basin", "season"):
    s = frame[col].value_counts(normalize=True) * 100
    shares[col] = (int(frame[col].nunique()), float(s.min()), float(s.max()))
    print(f"    {col:<12} {shares[col][0]:>3} levels, level share {shares[col][1]:.4f} % - {shares[col][2]:.4f} %")
enc = preprocess.named_transformers_["cat"].named_steps["onehot"]
print(f"    OneHotEncoder(handle_unknown='ignore', sparse_output=False) fitted on train out-width : {len(names) - blocks['num__']}")
for col, cats in zip(pl.CATEGORICAL_RAW, enc.categories_):
    print(f"      {col:<8} categories_: {list(cats)}")

# --------------------------------------------------------------------------------------
print("\n[7] DISCIPLINE CHECKS (each is also run standalone as a throwaway script)")
print("    (a) train-only vs full-panel fit - why fitting outside train would leak")
pp_train = preprocess
pp_full = pl.fit_preprocessor(frame[pl.RAW_FEATURES])       # deliberately WRONG, diagnostic only
m_tr, s_tr = scaler_of(pp_train)
m_full, s_full = scaler_of(pp_full)
rows = []
for i, col in enumerate(pl.NUMERIC_BRANCH_IN):
    rows.append(
        (
            col,
            f"{m_tr[i]:.6f}",
            f"{m_full[i]:.6f}",
            f"{abs(m_tr[i] - m_full[i]):.6f}",
            f"{s_tr[i]:.6f}",
            f"{s_full[i]:.6f}",
            f"{abs(s_tr[i] - s_full[i]):.6f}",
            f"{abs(m_tr[i] - m_full[i]) / s_tr[i]:.6f}",
        )
    )
table(rows, ("numeric column", "mean|train", "mean|full", "|d mean|", "scale|train", "scale|full", "|d scale|", "|d mean| / scale"))
worst = int(np.argmax(np.abs(m_tr - m_full)))
print(f"    largest |delta mean|  : {np.abs(m_tr - m_full).max():.6f} on {pl.NUMERIC_BRANCH_IN[worst]} "
      f"(= {np.abs(m_tr - m_full).max() / s_tr[worst]:.6f} train sigma)")
print(f"    largest |delta scale| : {np.abs(s_tr - s_full).max():.6f} on "
      f"{pl.NUMERIC_BRANCH_IN[int(np.argmax(np.abs(s_tr - s_full)))]}")
X_train_t_full = pl.transform(pp_full, X_train)
delta = np.abs(X_train_t - X_train_t_full)
print(f"    effect on the training rows themselves: max |X_train_t(train-fitted) - X_train_t(full-fitted)| "
      f"= {delta.max():.6f} sigma, mean {delta.mean():.6f} sigma, on {delta.shape[0]:,} x {delta.shape[1]} cells")
pp_test = pl.fit_preprocessor(frame.loc[frame["split"] == "test", pl.RAW_FEATURES])   # also diagnostic only
m_te, s_te = scaler_of(pp_test)
worst_te = int(np.argmax(np.abs(m_tr - m_te) / s_tr))
print(f"    for contrast, a scaler fitted on the TEST window alone would move the training rows by up to "
      f"{np.abs(m_tr - m_te).max() / s_tr[worst_te]:.6f} sigma "
      f"({pl.NUMERIC_BRANCH_IN[worst_te]}, |d mean| {np.abs(m_tr - m_te).max():.6f})")
print(f"    reading: on this panel the leak is small - the panel is stationary (year correlates with nothing, "
      f"|r| <= 0.037) - but it is not zero, and this is how we know.")

print("    (b) unseen category at transform time (the 'new category in the test set' case)")
probe = X_test.head(4).copy()
probe["basin"] = probe["basin"].astype(object)
probe["season"] = probe["season"].astype(object)
probe.loc[probe.index[0], "basin"] = "New Basin 2099"     # unseen basin AND unseen season
probe.loc[probe.index[0], "season"] = "Monsoon-X"
probe.loc[probe.index[1], "season"] = "Monsoon-X"         # unseen season only, known basin
probe.loc[probe.index[2], "basin"] = "New Basin 2099"     # unseen basin only, known season
out = pl.transform(pp_train, probe)
cat_idx = [i for i, n in enumerate(names) if n.startswith("cat__")]
basin_idx = [i for i, n in enumerate(names) if n.startswith("cat__basin_")]
season_idx = [i for i, n in enumerate(names) if n.startswith("cat__season_")]
print("      transform(...) raised      : no")
cases = (
    ("both levels unseen   ", 0),
    ("season unseen only   ", 1),
    ("basin unseen only    ", 2),
    ("both levels known    ", 3),
)
for label, i in cases:
    print(f"      {label} basin block sum {np.abs(out[i, basin_idx]).sum():.1f}, "
          f"season block sum {np.abs(out[i, season_idx]).sum():.1f}, "
          f"NaN {int(np.isnan(out[i]).sum())}")
print(f"      all {len(cat_idx)} categorical columns zero for the fully-unseen row : {bool(np.allclose(out[0, cat_idx], 0.0))}")
print("      (an unseen level maps to all-zeros instead of raising, because handle_unknown='ignore';")
print("       the known level in the same row still encodes - only the unknown block goes to zero)")

print("    (c) partial future day - missing numeric and missing categorical at transform time")
probe2 = X_test.head(2).copy()
probe2["basin"] = probe2["basin"].astype(object)
probe2.loc[probe2.index[0], "rainfall_mm"] = np.nan
probe2.loc[probe2.index[0], "basin"] = None
out2 = pl.transform(pp_train, probe2)
print("      transform(...) raised      : no")
print(f"      NP.isnan(out2).sum()       : {int(np.isnan(out2).sum())}; imputed num__rainfall_mm = "
      f"{out2[0, names.index('num__rainfall_mm')]:.6f} sigma (median -> not the mean), "
      f"{pl.EXTREME_FLAG_COL} = {out2[0, names.index('num__' + pl.EXTREME_FLAG_COL)]:.4f}")
print(f"      missing basin filled with  : {cat_imp.statistics_[0]!r}; its one-hot block sum = {np.abs(out2[0, basin_idx]).sum():.1f}")
print("      the same three representations of a missing category, encoder alone, no imputer:")
for label, frame in (
    ("unseen string level", pd.DataFrame({"basin": ["Atlantis"], "season": ["Spring"]})),
    ("None / object dtype", pd.DataFrame({"basin": [None], "season": ["Spring"]})),
    ("all-NaN float64    ", pd.DataFrame({"basin": [np.nan], "season": ["Spring"]})),
):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = np.asarray(enc.transform(frame))[0]
        print(f"        {label} (basin dtype {str(frame['basin'].dtype):<7}) : accepted, basin block sum {row[:10].sum():.1f}")
    except Exception as exc:  # noqa: BLE001
        print(f"        {label} (basin dtype {str(frame['basin'].dtype):<7}) : {type(exc).__name__}: {exc}")
print("        -> imputing first removes both the crash and the silent all-zero encoding")

# --------------------------------------------------------------------------------------
print("\n[8] FIGURES")
raw_train = {c: X_train[c].to_numpy() for c in pl.NUMERIC_RAW}
t_idx = {c: names.index(f"num__{c}") for c in MEASUREMENTS}

fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
for j, col in enumerate(MEASUREMENTS):
    ax = axes[0, j]
    ax.hist(raw_train[col], bins=80, color=TRAIN_COLOUR, edgecolor="white", linewidth=0.2)
    ax.set_yscale("log")
    finish(
        ax,
        title=f"raw {col}",
        xlabel=f"{col} ({UNITS[col]})",
        ylabel="station-days, train (count, log scale)",
        legend=False,
    )
    if col == pl.LOG1P_COLS[0]:
        ax.axvline(pl.EXTREME_RAINFALL_MM, color=FENCE_COLOUR, ls="--", lw=2)
        n_flag_panel = int(flag_rain.sum())
        rate_panel = (flood & flag_rain).sum() / n_flag_panel * 100
        ax.annotate(
            f"1.5xIQR fence {pl.EXTREME_RAINFALL_MM:.2f} mm\n{n_flag_panel:,} panel days above it\n"
            f"{rate_panel:.3f} % of them are same-day floods",
            xy=(pl.EXTREME_RAINFALL_MM, 1.4e3),
            xytext=(430, 6e3),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=FENCE_COLOUR),
            bbox=dict(boxstyle="round", fc="white", ec=FENCE_COLOUR, alpha=0.9),
        )

    ax2 = axes[1, j]
    ax2.hist(X_train_t[:, t_idx[col]], bins=80, color=PIPE_COLOUR, edgecolor="white", linewidth=0.2)
    ax2.axvline(0.0, color="black", ls=":", lw=1.5)
    finish(
        ax2,
        title=f"after pipeline: {col}",
        xlabel="pipeline output: log1p + StandardScaler (sigma)",
        ylabel="station-days, train (count)",
        legend=False,
    )

fig.legend(
    handles=[
        Patch(facecolor=TRAIN_COLOUR, label="raw measurement (before the pipeline)"),
        Patch(facecolor=PIPE_COLOUR, label="pipeline output, numeric branch (after)"),
        Line2D([0], [0], color=FENCE_COLOUR, ls="--", lw=2, label=f"1.5xIQR fence = {pl.EXTREME_RAINFALL_MM:.2f} mm"),
        Line2D([0], [0], color="black", ls=":", lw=1.5, label="train mean (0 sigma)"),
    ],
    loc="lower center",
    ncol=4,
    frameon=True,
    bbox_to_anchor=(0.5, -0.035),
)
fig.suptitle(
    "Figure 4A.1 - the four measurements before and after the numeric branch, train 2000-2016 "
    f"({len(X_train):,} station-days)",
    fontsize=13,
)
savefig(fig, "fig40_before_after_scaling")

fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
box_raw = axes[0].boxplot(
    [raw_train[c] for c in pl.NUMERIC_RAW],
    tick_labels=pl.NUMERIC_RAW,
    showmeans=True,
    showfliers=True,
    flierprops=dict(marker=".", markersize=2, alpha=0.25, markerfacecolor=FENCE_COLOUR, markeredgecolor="none"),
    medianprops=dict(color="darkorange", lw=2),
    meanprops=dict(marker="^", markerfacecolor="green", markeredgecolor="green", markersize=7),
)
finish(
    axes[0],
    title="A. Raw numeric inputs - six features, six incompatible scales",
    xlabel="raw feature in the input frame",
    ylabel="raw value (mm/day, m, %, degC, day-of-year, 0/1)",
    legend=False,
)
axes[0].tick_params(axis="x", rotation=20)

short = [n.split("__", 1)[1] for n in names]
axes[1].boxplot(
    [X_train_t[:, i] for i in range(X_train_t.shape[1])],
    tick_labels=short,
    showmeans=True,
    showfliers=False,
    medianprops=dict(color="darkorange", lw=2),
    meanprops=dict(marker="^", markerfacecolor="green", markeredgecolor="green", markersize=5),
)
axes[1].axhline(0.0, color="black", ls=":", lw=1.5)
finish(
    axes[1],
    title=f"B. Pipeline output - all {X_train_t.shape[1]} features on one scale",
    xlabel="transformed feature (num__ = scaled numeric branch, cat__ = one-hot categorical branch)",
    ylabel="pipeline output (standardised units, sigma)",
    legend=False,
)
axes[1].tick_params(axis="x", rotation=90, labelsize=8)

fig.legend(
    handles=[
        Line2D([0], [0], color="darkorange", lw=2, label="median"),
        Line2D([0], [0], marker="^", color="green", lw=0, markersize=8, label="mean"),
        Line2D([0], [0], marker=".", color=FENCE_COLOUR, lw=0, markersize=8, label="beyond 1.5xIQR (panel A only; hidden in B for legibility)"),
        Line2D([0], [0], color="black", ls=":", lw=1.5, label="train mean = 0 sigma"),
    ],
    loc="lower center",
    ncol=4,
    frameon=True,
    bbox_to_anchor=(0.5, -0.03),
)
fig.suptitle(
    f"Figure 4C.1 - the pipeline output is one comparable scale: raw inputs span "
    f"{float(X_train[pl.NUMERIC_RAW].min().min()):.2f} to {float(X_train[pl.NUMERIC_RAW].max().max()):.2f}, "
    f"transformed features {float(X_train_t.min()):.2f} to {float(X_train_t.max()):.2f} sigma",
    fontsize=13,
)
savefig(fig, "fig40_transformed_feature_scales")

print(f"\n    transformed range, all {X_train_t.shape[1]} features : "
      f"{float(X_train_t.min()):.4f} to {float(X_train_t.max()):.4f} sigma "
      f"(raw numeric range {float(X_train[pl.NUMERIC_RAW].min().min()):.4f} to {float(X_train[pl.NUMERIC_RAW].max().max()):.4f})")
print("\nDONE - no model was trained.")
