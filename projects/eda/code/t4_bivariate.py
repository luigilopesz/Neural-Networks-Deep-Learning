"""T4 - Stage 3 of the EDA deliverable: bivariate and multivariate analysis.

Items covered (rubric 2.0 / 10):

* **3A - numerical x numerical** (0.75): Spearman as the justified primary method with
  Pearson reported side by side, the redundant pair named, and the point-biserial
  correlation of every numeric column with the one-day-ahead target.
* **3B - categorical x target** (0.75): flood rate per level of ``basin``, ``country``,
  ``season``, ``monsoon_season``, ``station_id`` and ``month`` against the next-day label.
* **3C - numerical x categorical** (0.50): grouped boxplots interpreted in location *and*
  spread, by ``monsoon_season`` and by ``basin``.

Nine figures exactly (three per item - the statement caps stages 2 and 3 at 3 per item):

    fig30_corr_heatmap.png       3A.1  Pearson (upper) vs Spearman (lower)
    fig31_scatter_rain_river.png 3A.2  rainfall x river level, raw, by next-day label
    fig32_scatter_log1p.png      3A.3  the same on log1p axes
    fig33_flood_rate_month.png   3B.1  flood rate by month, monsoon window shaded
    fig34_flood_rate_basin.png   3B.2  flood rate by basin
    fig35_flood_rate_station.png 3B.3  flood rate by station
    fig36_rainfall_monsoon.png   3C.1  rainfall by monsoon flag, raw and log1p
    fig37_measurements_basin.png 3C.2  the four measurements by basin
    fig38_river_level_basin.png  3C.3  river level by basin

**No model is trained anywhere in this script.** Everything below is descriptive
statistics. ``roc_auc_score`` is applied to a single raw column, which is a rank
statistic, not a fitted estimator.

Run from this directory (``docs/projects/eda/code``) so that ``from data import ...``
resolves:

    python t4_bivariate.py

All randomness is seeded with ``SEED`` (42); the output is deterministic.
"""

from __future__ import annotations

import os

# Pin BLAS/OMP/joblib to one thread *before* numpy/sklearn are imported. The DSH file
# sandbox forbids named pipes, so any joblib Parallel pool dies with WinError 5.
for _var in ("OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from data import (  # noqa: E402
    SEED,
    TARGET,
    TARGET_NEXT,
    add_next_day_target,
    class_balance,
    configure_env,
    finish,
    load_raw,
    savefig,
)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
MEAS = ["rainfall_mm", "river_level_m", "soil_moisture_percent", "temperature_celsius"]

UNITS = {
    "rainfall_mm": "mm/day",
    "river_level_m": "m",
    "soil_moisture_percent": "%",
    "temperature_celsius": "\u00b0C",
}
LABELS = {
    "rainfall_mm": "rainfall",
    "river_level_m": "river level",
    "soil_moisture_percent": "soil moisture",
    "temperature_celsius": "temperature",
}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#: columns shown in the 3A.1 correlation heatmap (4 measurements + calendar +
#: engineered + the next-day target itself)
CORR_COLS = [
    "rainfall_mm",
    "river_level_m",
    "soil_moisture_percent",
    "temperature_celsius",
    "day_of_year",
    "month",
    "monsoon_season",
    "flood_risk_score",
    "latitude",
    "longitude",
    "data_quality_score",
    TARGET_NEXT,
]

#: the six pairs among the four measurements, reported Pearson beside Spearman
MEAS_PAIRS = [(a, b) for i, a in enumerate(MEAS) for b in MEAS[i + 1:]]

#: second-order redundancies and seasonality confounders named in the report
EXTRA_PAIRS = [
    ("flood_risk_score", "soil_moisture_percent"),
    ("flood_risk_score", "river_level_m"),
    ("flood_risk_score", "rainfall_mm"),
    ("month", "day_of_year"),
    ("latitude", "longitude"),
    ("temperature_celsius", "day_of_year"),
    ("soil_moisture_percent", "day_of_year"),
    ("rainfall_mm", "day_of_year"),
    ("river_level_m", "day_of_year"),
]

#: columns whose point-biserial correlation with the next-day target is reported
BISERIAL = [
    "rainfall_mm", "river_level_m", "soil_moisture_percent", "temperature_celsius",
    "day_of_year", "month", "day", "year", "monsoon_season", "flood_risk_score",
    "latitude", "longitude", "data_quality_score", TARGET,
]

#: single-column ROC-AUC against the next-day label - a rank statistic, not a fit
AUC_COLS = [
    "soil_moisture_percent", "rainfall_mm", "river_level_m", "temperature_celsius",
    "flood_risk_score", "monsoon_season", "day_of_year",
]

SCATTER_NEGATIVES = 30_000  # positives are never subsampled


# --------------------------------------------------------------------------------------
# Small printing helpers
# --------------------------------------------------------------------------------------
def rule(title: str, char: str = "=") -> None:
    print(f"\n{char * 78}\n{title}\n{char * 78}")


def rate_table(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """n / floods / flood-rate(%) / share-of-floods(%) per level of ``key``."""
    t = frame.groupby(key, observed=True)[TARGET_NEXT].agg(n="size", floods="sum")
    t["floods"] = t["floods"].astype(int)
    t["rate_pct"] = 100 * t["floods"] / t["n"]
    t["share_pct"] = 100 * t["floods"] / t["floods"].sum()
    return t.sort_values("rate_pct", ascending=False)


def box(ax, text: str, loc: str = "upper left") -> None:
    """White annotation box in axes coordinates."""
    xy = {"upper left": (0.02, 0.98), "upper right": (0.98, 0.98),
          "lower left": (0.02, 0.03), "lower right": (0.98, 0.03)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9, pad=3.5))


# --------------------------------------------------------------------------------------
# 3A - numerical x numerical
# --------------------------------------------------------------------------------------
def section_3a(d: pd.DataFrame, y: pd.Series) -> None:
    rule("3A - NUMERICAL x NUMERICAL")

    # ---- why Spearman: shape of the four measurements ------------------------------
    print("\n[3A.0] shape of the four measurements - skew and kurtosis (n = %d)" % len(d))
    shape = d[MEAS].agg(["skew", "kurt"]).T
    shape.columns = ["skew", "kurtosis"]
    print(shape.round(4).to_string())

    # ---- Pearson vs Spearman, side by side ----------------------------------------
    pear = d[MEAS].corr(method="pearson")
    spea = d[MEAS].corr(method="spearman")
    print("\n[3A.1] Pearson vs Spearman for the six pairs among the four measurements"
          f" (n = {len(d):,})")
    print(f"  {'pair':48s} {'Pearson':>9s} {'Spearman':>9s} {'P - S':>8s}")
    for a, b in MEAS_PAIRS:
        print(f"  {a + ' <-> ' + b:48s} {pear.loc[a, b]:9.4f} {spea.loc[a, b]:9.4f} "
              f"{pear.loc[a, b] - spea.loc[a, b]:+8.4f}")

    print("\n  Pearson matrix (upper triangle of figure 3A.1)")
    print(d[MEAS].corr(method="pearson").round(4).to_string())
    print("\n  Spearman matrix (lower triangle of figure 3A.1)")
    print(spea.round(4).to_string())

    # log1p view of the dominant pair - the evidence that the linearity is tail-driven
    log_r = np.log1p(d["rainfall_mm"])
    log_l = np.log1p(d["river_level_m"])
    r_log = float(np.corrcoef(log_r, log_l)[0, 1])
    print(f"\n  rainfall vs river level, Pearson on the raw values : {pear.loc['rainfall_mm', 'river_level_m']:.4f}")
    print(f"  rainfall vs river level, Pearson on log1p values    : {r_log:.4f}")
    print(f"  rainfall vs river level, Spearman (rank, transform-invariant): "
          f"{spea.loc['rainfall_mm', 'river_level_m']:.4f}")

    # ---- the trimming test: is the linear correlation carried by the tail? ----------
    mon = d[d["monsoon_season"] == 1]
    dry = d[d["monsoon_season"] == 0]
    print("\n[3A.1b] trimming test - Pearson with the top few percent of rainfall removed")
    print(f"  {'subset':36s} {'n':>9s} {'% panel':>8s} {'Pearson':>8s} {'P(log1p)':>9s} {'Spearman':>9s}")
    rows = []
    for label, frame, cuts in (("all rows", d, (None, 150.0, 100.0)),
                               ("monsoon only", mon, (None, 150.0, 100.0)),
                               ("outside monsoon", dry, (None,))):
        for cut in cuts:
            sub = frame if cut is None else frame[frame["rainfall_mm"] <= cut]
            p = float(sub[["rainfall_mm", "river_level_m"]].corr().iloc[0, 1])
            pl = float(np.corrcoef(np.log1p(sub["rainfall_mm"]),
                                   np.log1p(sub["river_level_m"]))[0, 1])
            s = float(sub[["rainfall_mm", "river_level_m"]].corr(method="spearman").iloc[0, 1])
            rows.append((label, cut, len(sub), p, pl, s))
            name = label if cut is None else f"{label}, rainfall <= {cut:.0f} mm"
            print(f"  {name:36s} {len(sub):9,d} {100 * len(sub) / len(d):7.2f}% "
                  f"{p:8.4f} {pl:9.4f} {s:9.4f}")
    mon_all = next(r for r in rows if r[0] == "monsoon only" and r[1] is None)
    mon_100 = next(r for r in rows if r[0] == "monsoon only" and r[1] == 100.0)
    print(f"  => inside the monsoon, deleting the "
          f"{100 * (mon['rainfall_mm'] > 100).mean():.2f} % of days above 100 mm drops "
          f"Pearson from {mon_all[3]:.4f} to {mon_100[3]:.4f} - onto the Spearman value "
          f"{mon_all[5]:.4f}, while Spearman itself moves only to {mon_100[5]:.4f}")

    # ---- regime structure visible in figure 3A.3 ------------------------------------
    print("\n[3A.1c] two regimes in rainfall_mm (the split visible in figure 3A.3)")
    print(f"  monsoon_season = 0 : {dry['rainfall_mm'].min():.2f} - "
          f"{dry['rainfall_mm'].max():.2f} mm (99.9th pct "
          f"{dry['rainfall_mm'].quantile(0.999):.2f})")
    print(f"  monsoon_season = 1 : {mon['rainfall_mm'].min():.2f} - "
          f"{mon['rainfall_mm'].max():.2f} mm (1st pct "
          f"{mon['rainfall_mm'].quantile(0.01):.2f})")
    gap = d[(d["rainfall_mm"] > 50) & (d["rainfall_mm"] < 84)]
    print(f"  rows with 50 mm < rainfall < 84 mm : {len(gap):,} of {len(d):,} "
          f"({100 * len(gap) / len(d):.4f} %) - a hole, not a tail")
    print(f"  river level        : {d['river_level_m'].min():.2f} - "
          f"{d['river_level_m'].max():.2f} m; outside the monsoon the maximum is "
          f"{dry['river_level_m'].max():.2f} m, inside the minimum is "
          f"{mon['river_level_m'].min():.2f} m")

    print("\n[3A.1d] next-day flood rate by rainfall decile")
    for label, frame in (("all rows", d), ("monsoon only", mon)):
        t = (frame.assign(_dec=pd.qcut(frame["rainfall_mm"], 10, duplicates="drop"))
             .groupby("_dec", observed=False)
             .agg(n=("y", "size"), floods=("y", "sum")))
        t["floods"] = t["floods"].astype(int)
        t["rate_pct"] = 100 * t["floods"] / t["n"]
        t.index = [f"({iv.left:7.2f}, {iv.right:7.2f}] mm" for iv in t.index]
        print(f"  -- {label}")
        print("  " + t.to_string().replace("\n", "\n  "))

    # ---- the redundant pair --------------------------------------------------------
    print("\n[3A.2] the redundant pair (results-table row 7)")
    print(f"  rainfall_mm <-> river_level_m : Pearson "
          f"{pear.loc['rainfall_mm', 'river_level_m']:.4f}, Spearman "
          f"{spea.loc['rainfall_mm', 'river_level_m']:.4f}")

    print("\n[3A.3] second-order redundancies and seasonality confounders (Pearson / Spearman)")
    for a, b in EXTRA_PAIRS:
        p = float(d[[a, b]].corr(method="pearson").iloc[0, 1])
        s = float(d[[a, b]].corr(method="spearman").iloc[0, 1])
        print(f"  {a + ' <-> ' + b:48s} {p:+9.4f} {s:+9.4f}")

    print("\n  |r| of the noise / drift columns against every other numeric column")
    for col in ["data_quality_score", "year"]:
        row = d[[col] + [c for c in CORR_COLS if c != col]].corr(method="pearson").loc[col].drop(col)
        print(f"    {col:22s} max |r| = {row.abs().max():.4f} (vs {row.abs().idxmax()})")

    # ---- why both members of the redundant pair stay -------------------------------
    print("\n[3A.4] why BOTH rainfall_mm and river_level_m stay in the feature set")
    print(f"  {'single-column predictor':26s} {'ROC-AUC (next-day)':>19s}")
    for c in AUC_COLS:
        print(f"  {c:26s} {roc_auc_score(y, d[c]):19.4f}")

    # ---- point-biserial with the next-day target -----------------------------------
    print(f"\n[3A.5] point-biserial correlation with {TARGET_NEXT} (n = {len(d):,})")
    pb = pd.Series({c: float(np.corrcoef(d[c].astype(float), y)[0, 1]) for c in BISERIAL})
    pb = pb.reindex(pb.abs().sort_values(ascending=False).index)
    for c, v in pb.items():
        print(f"  {c:26s} {v:+.4f}")
    print(f"\n  ranking (|r|): " + " > ".join(pb.index[:7]))
    print("  inversion vs the same-day picture: soil_moisture "
          f"{pb['soil_moisture_percent']:.4f} and temperature {pb['temperature_celsius']:.4f} "
          f"beat rainfall {pb['rainfall_mm']:.4f} and river level {pb['river_level_m']:.4f}")

    # ---- target column of the heatmap ----------------------------------------------
    print(f"\n[3A.6] correlation of every heatmap column with {TARGET_NEXT}")
    tgt = pd.Series({c: float(d[[c, TARGET_NEXT]].corr(method="pearson").iloc[0, 1])
                     for c in CORR_COLS if c != TARGET_NEXT})
    print(tgt.round(4).sort_values(ascending=False).to_string())

    # =================================================================================
    # Figure 3A.1 - Pearson above the diagonal, Spearman below
    # =================================================================================
    n = len(CORR_COLS)
    mixed = pd.DataFrame(np.nan, index=CORR_COLS, columns=CORR_COLS)
    for i, a in enumerate(CORR_COLS):
        for j, b in enumerate(CORR_COLS):
            if i < j:
                mixed.loc[a, b] = float(d[[a, b]].corr(method="pearson").iloc[0, 1])
            elif i > j:
                mixed.loc[a, b] = float(d[[a, b]].corr(method="spearman").iloc[0, 1])
    annot = mixed.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    mask = np.eye(n, dtype=bool)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(mixed, mask=mask, annot=annot, fmt="", cmap="RdBu_r", center=0,
                vmin=-1.0, vmax=1.0, square=True, linewidths=0.6, linecolor="white",
                annot_kws={"size": 7}, cbar_kws={"label": "correlation coefficient (r / rho)",
                                                 "shrink": 0.65}, ax=ax)
    finish(ax,
           title=(f"Figure 3A.1 - Correlation matrix of the numeric columns: Pearson "
                  f"above the diagonal, Spearman below (n = {len(d):,} station-days)\n"
                  "diverging scale centred at 0; the diagonal is masked because it is "
                  "1.00 by construction"),
           xlabel="variable", ylabel="variable", legend=False)
    ax.grid(False)
    savefig(fig, "fig30_corr_heatmap")

    # =================================================================================
    # Figures 3A.2 / 3A.3 - rainfall x river level, raw and log1p
    # =================================================================================
    rng = np.random.default_rng(SEED)
    pos = d[d["y"] == 1]
    neg = d[d["y"] == 0].sample(n=SCATTER_NEGATIVES, random_state=SEED)
    samp = pd.concat([neg, pos], ignore_index=True)
    print(f"\n[3A.7] scatter subsample: all {len(pos):,} positives + "
          f"{len(neg):,} sampled negatives = {len(samp):,} points "
          f"(random_state={SEED}, alpha=0.25)")

    for name, xv, yv, xlab, ylab in [
        ("fig31_scatter_rain_river", samp["rainfall_mm"], samp["river_level_m"],
         "rainfall (mm/day)", "river level (m)"),
        ("fig32_scatter_log1p", np.log1p(samp["rainfall_mm"]), np.log1p(samp["river_level_m"]),
         "log1p rainfall (log1p of mm/day)", "log1p river level (log1p of m)"),
    ]:
        logged = name.endswith("log1p")
        fig, ax = plt.subplots(figsize=(9.5, 7))
        ax.scatter(xv[samp["y"] == 0], yv[samp["y"] == 0], s=5, alpha=0.22,
                   color="steelblue", linewidths=0,
                   label=f"no flood on day t+1 ({len(neg):,} of "
                         f"{int((d['y'] == 0).sum()):,} shown, random_state={SEED})")
        ax.scatter(xv[samp["y"] == 1], yv[samp["y"] == 1], s=9, alpha=0.55,
                   color="crimson", linewidths=0, label=f"flood on day t+1 (all {len(pos):,})")
        if not logged:
            ax.axvline(56.30, color="black", ls=":", lw=1.4,
                       label="rainfall IQR fence 56.30 mm (stage 2A)")
            ax.annotate(f"the other {100 - 100 * len(samp) / len(d):.1f} % of the panel is "
                        "heaped in this corner:\n"
                        f"99 % of all station-days have rainfall <= "
                        f"{d['rainfall_mm'].quantile(0.99):.0f} mm",
                        xy=(30, 6), xytext=(0.30, 0.42), textcoords="axes fraction",
                        fontsize=8.5, ha="left",
                        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2),
                        bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9, pad=3))
            note = (f"full panel (n = {len(d):,})\n"
                    f"Pearson  r = {pear.loc['rainfall_mm', 'river_level_m']:.4f}\n"
                    f"Spearman rho = {spea.loc['rainfall_mm', 'river_level_m']:.4f}\n"
                    "one straight line - because both variables\n"
                    "explode together on the same few thousand days")
        else:
            note = ("monsoon band only (n = 191,250 rows)\n"
                    f"Pearson {mon_all[3]:.4f} -> {mon_100[3]:.4f} once the "
                    f"{100 * (mon['rainfall_mm'] > 100).mean():.1f} % of days\n"
                    "above 100 mm are removed; Spearman stays\n"
                    f"{mon_all[5]:.4f} -> {mon_100[5]:.4f}")
        box(ax, note, "upper left" if not logged else "lower right")
        if logged:
            ax.annotate(f"no rows here: only {len(gap):,} of {len(d):,} station-days\n"
                        "have 50 mm < rainfall < 84 mm",
                        xy=(4.15, 3.6), xytext=(0.30, 0.72), textcoords="axes fraction",
                        fontsize=8.5, ha="left",
                        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2),
                        bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9, pad=3))
        finish(ax,
               title=("Figure 3A.2 - Rainfall vs river level, raw scale: one apparent straight "
                      "line, and the whole\nnegative class collapsed into the bottom-left corner"
                      if not logged else
                      "Figure 3A.3 - The same pair on log1p axes: two disjoint regimes, and the "
                      "linearity lives\nonly in the top few percent of days"),
               xlabel=xlab, ylabel=ylab)
        savefig(fig, name)


# --------------------------------------------------------------------------------------
# 3B - categorical x target
# --------------------------------------------------------------------------------------
def section_3b(d: pd.DataFrame) -> None:
    rule("3B - CATEGORICAL x TARGET (next-day)")

    overall = 100 * d[TARGET_NEXT].mean()
    print(f"\n[3B.0] overall next-day flood rate = {overall:.4f} % "
          f"({int(d[TARGET_NEXT].sum()):,} of {len(d):,} rows)")

    tables = {}
    for key in ["month", "monsoon_season", "season", "basin", "country", "station_id"]:
        t = rate_table(d, key)
        tables[key] = t
        print(f"\n[3B] flood rate by {key}")
        print(t.round(4).to_string())

    # ---- month ---------------------------------------------------------------------
    m = tables["month"].sort_index()
    print("\n[3B.1] month detail (next-day label)")
    print(m.round(4).to_string())
    zero_months = [MONTH_NAMES[i - 1] for i in m.index if m.loc[i, "floods"] == 0]
    print(f"  months with exactly 0 floods : {', '.join(zero_months)}")
    print(f"  monsoon window (Jun-Oct)     : {100 * m.loc[[6, 7, 8, 9, 10], 'floods'].sum() / m['floods'].sum():.2f} % of all floods")
    print(f"  Jul+Aug+Sep share of floods  : {100 * m.loc[[7, 8, 9], 'floods'].sum() / m['floods'].sum():.2f} %")
    print(f"  mean rainfall by month (mm)  : "
          + ", ".join(f"{MONTH_NAMES[i - 1]} {d[d.month == i].rainfall_mm.mean():.2f}" for i in range(1, 13)))

    # ---- monsoon gate --------------------------------------------------------------
    mo = tables["monsoon_season"]
    print("\n[3B.2] the monsoon gate")
    print(f"  monsoon_season = 1 : {mo.loc[1, 'floods']:,} floods / {mo.loc[1, 'n']:,} rows "
          f"= {mo.loc[1, 'rate_pct']:.4f} %")
    print(f"  monsoon_season = 0 : {mo.loc[0, 'floods']:.0f} flood  / {mo.loc[0, 'n']:,} rows "
          f"= {mo.loc[0, 'rate_pct']:.5f} %")
    print(f"  ratio of the two rates = {mo.loc[1, 'rate_pct'] / mo.loc[0, 'rate_pct']:,.0f}x "
          f"(the single flood outside Jun-Oct is row {int(d[(d.monsoon_season == 0) & (d[TARGET_NEXT] == 1)].index[0])})")

    # ---- basin ---------------------------------------------------------------------
    b = tables["basin"]
    print("\n[3B.3] basin spread")
    print(f"  highest {b.index[0]} {b.rate_pct.iloc[0]:.4f} % | lowest {b.index[-1]} "
          f"{b.rate_pct.iloc[-1]:.4f} % | ratio {b.rate_pct.iloc[0] / b.rate_pct.iloc[-1]:.4f}x "
          f"| n = {int(b.n.iloc[0]):,} per basin")
    print(f"  basin rates vs the {overall:.4f} % overall mean: max deviation "
          f"{max(abs(b.rate_pct - overall)):.4f} pp")

    # ---- station -------------------------------------------------------------------
    s = tables["station_id"].join(d.groupby("station_id", observed=True)["station_name"].first())
    print("\n[3B.4] station spread (all 50 levels, sorted)")
    print(s.round(4).to_string())
    print(f"  highest {s.station_name.iloc[0]} ({s.index[0]}) {s.rate_pct.iloc[0]:.4f} % | "
          f"lowest {s.station_name.iloc[-1]} ({s.index[-1]}) {s.rate_pct.iloc[-1]:.4f} %")
    print(f"  ratio {s.rate_pct.iloc[0] / s.rate_pct.iloc[-1]:.4f}x | n = {int(s.n.iloc[0]):,} "
          f"per station (balanced panel) | no thin cell anywhere")
    print(f"  station rates: mean {s.rate_pct.mean():.4f} %, std {s.rate_pct.std():.4f} pp, "
          f"iqr {s.rate_pct.quantile(0.75) - s.rate_pct.quantile(0.25):.4f} pp")

    # =================================================================================
    # Figure 3B.1 - flood rate by month, monsoon window shaded
    # =================================================================================
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    x = np.arange(12)
    ax.bar(x, m["rate_pct"].to_numpy(), width=0.72, color="steelblue",
           label="next-day flood rate")
    ax.axvspan(4.5, 9.5, color="orange", alpha=0.16,
               label="monsoon window, Jun-Oct (monsoon_season = 1)")
    ax.axhline(overall, color="black", ls="--", lw=1.6,
               label=f"overall mean rate {overall:.3f} %")
    for xi, v in zip(x, m["rate_pct"].to_numpy()):
        ax.text(xi, v + 0.15, f"{v:.3f} %", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, 9.6)
    ax.set_xticks(x, [f"{MONTH_NAMES[i]}\nn = {int(m['n'].iloc[i]):,}\n"
                      f"{int(m['floods'].iloc[i]):,} floods" for i in x], fontsize=7.5)
    finish(ax,
           title=(f"Figure 3B.1 - Next-day flood rate by month against the {overall:.3f} % "
                  "overall mean\n"
                  "the monsoon window is shaded; the label is flood_event_occurred shifted "
                  "one day forward per station"),
           xlabel="month of day t (n = rows, floods = positives attributed to that month)",
           ylabel="flood rate on day t+1 (%)")
    savefig(fig, "fig33_flood_rate_month")

    # =================================================================================
    # Figure 3B.2 - flood rate by basin
    # =================================================================================
    fig, ax = plt.subplots(figsize=(11, 6.4))
    bb = b.sort_values("rate_pct")
    pos_y = np.arange(len(bb))
    ax.barh(pos_y, bb["rate_pct"].to_numpy(), height=0.68, color="seagreen",
            label="next-day flood rate")
    ax.axvline(overall, color="black", ls="--", lw=1.6,
               label=f"overall mean rate {overall:.3f} %")
    for yi, v in zip(pos_y, bb["rate_pct"].to_numpy()):
        ax.text(v - 0.05, yi, f"{v:.3f} %   (n = {int(bb['n'].iloc[yi]):,})",
                va="center", ha="right", fontsize=8.5, color="white")
    ax.set_xlim(0, 3.4)
    ax.set_yticks(pos_y, bb.index)
    box(ax, f"max/min = {b.rate_pct.iloc[0] / b.rate_pct.iloc[-1]:.2f}x\n"
            f"span {b.rate_pct.iloc[-1]:.3f} % - {b.rate_pct.iloc[0]:.3f} %\n"
            f"5 stations and {int(b.n.iloc[0]):,} rows per basin", "lower right")
    finish(ax,
           title=("Figure 3B.2 - Next-day flood rate by basin: a 1.27x span against a "
                  "2.166 % mean\ngeography is nearly worthless as a predictor here"),
           xlabel="flood rate on day t+1 (%)", ylabel="river basin (10 levels, sorted)",
           legend=False)
    ax.legend(loc="upper right", frameon=True)
    savefig(fig, "fig34_flood_rate_basin")

    # =================================================================================
    # Figure 3B.3 - flood rate by station
    # =================================================================================
    fig, ax = plt.subplots(figsize=(10.5, 13))
    ss = s.sort_values("rate_pct")
    pos_y = np.arange(len(ss))
    ax.barh(pos_y, ss["rate_pct"].to_numpy(), height=0.7, color="slateblue",
            label="next-day flood rate")
    ax.axvline(overall, color="black", ls="--", lw=1.6,
               label=f"overall mean rate {overall:.3f} %")
    ax.set_yticks(pos_y, [f"{n} ({i})" for i, n in zip(ss.index, ss.station_name)], fontsize=7)
    ax.set_xlim(0, 3.4)
    for yi in (0, len(ss) - 1):
        ax.text(ss["rate_pct"].iloc[yi] - 0.04, yi,
                f"{ss['rate_pct'].iloc[yi]:.3f} %", va="center", ha="right",
                fontsize=8.5, color="white", fontweight="bold")
    box(ax, f"max/min = {s.rate_pct.iloc[0] / s.rate_pct.iloc[-1]:.2f}x\n"
            f"{s.station_name.iloc[-1]} {s.rate_pct.iloc[-1]:.3f} % -> "
            f"{s.station_name.iloc[0]} {s.rate_pct.iloc[0]:.3f} %\n"
            f"n = {int(s.n.iloc[0]):,} rows for every one of the 50 stations\n"
            f"rate std across stations = {s.rate_pct.std():.3f} pp", "lower right")
    finish(ax,
           title=("Figure 3B.3 - Next-day flood rate by station, sorted: 1.63 % to 2.58 %, a "
                  "1.58x span\nall 50 stations carry 9,131 rows - there is no thin cell to "
                  "explain the spread"),
           xlabel="flood rate on day t+1 (%)",
           ylabel="station (station_name (station_id))", legend=False)
    ax.legend(loc="upper right", frameon=True)
    savefig(fig, "fig35_flood_rate_station")


# --------------------------------------------------------------------------------------
# 3C - numerical x categorical
# --------------------------------------------------------------------------------------
def section_3c(d: pd.DataFrame) -> None:
    rule("3C - NUMERICAL x CATEGORICAL (location AND spread)")

    # ---- by monsoon flag -----------------------------------------------------------
    g = d.groupby("monsoon_season")[MEAS].agg(["size", "mean", "std", "median"])
    print("\n[3C.0] the four measurements by monsoon_season (location and spread)")
    print(g.round(4).to_string())

    rain = d.groupby("monsoon_season")["rainfall_mm"].agg(["size", "mean", "std", "median"])
    loc = rain.loc[1, "mean"] / rain.loc[0, "mean"]
    spr = rain.loc[1, "std"] / rain.loc[0, "std"]
    print(f"\n[3C.1] rainfall by monsoon flag")
    for lvl in (0, 1):
        print(f"  monsoon_season={lvl}: n = {int(rain.loc[lvl, 'size']):,}, "
              f"mean = {rain.loc[lvl, 'mean']:.4f} mm/day, std = {rain.loc[lvl, 'std']:.4f} mm, "
              f"median = {rain.loc[lvl, 'median']:.4f} mm")
    print(f"  LOCATION moves {loc:.4f}x  ({rain.loc[0, 'mean']:.3f} -> {rain.loc[1, 'mean']:.3f} mm)")
    print(f"  SPREAD   moves {spr:.4f}x  ({rain.loc[0, 'std']:.3f} -> {rain.loc[1, 'std']:.3f} mm)")

    log = np.log1p(d["rainfall_mm"])
    lg = log.groupby(d["monsoon_season"]).agg(["mean", "std"])
    print(f"  after log1p: mean {lg.loc[0, 'mean']:.4f} -> {lg.loc[1, 'mean']:.4f} "
          f"(ratio {lg.loc[1, 'mean'] / lg.loc[0, 'mean']:.4f}x), std {lg.loc[0, 'std']:.4f} -> "
          f"{lg.loc[1, 'std']:.4f} (ratio {lg.loc[1, 'std'] / lg.loc[0, 'std']:.4f}x)")
    print(f"  => log1p cuts the spread ratio from {spr:.1f}x to "
          f"{lg.loc[1, 'std'] / lg.loc[0, 'std']:.2f}x while keeping the location shift")

    riv = d.groupby("monsoon_season")["river_level_m"].agg(["mean", "std"])
    print(f"\n  river level: location {riv.loc[0, 'mean']:.4f} -> {riv.loc[1, 'mean']:.4f} m "
          f"({riv.loc[1, 'mean'] / riv.loc[0, 'mean']:.4f}x), spread "
          f"{riv.loc[0, 'std']:.4f} -> {riv.loc[1, 'std']:.4f} m "
          f"({riv.loc[1, 'std'] / riv.loc[0, 'std']:.4f}x)")
    sm = d.groupby("monsoon_season")["soil_moisture_percent"].agg(["mean", "std"])
    print(f"  soil moisture: location {sm.loc[0, 'mean']:.4f} -> {sm.loc[1, 'mean']:.4f} % "
          f"({sm.loc[1, 'mean'] / sm.loc[0, 'mean']:.4f}x), spread "
          f"{sm.loc[0, 'std']:.4f} -> {sm.loc[1, 'std']:.4f} pp "
          f"({sm.loc[1, 'std'] / sm.loc[0, 'std']:.4f}x)")
    tp = d.groupby("monsoon_season")["temperature_celsius"].agg(["mean", "std"])
    print(f"  temperature:   location {tp.loc[0, 'mean']:.4f} -> {tp.loc[1, 'mean']:.4f} degC "
          f"({tp.loc[1, 'mean'] / tp.loc[0, 'mean']:.4f}x), spread "
          f"{tp.loc[0, 'std']:.4f} -> {tp.loc[1, 'std']:.4f} degC "
          f"({tp.loc[1, 'std'] / tp.loc[0, 'std']:.4f}x)")

    # ---- by basin ------------------------------------------------------------------
    b = d.groupby("basin", observed=True)[MEAS].agg(["mean", "std"])
    print("\n[3C.2] the four measurements by basin (location and spread)")
    print(b.round(4).to_string())
    print("\n  across the 10 basins: location ratio (max mean / min mean) and "
          "spread ratio (max std / min std)")
    for c in MEAS:
        mu, sd = b[(c, "mean")], b[(c, "std")]
        print(f"    {c:24s} mean {mu.min():9.4f} - {mu.max():9.4f} {UNITS[c]:6s} "
              f"({mu.max() / mu.min():.4f}x) | std {sd.min():8.4f} - {sd.max():8.4f} "
              f"({sd.max() / sd.min():.4f}x)")
    print(f"    within-basin std is "
          f"{b[('rainfall_mm', 'std')].mean() / b[('rainfall_mm', 'mean')].mean():.2f}x the "
          f"across-basin spread of the rainfall means -> the boxes overlap completely")

    # ---- variance decomposition ----------------------------------------------------
    print("\n[3C.3] one-way variance decomposition of rainfall_mm (the null result, quantified)")
    grand = d["rainfall_mm"].mean()
    for key in ["station_id", "basin"]:
        grp = d.groupby(key, observed=True)["rainfall_mm"]
        means = grp.mean()
        n_i = grp.size()
        ss_b = float((n_i * (means - grand) ** 2).sum())
        ss_w = float(sum(((v - means[k]) ** 2).sum() for k, v in grp))
        ss_t = float(((d["rainfall_mm"] - grand) ** 2).sum())
        G, N = len(means), len(d)
        print(f"  by {key:11s}: eta^2 = SS_between/SS_total = {100 * ss_b / ss_t:.4f} %")
        print(f"                MS_between = {ss_b / (G - 1):.4f} | MS_within = {ss_w / (N - G):.4f}"
              f" | total variance = {ss_t / (N - 1):.4f}")
        print(f"                variance of the {G} group means = {means.var(ddof=0):.4f} mm^2 vs "
              f"mean within-group variance = {float(grp.var(ddof=1).mean()):.4f} mm^2 "
              f"({100 * means.var(ddof=0) / grp.var(ddof=1).mean():.4f} %)")

    # =================================================================================
    # Figure 3C.1 - rainfall by monsoon, raw vs log1p
    # =================================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
    boxprops = dict(facecolor="lightsteelblue", edgecolor="0.2")
    log_mean = [float(lg.loc[i, "mean"]) for i in (0, 1)]
    log_std = [float(lg.loc[i, "std"]) for i in (0, 1)]
    log_ratio = log_std[1] / log_std[0]
    panels = [
        (axes[0], "rainfall_mm", "rainfall (mm/day)",
         f"raw scale - a {spr:.1f}x change in spread",
         [f"{rain.loc[i, 'mean']:.3f} mm, std {rain.loc[i, 'std']:.3f} mm" for i in (0, 1)]),
        (axes[1], "_log1p", "log1p rainfall (log1p of mm/day)",
         f"after log1p - a {log_ratio:.2f}x change in spread",
         [f"{log_mean[i]:.3f}, std {log_std[i]:.3f}" for i in (0, 1)]),
    ]
    for ax, col, lab, title, stats_txt in panels:
        data = d.assign(_log1p=log)
        sns.boxplot(data=data, x="monsoon_season", y=col, order=[0, 1], ax=ax,
                    width=0.5, color="lightsteelblue", showmeans=True, fliersize=1.6,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6),
                    medianprops=dict(color="0.15", lw=1.6), boxprops=boxprops)
        ax.set_xticks([0, 1], [f"0 - outside monsoon\n(n = {int(rain.loc[0, 'size']):,})\n"
                               f"mean {stats_txt[0]}",
                               f"1 - monsoon Jun-Oct\n(n = {int(rain.loc[1, 'size']):,})\n"
                               f"mean {stats_txt[1]}"], fontsize=9)
        finish(ax, title=title, xlabel="monsoon_season flag", ylabel=lab, legend=False)
        handles = [Line2D([], [], marker="D", ls="none", markerfacecolor="white",
                          markeredgecolor="black", markersize=6, label="mean"),
                   Line2D([], [], color="0.15", lw=1.6, label="median"),
                   Patch(facecolor="lightsteelblue", edgecolor="0.2",
                         label="box = IQR, whiskers = 1.5 x IQR, dots = outliers")]
        ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True)
    fig.suptitle("Figure 3C.1 - Rainfall by monsoon flag, raw and log1p: the monsoon moves "
                 "location 5.4x and spread 16.6x", fontsize=12)
    savefig(fig, "fig36_rainfall_monsoon")

    # =================================================================================
    # Figure 3C.2 - the four measurements by basin
    # =================================================================================
    basins = sorted(d["basin"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    spec = [
        ("rainfall_mm", True), ("river_level_m", True),
        ("soil_moisture_percent", False), ("temperature_celsius", False),
    ]
    for ax, (col, logscale) in zip(axes.ravel(), spec):
        sns.boxplot(data=d, x="basin", y=col, order=basins, ax=ax, width=0.62,
                    color="lightsteelblue", showmeans=True, fliersize=1.4,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=5),
                    medianprops=dict(color="0.15", lw=1.4),
                    boxprops=dict(facecolor="lightsteelblue", edgecolor="0.2"))
        if logscale:
            ax.set_yscale("log")
        ax.set_xticks(range(len(basins)), basins, rotation=45, ha="right", fontsize=8)
        mu = d.groupby("basin", observed=True)[col].mean()
        sd = d.groupby("basin", observed=True)[col].std()
        unit = UNITS[col]
        finish(ax,
               title=(f"{LABELS[col]}: means {mu.min():.2f}-{mu.max():.2f} {unit} "
                      f"({mu.max() / mu.min():.2f}x), stds {sd.min():.2f}-{sd.max():.2f}"),
               xlabel="river basin (10 levels, alphabetical)",
               ylabel=f"{LABELS[col]} ({unit}{', log scale' if logscale else ''})",
               legend=False)
        handles = [Line2D([], [], marker="D", ls="none", markerfacecolor="white",
                          markeredgecolor="black", markersize=5, label="mean"),
                   Line2D([], [], color="0.15", lw=1.4, label="median"),
                   Patch(facecolor="lightsteelblue", edgecolor="0.2",
                         label="box = IQR, whiskers = 1.5 x IQR")]
        ax.legend(handles=handles, loc="upper right", fontsize=7.5, frameon=True)
    fig.suptitle("Figure 3C.2 - The four measurements by basin: location and spread are "
                 "essentially identical across all 10 basins (n = 456,550)", fontsize=12)
    savefig(fig, "fig37_measurements_basin")

    # =================================================================================
    # Figure 3C.3 - river level by basin
    # =================================================================================
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    order = list(b[("river_level_m", "mean")].sort_values().index)
    sns.boxplot(data=d, y="basin", x="river_level_m", order=order, ax=ax, width=0.66,
                color="lightsteelblue", showmeans=True, fliersize=1.4,
                meanprops=dict(marker="D", markerfacecolor="white",
                               markeredgecolor="black", markersize=5),
                medianprops=dict(color="0.15", lw=1.4),
                boxprops=dict(facecolor="lightsteelblue", edgecolor="0.2"))
    ax.set_xscale("log")
    mu = b[("river_level_m", "mean")]
    sd = b[("river_level_m", "std")]
    ax.set_yticks(range(len(order)),
                  [f"{k}   mean {mu[k]:.2f} m, std {sd[k]:.2f} m" for k in order], fontsize=8.5)
    box(ax, f"location: {mu.min():.3f} - {mu.max():.3f} m ({mu.max() / mu.min():.2f}x)\n"
            f"spread:   {sd.min():.3f} - {sd.max():.3f} m ({sd.max() / sd.min():.2f}x)\n"
            f"n = {int(d['basin'].value_counts().iloc[0]):,} rows per basin", "lower right")
    finish(ax,
           title=("Figure 3C.3 - River level by basin (log scale): every basin spans the same "
                  "range,\nthe 10 distributions are indistinguishable in location and spread"),
           xlabel="river level (m, log scale)", ylabel="river basin (sorted by mean)")
    savefig(fig, "fig38_river_level_basin")


# --------------------------------------------------------------------------------------
def main() -> None:
    configure_env()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    rule("T4 - STAGE 3 : BIVARIATE AND MULTIVARIATE ANALYSIS", char="#")
    raw = load_raw()
    d = add_next_day_target(raw)
    print(f"raw panel    : {len(raw):,} rows x {raw.shape[1]} columns, "
          f"{raw.station_id.nunique()} stations, {raw.date.nunique()} days")
    print(f"usable rows  : {len(d.dropna(subset=[TARGET_NEXT])):,} "
          f"(the last day of each station has no t+1)")
    d = d.dropna(subset=[TARGET_NEXT]).copy()
    d["y"] = d[TARGET_NEXT].astype(int)
    y = d["y"]
    print(f"label        : {TARGET_NEXT} = {TARGET}(t+1), "
          f"{int(y.sum()):,} positives = {100 * y.mean():.4f} %, "
          f"imbalance {int((y == 0).sum()) / y.sum():.2f} : 1")
    print("\nclass balance")
    print(class_balance(y).to_string())
    print("\nNo model is trained in this script: every number below is a descriptive "
          "statistic,\nand roc_auc_score is applied to single raw columns only.")

    section_3a(d, y)
    section_3b(d)
    section_3c(d)

    rule("T4 - DONE : 9 figures written to docs/projects/eda/figures/", char="#")


if __name__ == "__main__":
    main()
