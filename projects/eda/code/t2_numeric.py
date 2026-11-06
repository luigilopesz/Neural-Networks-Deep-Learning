#!/usr/bin/env python
"""EDA stage 2A - univariate analysis of the numerical features.

Task T2 of the EDA deliverable. Answers the rubric item *2A. Numerical*:
descriptive statistics for **every** numerical column, plus exactly three figures
chosen and interpreted. No model is trained anywhere in this file: only counts,
moments, quantiles, boundary diagnostics and matplotlib.

What it prints (every number quoted in ``sections/s2a-numeric.md``):

* the descriptive table for all 13 numerical columns of the raw panel
  (count / mean / std / min / 1 % / 25 % / median / 75 % / 99 % / max /
  skewness / excess kurtosis / n and % outside the 1.5 x IQR fence);
* the 1.5 x IQR fence bounds per column;
* raw vs ``log1p`` shape comparison for the four measurements;
* boundary-mass diagnostics - exact floor/ceiling detection with bin evidence;
* the coordinate-cardinality check behind the flat 1 %/99 % quantiles;
* the cross-reference to the stage-1B verdict on the dropped columns.

What it writes (exactly three PNGs, the stage-2 budget is 3 per item):

* ``figures/fig10_measurements_hist_log.png``
* ``figures/fig10_measurements_box.png``
* ``figures/fig10_temperature_shape.png``

Run from this directory so that ``from data import ...`` resolves::

    cd docs/projects/eda/code
    python t2_numeric.py

Deterministic: ``SEED`` (= 42) drives the only stochastic step, the KDE subsample.
"""

from __future__ import annotations

import os

# The DSH file sandbox forbids named pipes, so any BLAS / joblib / OpenMP thread pool
# dies with PermissionError [WinError 5]. Pin every pool to one thread *before* numpy
# is imported - after the import this has no effect. ``configure_env()`` below is the
# seam's idempotent copy of the same contract.
for _var in ("OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys  # noqa: E402

# Windows defaults stdout to cp1252, which cannot encode the U+2212 minus sign used in the
# log1p label below. Piped or redirected output would then die with UnicodeEncodeError before
# a single figure was written. Force UTF-8 so `python t2_numeric.py > out.txt` works.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from data import (  # noqa: E402
    DROPPED,
    N_STATIONS,
    RAW_MEASUREMENTS,
    SEED,
    TARGET_NEXT,
    add_next_day_target,
    configure_env,
    finish,
    load_raw,
    savefig,
)

# --------------------------------------------------------------------------------------
# Scope of the univariate stage
# --------------------------------------------------------------------------------------
MEASUREMENTS = list(RAW_MEASUREMENTS)  # the four genuine sensor columns

#: every numerical column of the raw panel, in report order, with the role it plays.
#: ``flood_event_occurred`` is deliberately absent - it is the label, handled in 1C.
NUMERIC_COLUMNS = [
    "rainfall_mm",
    "river_level_m",
    "soil_moisture_percent",
    "temperature_celsius",
    "latitude",
    "longitude",
    "flood_risk_score",
    "data_quality_score",
    "year",
    "month",
    "day",
    "day_of_year",
    "monsoon_season",
]

#: group separators used in the emitted markdown table, so the report keeps one table
#: for the whole stage while still grouping the columns by role.
TABLE_GROUPS = {
    "rainfall_mm": "**Measurements** (raw sensor columns)",
    "latitude": "**Static geography** (constant within station)",
    "flood_risk_score": "**Outcome-derived** (dropped \u2014 see the caveat below)",
    "year": "**Calendar** (deterministic re-encodings of `date`)",
    "monsoon_season": "**Seasonal gate**",
}

UNITS = {
    "rainfall_mm": "mm",
    "river_level_m": "m",
    "soil_moisture_percent": "%",
    "temperature_celsius": "\u00b0C",
    "latitude": "decimal degrees",
    "longitude": "decimal degrees",
    "flood_risk_score": "index 1-10",
    "data_quality_score": "score 0-1",
    "year": "year",
    "month": "month",
    "day": "day of month",
    "day_of_year": "day of year",
    "monsoon_season": "0/1 flag",
}

#: display precision of the location statistics. Moment statistics never go below 2 dp.
DECIMALS = {
    "latitude": 4,
    "longitude": 4,
    "data_quality_score": 4,
    "year": 0,
    "month": 0,
    "day": 0,
    "day_of_year": 0,
    "monsoon_season": 0,
}
DEFAULT_DECIMALS = 2
SKEW_DECIMALS = 3

SEP = "=" * 100
SUB = "-" * 100


def fmt(value: float, decimals: int) -> str:
    return f"{value:,.{decimals}f}"


def skew(values: np.ndarray) -> float:
    """Fisher-Pearson adjusted sample skewness (scipy ``bias=False``, = pandas ``.skew()``)."""
    return float(stats.skew(values, bias=False))


def kurtosis(values: np.ndarray) -> float:
    """Excess kurtosis, bias-corrected (scipy ``bias=False``, = pandas ``.kurt()``)."""
    return float(stats.kurtosis(values, bias=False))


def fence(values: np.ndarray) -> tuple[float, float, float, int]:
    """Return ``(q1, q3, upper_fence, n_outside)`` for the 1.5 x IQR rule."""
    q1, q3 = (float(x) for x in np.percentile(values, [25, 75]))
    iqr = q3 - q1
    upper, lower = q3 + 1.5 * iqr, q1 - 1.5 * iqr
    n_out = int(np.sum((values < lower) | (values > upper)))
    return q1, q3, upper, n_out


def log1p_panel(values: np.ndarray, column: str) -> tuple[np.ndarray, str]:
    """``log1p`` of the column, shifted when it has negative values (temperature only)."""
    if values.min() < 0:
        shift = float(values.min())
        return np.log1p(values - shift), f"log1p({column} \u2212 {abs(shift):.2f}) (shifted, dimensionless)"
    return np.log1p(values), f"log1p({column}) (dimensionless)"


# --------------------------------------------------------------------------------------
# 1. Descriptive statistics for every numerical column
# --------------------------------------------------------------------------------------
def describe_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric summary and fence summary for all numerical columns."""
    records, fences = [], []
    for column in NUMERIC_COLUMNS:
        v = df[column].to_numpy(dtype=float)
        q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
        p1, p99 = (float(x) for x in np.percentile(v, [1, 99]))
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n_out = int(np.sum((v < lower) | (v > upper)))
        records.append(
            {
                "column": column, "count": v.size, "mean": v.mean(), "std": v.std(ddof=1),
                "min": v.min(), "p1": p1, "q25": q1, "median": med, "q75": q3, "p99": p99,
                "max": v.max(), "skewness": skew(v), "kurtosis": kurtosis(v),
                "n_out": n_out, "pct_out": 100.0 * n_out / v.size,
            }
        )
        fences.append(
            {
                "column": column, "q25": q1, "q75": q3, "iqr": q3 - q1,
                "lower_fence": lower, "upper_fence": upper,
                "n_out": n_out, "pct_out": 100.0 * n_out / v.size,
            }
        )
    return pd.DataFrame(records), pd.DataFrame(fences)


def markdown_descriptives(stats_df: pd.DataFrame) -> None:
    """Print the descriptive table in GitHub/Material markdown, ready to paste."""
    header = [
        "column", "count", "mean", "std", "min", "1 %", "25 %", "median",
        "75 %", "99 %", "max", "skewness", "kurtosis", "n outside", "% outside",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] + [":--:"] * (len(header) - 1)) + "|")
    for _, row in stats_df.iterrows():
        column = row["column"]
        if column in TABLE_GROUPS:
            print("| " + " | ".join([TABLE_GROUPS[column]] + [""] * (len(header) - 1)) + " |")
        loc_dec = DECIMALS.get(column, DEFAULT_DECIMALS)
        mom_dec = max(loc_dec, 2)
        cells = [
            f"`{column}`",
            f"{int(row['count']):,}",
            fmt(row["mean"], mom_dec),
            fmt(row["std"], mom_dec),
            fmt(row["min"], loc_dec),
            fmt(row["p1"], loc_dec),
            fmt(row["q25"], loc_dec),
            fmt(row["median"], loc_dec),
            fmt(row["q75"], loc_dec),
            fmt(row["p99"], loc_dec),
            fmt(row["max"], loc_dec),
            f"{row['skewness']:.{SKEW_DECIMALS}f}",
            f"{row['kurtosis']:.{SKEW_DECIMALS}f}",
            f"{int(row['n_out']):,}",
            f"{row['pct_out']:.2f}",
        ]
        print("| " + " | ".join(cells) + " |")


def markdown_fences(fences: pd.DataFrame) -> None:
    """Print the 1.5 x IQR fence table in markdown."""
    header = ["column", "25 % (Q1)", "75 % (Q3)", "IQR", "lower fence", "upper fence",
              "n outside", "% outside"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] + [":--:"] * (len(header) - 1)) + "|")
    for _, row in fences.iterrows():
        column = row["column"]
        dec = DECIMALS.get(column, DEFAULT_DECIMALS)
        cells = [
            f"`{column}`", fmt(row["q25"], dec), fmt(row["q75"], dec), fmt(row["iqr"], dec),
            fmt(row["lower_fence"], dec), fmt(row["upper_fence"], dec),
            f"{int(row['n_out']):,}", f"{row['pct_out']:.2f}",
        ]
        print("| " + " | ".join(cells) + " |")


# --------------------------------------------------------------------------------------
# 2. Boundary diagnostics - is a min/max a real order statistic or a clamp?
# --------------------------------------------------------------------------------------
def boundary_bins(values: np.ndarray, step: float = 0.1, side: str = "top") -> np.ndarray:
    """Counts in the seven ``step``-wide bins that end (top) or start (bottom) at the bound."""
    if side == "top":
        edges = np.round(values.max() - step * np.arange(7, -1, -1), 6)
    else:
        edges = np.round(values.min() + step * np.arange(0, 8), 6)
    counts, _ = np.histogram(values, bins=edges)
    return counts


# --------------------------------------------------------------------------------------
# 3. Figures
# --------------------------------------------------------------------------------------
def figure_hist_log(df: pd.DataFrame) -> None:
    """Figure 2A.1 - raw vs log1p histograms of the four measurements."""
    fig, axes = plt.subplots(2, 4, figsize=(20.0, 9.2))
    colors = {"raw": "#2b6cb0", "log1p": "#2f855a"}

    for col_index, column in enumerate(MEASUREMENTS):
        raw = df[column].to_numpy(dtype=float)
        logv, log_label = log1p_panel(raw, column)
        n_cap = int(np.sum(raw == raw.max()))
        for row_index, (values, tag) in enumerate(((raw, "raw"), (logv, "log1p"))):
            ax = axes[row_index, col_index]
            ax.hist(values, bins=80, color=colors[tag], edgecolor="white", linewidth=0.25, label=tag)
            ax.axvline(values.mean(), color="#c53030", ls="--", lw=1.3, label=f"mean = {values.mean():,.2f}")
            ax.axvline(float(np.median(values)), color="#b7791f", ls=":", lw=1.8,
                       label=f"median = {np.median(values):,.2f}")
            title = (f"{column}\n{tag}: skew {skew(values):.2f}, excess kurtosis {kurtosis(values):.2f}")
            if tag == "raw":
                xlabel, ylabel = f"{column} ({UNITS[column]})", f"Count, raw (n = {values.size:,})"
            else:
                xlabel, ylabel = log_label, "Count, log1p"
            if column == "soil_moisture_percent" and tag == "raw":
                ax.annotate(
                    f"censored ceiling: {n_cap:,} rows\nat exactly {raw.max():.2f} "
                    f"({100 * n_cap / raw.size:.2f} %)",
                    xy=(raw.max(), n_cap), xytext=(0.34, 0.58), textcoords="axes fraction",
                    fontsize=9, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0),
                    bbox=dict(boxstyle="round", fc="#fffaf0", ec="#b7791f", alpha=0.95),
                )
            if column == "soil_moisture_percent" and tag == "log1p":
                ax.annotate(
                    f"the ceiling stays a single tied value:\nlog1p({raw.max():.2f}) = "
                    f"{np.log1p(raw.max()):.2f}, still {n_cap:,} rows ({100 * n_cap / raw.size:.2f} %)",
                    xy=(np.log1p(raw.max()), n_cap), xytext=(0.30, 0.62), textcoords="axes fraction",
                    fontsize=9, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0),
                    bbox=dict(boxstyle="round", fc="#f0fff4", ec="#2f855a", alpha=0.95),
                )
            finish(ax, title=title, xlabel=xlabel, ylabel=ylabel)
            ax.get_legend().get_frame().set_alpha(0.95)
            ax.title.set_fontsize(11)

    fig.suptitle(
        "Figure 2A.1 \u2014 the four measurements, raw (top) vs log1p (bottom): log1p repairs the two "
        "severe right skews, leaves the censored tie intact, and creates skew where there was none",
        fontsize=13.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    savefig(fig, "fig10_measurements_hist_log", tight=False)


def figure_boxplots(df: pd.DataFrame) -> None:
    """Figure 2A.2 - boxplots of the four measurements with the 1.5 x IQR fence."""
    fig, axes = plt.subplots(2, 2, figsize=(17.0, 9.6))

    for ax, column in zip(axes.ravel(), MEASUREMENTS):
        v = df[column].to_numpy(dtype=float)
        unit = UNITS[column]
        q1, q3, upper, n_out = fence(v)
        lower = q1 - 1.5 * (q3 - q1)
        median = float(np.median(v))
        n_at_max = int(np.sum(v == v.max()))

        ax.boxplot(
            v, orientation="horizontal", widths=0.30, patch_artist=True, whis=1.5,
            boxprops=dict(facecolor="#bee3f8", edgecolor="#2b6cb0"),
            medianprops=dict(color="black", linewidth=2.0),
            whiskerprops=dict(color="#2b6cb0"),
            capprops=dict(color="#2b6cb0", linewidth=1.5),
            flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#c53030",
                            markeredgecolor="none", alpha=0.30),
        )
        ax.axvline(upper, color="crimson", ls="--", lw=1.8,
                   label=f"1.5\u00d7IQR fence = {upper:,.2f} {unit}")
        ax.axvline(q1, color="#4a5568", ls=":", lw=1.2)
        ax.axvline(q3, color="#4a5568", ls=":", lw=1.2, label=f"Q1 / Q3 = {q1:,.2f} / {q3:,.2f} {unit}")

        x_lo, x_hi = v.min(), max(v.max(), upper)
        pad = 0.05 * (x_hi - x_lo)
        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.set_ylim(0.35, 1.65)
        ax.set_yticks([])

        if n_at_max >= 100:
            tail_text = (f"right-censored at the maximum:\n{n_at_max:,} rows at exactly "
                         f"{v.max():,.2f} {unit} ({100 * n_at_max / v.size:.2f} %)")
        else:
            tail_text = (f"extreme tail: max = {v.max():,.2f} {unit}\n"
                         f"= {v.max() / median:,.1f}\u00d7 the median")
        ax.annotate(
            tail_text, xy=(v.max(), 1.0), xytext=(0.985, 0.95), textcoords="axes fraction",
            ha="right", va="top", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0),
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95),
        )
        finish(ax,
               title=f"{column} \u2014 {n_out:,} station-days outside the fence ({100 * n_out / v.size:.2f} %)",
               xlabel=f"{column} ({unit})", ylabel=f"All station-days (n = {v.size:,})", legend=False)
        ax.title.set_fontsize(11.5)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

        if n_out > 0:
            inset = ax.inset_axes((0.42, 0.13, 0.555, 0.30))
            inset.boxplot(
                v[v <= upper], orientation="horizontal", widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="#fbd38d", edgecolor="#b7791f"),
                medianprops=dict(color="black", linewidth=1.6),
                flierprops=dict(marker="o", markersize=1.5, markerfacecolor="#c53030",
                                markeredgecolor="none", alpha=0.4),
            )
            inset.set_title(f"bulk view: min \u2192 fence (n = {int((v <= upper).sum()):,})", fontsize=8)
            inset.set_xlabel(f"{column} ({unit})", fontsize=7, labelpad=1.5)
            inset.set_yticks([])
            inset.tick_params(labelsize=7)
            inset.grid(alpha=0.3)
            for spine in inset.spines.values():
                spine.set_edgecolor("#b7791f")
        else:
            ax.text(0.5, 0.16,
                    f"no point outside the fence: it sits at {upper:,.2f} {unit},\n"
                    f"{upper - v.max():,.2f} {unit} beyond the maximum",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round", fc="#f0fff4", ec="#2f855a", alpha=0.95))

    fig.suptitle(
        "Figure 2A.2 \u2014 1.5\u00d7IQR fences on the four measurements: the flagged station-days are the "
        "tail of the distribution, not measurement error",
        fontsize=13.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    savefig(fig, "fig10_measurements_box", tight=False)


def figure_temperature(df: pd.DataFrame) -> None:
    """Figure 2A.3 - why temperature_celsius is flat: it is a seasonal sinusoid."""
    column = "temperature_celsius"
    v = df[column].to_numpy(dtype=float)
    monthly = df.groupby("month")[column].agg(["mean", "std", "size"])
    n_at_max = int(np.sum(v == v.max()))

    fig, (ax_hist, ax_cycle) = plt.subplots(1, 2, figsize=(18.0, 6.4))

    # --- panel A: pooled distribution, with the 12 monthly means as a rug overlay
    ax_hist.hist(v, bins=120, density=True, color="#cbd5e0", edgecolor="white",
                 linewidth=0.15, label=f"pooled histogram (n = {v.size:,})")
    rng = np.random.default_rng(SEED)
    sub = rng.choice(v, size=20_000, replace=False)
    grid = np.linspace(v.min(), v.max(), 500)
    ax_hist.plot(grid, stats.gaussian_kde(sub)(grid), color="#2b6cb0", lw=2.0,
                 label="Gaussian KDE (20,000-row subsample, seed 42)")
    y_top = ax_hist.get_ylim()[1]
    ax_hist.plot(monthly["mean"].to_numpy(), np.full(len(monthly), 0.04 * y_top), marker="v", ls="none",
                 markersize=9, color="#dd6b20",
                 label="monthly mean temperature (\u00b0C) \u2014 the seasonal cycle")
    ax_hist.set_ylim(0.0, y_top * 1.18)
    ax_hist.annotate(
        f"skew {skew(v):.3f}, excess kurtosis {kurtosis(v):.2f}\n"
        f"min {v.min():.2f} \u00b0C, max {v.max():.2f} \u00b0C, 0 rows outside the IQR fence",
        xy=(0.42, 0.60), xycoords="axes fraction", ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95),
    )
    ax_hist.annotate(
        f"also right-censored: {n_at_max:,} rows\nat exactly {v.max():.2f} \u00b0C "
        f"({100 * n_at_max / v.size:.2f} %)",
        xy=(v.max(), n_at_max / v.size / (v.max() - v.min()) * 120 * 0.9),
        xytext=(0.70, 0.86), textcoords="axes fraction", fontsize=9, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0),
        bbox=dict(boxstyle="round", fc="#fffaf0", ec="#b7791f", alpha=0.95),
    )
    finish(ax_hist, title=f"{column} \u2014 pooled density across all {v.size:,} station-days",
           xlabel="Temperature (\u00b0C)", ylabel="Probability density (1/\u00b0C)")
    ax_hist.title.set_fontsize(11.5)
    ax_hist.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # --- panel B: the sinusoid that generates the flat pooled shape
    months = monthly.index.to_numpy()
    ax_cycle.errorbar(months, monthly["mean"], yerr=monthly["std"], color="#2f855a",
                      marker="o", markersize=7, lw=2.0, capsize=4,
                      label=f"monthly mean \u00b1 1 sd (n = {int(monthly['size'].iloc[0]):,} station-days per month)")
    ax_cycle.axhline(v.mean(), color="#4a5568", ls="--", lw=1.3,
                     label=f"annual mean = {v.mean():.2f} \u00b0C")
    lo_m, hi_m = monthly["mean"].idxmin(), monthly["mean"].idxmax()
    ax_cycle.annotate(
        f"cycle amplitude {monthly['mean'].max() - monthly['mean'].min():.2f} \u00b0C\n"
        f"({monthly['mean'].loc[lo_m]:.2f} \u00b0C in month {lo_m} \u2192 "
        f"{monthly['mean'].loc[hi_m]:.2f} \u00b0C in month {hi_m})",
        xy=(hi_m, monthly["mean"].loc[hi_m]),
        xytext=(hi_m - 5.2, monthly["mean"].min() + 3.2),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0),
        bbox=dict(boxstyle="round", fc="#f0fff4", ec="#2f855a", alpha=0.95),
    )
    ax_cycle.set_xticks(months)
    ax_cycle.set_ylim(monthly["mean"].min() - 3.0, monthly["mean"].max() + 3.4)
    finish(ax_cycle, title="monthly mean \u00b1 1 sd \u2014 one smooth sinusoid over the year",
           xlabel="Month (1 = January \u2026 12 = December)", ylabel="Mean temperature (\u00b0C)")
    ax_cycle.title.set_fontsize(11.5)
    ax_cycle.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "Figure 2A.3 \u2014 temperature_celsius is flat, not bell-shaped, because one smooth seasonal "
        "cycle sweeps the whole range",
        fontsize=13.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    savefig(fig, "fig10_temperature_shape", tight=False)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    configure_env()

    print(SEP)
    print("T2 / stage 2A - univariate analysis of the numerical features")
    print(SEP)

    df = load_raw()
    df_next = add_next_day_target(df)
    print(f"\n[panel]        raw frame            : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[panel]        stations x days      : {df['station_id'].nunique()} x {df['date'].nunique()} "
          f"(= {df.shape[0]:,} rows, balanced panel)")
    print(f"[panel]        usable for the label : {int(df_next[TARGET_NEXT].notna().sum()):,} rows "
          f"(last day of each station has no t+1)")
    print(f"[panel]        missing values       : {int(df.isna().sum().sum())} "
          f"(descriptive statistics below use all {df.shape[0]:,} rows)")

    # ---------------------------------------------------------------- descriptive table
    stats_df, fences_df = describe_all(df)

    print(f"\n{SUB}\n[1] DESCRIPTIVE STATISTICS - ALL 13 NUMERICAL COLUMNS (markdown, paste-ready)\n{SUB}")
    print("Skewness = Fisher-Pearson adjusted G1 (scipy bias=False, identical to pandas .skew()).")
    print("Kurtosis = EXCESS kurtosis, bias-corrected (scipy bias=False, identical to pandas .kurt()).")
    print("'outside' = rows below Q1-1.5*IQR or above Q3+1.5*IQR.\n")
    markdown_descriptives(stats_df)

    print(f"\n{SUB}\n[2] 1.5 x IQR FENCES (markdown, paste-ready)\n{SUB}\n")
    markdown_fences(fences_df)

    # ------------------------------------------------------------ headline measurements
    print(f"\n{SUB}\n[3] THE FOUR MEASUREMENTS - HEADLINE NUMBERS CITED IN THE REPORT\n{SUB}")
    print(f"{'column':<23} {'min':>9} {'median':>9} {'mean':>9} {'max':>10} {'skew':>7} {'kurt':>8} "
          f"{'fence':>8} {'out':>7} {'%out':>6} {'max/med':>8} {'max/fence':>9}")
    for column in MEASUREMENTS:
        v = df[column].to_numpy(dtype=float)
        _, _, upper, n_out = fence(v)
        print(f"{column:<23} {v.min():>9.2f} {np.median(v):>9.2f} {v.mean():>9.2f} {v.max():>10.2f} "
              f"{skew(v):>7.2f} {kurtosis(v):>8.2f} {upper:>8.2f} {n_out:>7,} "
              f"{100 * n_out / v.size:>5.2f}% {v.max() / np.median(v):>7.1f}x "
              f"{v.max() / upper:>8.1f}x")
    print("\n(units: mm, m, %, \u00b0C in the order printed above)")
    print("\nNon-measurement columns cited in the report:")
    for column in ("latitude", "longitude", "flood_risk_score", "data_quality_score"):
        v = df[column].to_numpy(dtype=float)
        print(f"{column:<23} min={v.min():>9.4f} max={v.max():>9.4f} n_unique={df[column].nunique():>5} "
              f"skew={skew(v):>7.3f} kurt={kurtosis(v):>8.3f} p1={np.percentile(v, 1):.4f} "
              f"p99={np.percentile(v, 99):.4f}")
    print(f"\ndata_quality_score grid step check: all values on a 0.001 grid -> "
          f"{bool(np.allclose(df['data_quality_score'] * 1000, np.round(df['data_quality_score'] * 1000)))}")
    print(f"flood_risk_score  grid step check: all values on a 0.01 grid  -> "
          f"{bool(np.allclose(df['flood_risk_score'] * 100, np.round(df['flood_risk_score'] * 100)))}")

    # --------------------------------------------------------------- raw vs log1p shape
    print(f"\n{SUB}\n[4] RAW vs log1p - WHICH DISTRIBUTIONS THE TRANSFORM ACTUALLY REPAIRS\n{SUB}")
    for column in MEASUREMENTS:
        v = df[column].to_numpy(dtype=float)
        logv, label = log1p_panel(v, column)
        print(f"\n{column}  ({UNITS[column]})   transform: {label}")
        print(f"  raw    mean {v.mean():>9.2f}  median {np.median(v):>9.2f} "
              f"(mean = {v.mean() / np.median(v):.2f}x median)   skew {skew(v):>7.2f}  "
              f"excess kurtosis {kurtosis(v):>8.2f}")
        print(f"  log1p  mean {logv.mean():>9.2f}  median {np.median(logv):>9.2f} "
              f"(mean = {logv.mean() / np.median(logv):.2f}x median)   skew {skew(logv):>7.2f}  "
              f"excess kurtosis {kurtosis(logv):>8.2f}")

    # ------------------------------------------------------------- boundary / censoring
    print(f"\n{SUB}\n[5] BOUNDARY MASS - IS A MIN/MAX A REAL ORDER STATISTIC OR A CLAMP?\n{SUB}")
    print(f"{'column':<23} {'min':>10} {'n at min':>10} {'% at min':>9} {'max':>10} {'n at max':>10} {'% at max':>9}")
    for column in MEASUREMENTS:
        v = df[column].to_numpy(dtype=float)
        n_lo, n_hi = int(np.sum(v == v.min())), int(np.sum(v == v.max()))
        print(f"{column:<23} {v.min():>10.2f} {n_lo:>10,} {100 * n_lo / v.size:>8.3f}% "
              f"{v.max():>10.2f} {n_hi:>10,} {100 * n_hi / v.size:>8.3f}%")

    print(f"\n{SUB}\n[5b] BIN EVIDENCE - IS THE EXACT BOUND VALUE A CLAMP OR AN ORDER STATISTIC?\n{SUB}")
    print("n_exact   = rows whose value equals the bound exactly (all four columns are recorded")
    print("            to 2 decimals, so the smallest distinguishable slot is 0.01 wide).")
    print("ref/0.01  = mean count of the six adjacent 0.1-wide bins, divided by 10 = the count a")
    print("            single 0.01-wide slot would hold if the density near the bound were smooth.")
    print("verdict   = CLAMPED when n_exact >= 100 and the ratio exceeds 10x; otherwise the bound")
    print("            is one ordinary order statistic in a sparse tail.\n")
    print(f"{'column':<23} {'bound':>10} {'n_exact':>9} {'mean_6bins':>11} {'ref/0.01':>9} "
          f"{'ratio':>10}   verdict")
    clamp_report = {}
    for column in MEASUREMENTS:
        v = df[column].to_numpy(dtype=float)
        for side, bound in (("top", v.max()), ("bottom", v.min())):
            bins = boundary_bins(v, 0.1, side)
            n_exact = int(np.sum(v == bound))
            neighbours = bins[:-1] if side == "top" else bins[1:]
            ref = float(neighbours.mean() / 10.0)
            ratio = n_exact / ref if ref > 0 else float("inf")
            clamped = n_exact >= 100 and ref > 0 and ratio > 10
            label = f"{'CEILING - CLAMPED' if clamped else 'order statistic'}" if side == "top" \
                else f"{'FLOOR - CLAMPED' if clamped else 'order statistic'}"
            print(f"{column:<23} {bound:>10.2f} {n_exact:>9,} {neighbours.mean():>11.1f} {ref:>9.1f} "
                  f"{(f'{ratio:,.0f}x' if np.isfinite(ratio) else 'n/a'):>10}   {label}")
            clamp_report[(column, side)] = clamped
        print(f"{'':<23} {'7 bins ending at the max:':<33} "
              f"{', '.join(f'{c:,}' for c in boundary_bins(v, 0.1, 'top'))}")

    soil = df["soil_moisture_percent"].to_numpy(dtype=float)
    temp = df["temperature_celsius"].to_numpy(dtype=float)
    cap = int(np.sum(soil == 100.0))
    print(f"\nsoil_moisture_percent: {cap:,} rows exactly at 100.00 ({100 * cap / soil.size:.2f} %), "
          f"{df['soil_moisture_percent'].nunique():,} distinct values, next value below = "
          f"{np.sort(np.unique(soil))[-2]:.2f}")
    print(f"  of the 9,180 rows outside the fence, {cap:,} are the ceiling tie and "
          f"{9180 - cap:,} are genuine tail readings in (98.32, 100.00)")
    n45 = int(np.sum(temp == 45.0))
    print(f"\ntemperature_celsius : {n45:,} rows exactly at 45.00 ({100 * n45 / temp.size:.3f} %), "
          f"{df['temperature_celsius'].nunique():,} distinct values -> temperature is ALSO "
          f"right-censored, {9153 / n45:.1f}x less often than soil moisture")
    print(f"  the temperature fence sits at 61.55 \u00b0C, {61.55 - temp.max():.2f} \u00b0C beyond the "
          f"ceiling, so the censoring is invisible to the 1.5xIQR rule (0 rows flagged)")
    print(f"  clamp verdicts: soil_moisture ceiling={clamp_report[('soil_moisture_percent', 'top')]}, "
          f"temperature ceiling={clamp_report[('temperature_celsius', 'top')]}, "
          f"rainfall ceiling={clamp_report[('rainfall_mm', 'top')]}, "
          f"river ceiling={clamp_report[('river_level_m', 'top')]}")

    # -------------------------------------------------------- coordinate cardinality
    print(f"\n{SUB}\n[6] WHY latitude/longitude HAVE FLAT 1 % / 99 % QUANTILES\n{SUB}")
    pairs = df[["latitude", "longitude"]].drop_duplicates()
    print(f"distinct (latitude, longitude) pairs: {len(pairs):,}  "
          f"(= {N_STATIONS} stations; both columns are constant within station)")
    print(f"distinct latitude values: {df['latitude'].nunique():,} | distinct longitude values: "
          f"{df['longitude'].nunique():,}")
    for column in ("latitude", "longitude"):
        v = df[column].to_numpy(dtype=float)
        p1, p99 = np.percentile(v, [1, 99])
        print(f"{column:<10} min={v.min():.4f} p1={p1:.4f} -> equal: {bool(p1 == v.min())} | "
              f"max={v.max():.4f} p99={p99:.4f} -> equal: {bool(p99 == v.max())} | "
              f"mass at the smallest value = {100 * np.mean(v == v.min()):.3f} % (> 1 %)")
    print("Each of the 50 coordinate values carries 9,132/456,600 = 2.000 % of the rows, so the")
    print("1st percentile cannot leave the smallest value and the 99th cannot leave the largest.")

    # ---------------------------------------------------------- calendar determinism
    print(f"\n{SUB}\n[7] CALENDAR COLUMNS - DETERMINISTIC RE-ENCODINGS OF date\n{SUB}")
    rec = df["date"].dt.year * 10000 + df["date"].dt.month * 100 + df["date"].dt.day
    print(f"date vs (year, month, day) mismatches : "
          f"{int(np.sum(rec != df['year'] * 10000 + df['month'] * 100 + df['day']))}")
    print(f"date vs day_of_year mismatches        : "
          f"{int(np.sum(df['date'].dt.dayofyear.to_numpy() != df['day_of_year'].to_numpy()))}")
    print(f"day_of_year range {df['day_of_year'].min()}-{df['day_of_year'].max()}, rows at 366 (leap day): "
          f"{int(np.sum(df['day_of_year'] == 366)):,}")
    for column in ("year", "month", "day", "day_of_year", "monsoon_season"):
        v = df[column].to_numpy(dtype=float)
        q1, q3 = np.percentile(v, [25, 75])
        _, _, upper, n_out = fence(v)
        print(f"{column:<15} unique={df[column].nunique():>4} min={v.min():>6.0f} max={v.max():>6.0f} "
              f"fence=({q1 - 1.5 * (q3 - q1):>8.1f}, {upper:>8.1f}) outside={n_out:,}")

    # ------------------------------------------------------------- monthly temperature
    print(f"\n{SUB}\n[8] MONTHLY MEAN TEMPERATURE - THE SINUSOID BEHIND FIGURE 2A.3\n{SUB}")
    monthly = df.groupby("month")["temperature_celsius"].agg(["mean", "std", "min", "max", "size"])
    print(monthly.round(3).to_string())
    amp = monthly["mean"].max() - monthly["mean"].min()
    print(f"\namplitude of the seasonal cycle (max monthly mean - min monthly mean) = {amp:.2f} \u00b0C")
    print(f"coldest month {monthly['mean'].idxmin()} ({monthly['mean'].min():.2f} \u00b0C), "
          f"warmest month {monthly['mean'].idxmax()} ({monthly['mean'].max():.2f} \u00b0C)")
    print(f"pooled mean {df['temperature_celsius'].mean():.2f} \u00b0C; within-month sd spans "
          f"{monthly['std'].min():.2f}-{monthly['std'].max():.2f} \u00b0C -> the sd is far smaller than "
          f"the {amp:.2f} \u00b0C cycle, which is why the pooled density is flat")
    print(f"cycle / within-month-sd ratio: {amp / monthly['std'].max():.1f}x (largest monthly sd) to "
          f"{amp / monthly['std'].min():.1f}x (smallest monthly sd)")

    # ------------------------------------------------------------------- dropped cols
    print(f"\n{SUB}\n[9] DROPPED COLUMNS - CROSS-REFERENCE TO STAGE 1B (not re-derived here)\n{SUB}")
    for column in ("flood_risk_score", "data_quality_score", "latitude", "longitude"):
        print(f"{column}:\n    {DROPPED[column]}")
    print("\nThe two columns are described above for completeness of the univariate stage only;")
    print("the leakage rule and the null correlation were established on all 456,600 rows in 1B.")

    # ------------------------------------------------------------------------ figures
    print(f"\n{SUB}\n[10] FIGURES - EXACTLY THREE (the stage-2 budget is 3 per item)\n{SUB}")
    figure_hist_log(df)
    figure_boxplots(df)
    figure_temperature(df)
    print("\n3 figures written to docs/projects/eda/figures/ (prefix fig10_).")

    print(f"\n{SEP}\nDONE - no model was trained, no estimator was fitted.\n{SEP}")


if __name__ == "__main__":
    main()
