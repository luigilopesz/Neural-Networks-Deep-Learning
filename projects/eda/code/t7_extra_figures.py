"""Extra figures for the EDA deliverable: leakage audit, split audit, outlier decision,
feature signal and the monsoon gate.

Unlike t1-t6, this script does NOT read the CSV. Every input below is a number the earlier
scripts already print and the report already tabulates; each constant names the table it
comes from, and the ``check()`` calls at the top assert that the transcribed numbers still add
up to the totals the report quotes (456,600 rows, 456,550 usable, 9,889 floods ...). If a
number in the report changes, this script fails loudly instead of drawing a stale chart.

    python t7_extra_figures.py        # writes five PNGs into ../figures/

Figures written
    fig01_leakage_audit.png    Figure 1.4  (item 1B)  the label is a threshold on a leaked column
    fig01_split_audit.png      Figure 1.5  (item 1D)  why a random split is invalid here
    fig40_outlier_decision.png Figure 4A.2 (item 4A)  what an IQR filter would delete
    fig60_feature_signal.png   Figure 5.1  (stage 5)  today's soil moisture beats today's rain
    fig60_monsoon_gate.png     Figure 5.2  (stage 5)  where the floods are, and a free baseline
"""

from __future__ import annotations

from data import configure_env, savefig

configure_env()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BLUE, NAVY, RED, AMBER, GREY = "#8fa8c8", "#3b6ea5", "#c0392b", "#e0a030", "#9a9a9a"
INK = "#2b2b2b"

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})

# --------------------------------------------------------------------------------------
# Inputs, each transcribed from a table of the report
# --------------------------------------------------------------------------------------
# Table 1B.3 - severity_level vs flood_risk_score vs same-day label (456,600 rows)
SEVERITY = [("Low", 234_226, 0), ("Moderate", 129_974, 0), ("High", 82_479, 0), ("Extreme", 9_921, 9_889)]
ROWS_ALL, FLOODS_SAME_DAY = 456_600, 9_889
RISK_MAX_NO_FLOOD, RISK_MIN_FLOOD, RISK_RANGE = 7.00, 7.01, (1.00, 10.00)
N_RISK_EQ_7 = 32  # Extreme but not a flood: the one internal contradiction

# Section 1B - ROC-AUC of single columns
AUC_RISK_SAME_DAY, AUC_RISK_NEXT_DAY, AUC_SOIL_NEXT_DAY = 1.0000, 0.8088, 0.8364

# Section 1D - random 80/20 split (random_state=42) vs the chronological split
RANDOM_SPLIT = {"blocks_both_sides": (1_250, 1_250), "dates_shared": (9_132, 9_132), "straddling_pairs": 145_898}
CHRONO_SPLIT = {"blocks_both_sides": (0, 1_250), "dates_shared": (0, 9_132), "straddling_pairs": 100}
# Section 1D - pooled within-station lag-1 autocorrelation
LAG1 = [("temperature_celsius", 0.968), ("soil_moisture_percent", 0.820), ("flood_risk_score", 0.772),
        ("river_level_m", 0.220), ("rainfall_mm", 0.162)]

# Section 4A.2 - the 1.5 x IQR fence on rainfall_mm (56.30 mm)
FENCE_MM = 56.30
SAME_DAY = {"flagged": 9_274, "rows": 456_600, "floods": 9_889, "rate_above": 100.000, "rate_below": 0.137}
NEXT_DAY_TRAIN = {"flagged": 6_244, "rows": 310_500, "floods": 6_566, "rate_above": 8.456, "rate_below": 1.985}

# Table 3A.4 / 3A text - single-column next-day ROC-AUC (leaky column flagged)
AUC_NEXT = [("soil_moisture_percent", 0.8364, False), ("rainfall_mm", 0.8091, False),
            ("flood_risk_score", 0.8088, True), ("monsoon_season", 0.7969, False),
            ("temperature_celsius", 0.7956, False), ("river_level_m", 0.7948, False),
            ("day_of_year", 0.6313, False), ("data_quality_score", 0.5072, False)]
# Section 3A - next-day flood rate by upper rainfall deciles
RAIN_DECILES = [("(10.76, 18.70]", 45_616, 32), ("(18.70, 23.63]", 45_675, 1_633), ("(23.63, 28.52]", 45_668, 3_491),
                ("(28.52, 33.22]", 45_667, 2_932), ("(33.22, 1775.72]", 45_589, 1_800)]

# Table 3B.1 - next-day label by month: (rows, floods)
MONTHS = {"Jan": (38_750, 0), "Feb": (35_350, 0), "Mar": (38_750, 0), "Apr": (37_500, 0), "May": (38_750, 1),
          "Jun": (37_500, 92), "Jul": (38_750, 3_149), "Aug": (38_750, 3_132), "Sep": (37_500, 3_045),
          "Oct": (38_750, 470), "Nov": (37_500, 0), "Dec": (38_700, 0)}
USABLE, FLOODS_NEXT_DAY = 456_550, 9_889


def check() -> None:
    """The transcribed numbers must still add up to the totals quoted in the report."""
    assert sum(r for _, r, _ in SEVERITY) == ROWS_ALL
    assert sum(f for _, _, f in SEVERITY) == FLOODS_SAME_DAY
    assert sum(r for r, _ in MONTHS.values()) == USABLE, sum(r for r, _ in MONTHS.values())
    assert sum(f for _, f in MONTHS.values()) == FLOODS_NEXT_DAY
    for d in (SAME_DAY, NEXT_DAY_TRAIN):
        # flood rate above/below the fence must reproduce the quoted total of floods
        above = round(d["flagged"] * d["rate_above"] / 100)
        below = d["floods"] - above
        assert abs(below / (d["rows"] - d["flagged"]) * 100 - d["rate_below"]) < 0.001, (d, below)


def pct(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f} %"


# --------------------------------------------------------------------------------------
# Figure 1.4 - leakage audit (item 1B)
# --------------------------------------------------------------------------------------
def fig_leakage() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})

    ax = axes[0]
    names = [s for s, _, _ in SEVERITY]
    rows = np.array([r for _, r, _ in SEVERITY])
    floods = np.array([f for _, _, f in SEVERITY])
    x = np.arange(len(names))
    ax.bar(x, rows - floods, color=BLUE, label="no flood (same day)")
    ax.bar(x, floods, bottom=rows - floods, color=RED, label="flood (same day)")
    ax.set_yscale("log")
    ax.set_ylim(1e3, 4e7)
    for i, (r, f) in enumerate(zip(rows, floods)):
        ax.text(i, r * 1.25, f"{r:,} rows\n{f:,} floods", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{n}\n{b}" for n, b in zip(names, ["[1, 3)", "[3, 5)", "[5, 7)", "[7, 10]"])])
    ax.set_xlabel("severity_level (bin of flood_risk_score)")
    ax.set_ylabel("Station-days (count, log scale)")
    ax.set_title("(a) Three of four severity bins contain no flood at all")
    ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=9)

    ax = axes[1]
    lo, hi = RISK_RANGE
    ax.barh([1], [RISK_MAX_NO_FLOOD - lo], left=lo, color=BLUE, height=0.5, label="no flood: max risk 7.00")
    ax.barh([0], [hi - RISK_MIN_FLOOD], left=RISK_MIN_FLOOD, color=RED, height=0.5, label="flood: min risk 7.01")
    ax.axvline(7.0, color=INK, ls="--", lw=1.4, label="rule: risk > 7.0")
    ax.annotate(f"{N_RISK_EQ_7} rows sit exactly on 7.00:\n'Extreme' bin, yet not floods\n(the dataset's one contradiction)",
                xy=(7.0, 1), xytext=(1.0, 0.45), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK),
                bbox=dict(boxstyle="round", fc="#fdf2d0", ec=GREY))
    ax.set_yticks([0, 1], ["flood = 1\n(9,889 rows)", "flood = 0\n(446,711 rows)"])
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("flood_risk_score (index 1-10)")
    ax.set_title("(b) The two classes touch at 7.00 and never overlap")
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    ax = axes[2]
    labels = ["risk score ->\nsame-day label", "risk score ->\nnext-day label", "soil moisture ->\nnext-day label"]
    vals = [AUC_RISK_SAME_DAY, AUC_RISK_NEXT_DAY, AUC_SOIL_NEXT_DAY]
    bars = ax.bar(labels, vals, color=[RED, BLUE, NAVY])
    ax.axhline(0.5, color=GREY, ls=":", lw=1.4)
    ax.text(2.45, 0.512, "chance (0.5)", ha="right", fontsize=9, color=GREY)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}", ha="center", fontsize=10, color=INK)
    ax.set_ylim(0.4, 1.1)
    ax.set_ylabel("ROC-AUC of one raw column")
    ax.set_title("(c) Leak for today, merely redundant for tomorrow")
    ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("Figure 1.4  Leakage audit: flood_event_occurred == (flood_risk_score > 7.0) on all 456,600 rows",
                 fontsize=13, y=1.02)
    savefig(fig, "fig01_leakage_audit")


# --------------------------------------------------------------------------------------
# Figure 1.5 - split audit (item 1D)
# --------------------------------------------------------------------------------------
def fig_split() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), gridspec_kw={"width_ratios": [1.1, 0.9, 1.0]})

    ax = axes[0]
    metrics = ["(station, year) blocks\nwith rows on both sides", "calendar dates\nseen on both sides"]
    keys = ["blocks_both_sides", "dates_shared"]
    rnd = [RANDOM_SPLIT[k][0] / RANDOM_SPLIT[k][1] * 100 for k in keys]
    chr_ = [CHRONO_SPLIT[k][0] / CHRONO_SPLIT[k][1] * 100 for k in keys]
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, rnd, w, color=RED, label="random 80/20, random_state=42")
    ax.bar(x + w / 2, chr_, w, color=NAVY, label="chronological (adopted)")
    for xi, v in zip(x - w / 2, rnd):
        ax.text(xi, v + 2, pct(v, 1), ha="center", fontsize=10)
    for xi, v in zip(x + w / 2, chr_):
        ax.text(xi, v + 2, pct(v, 1), ha="center", fontsize=10)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 150)
    ax.set_ylabel("Share leaking across the boundary (%)")
    ax.set_title("(a) A random split shares every block and every date")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    ax = axes[1]
    vals = [RANDOM_SPLIT["straddling_pairs"], CHRONO_SPLIT["straddling_pairs"]]
    bars = ax.bar(["random", "chronological"], vals, color=[RED, NAVY])
    ax.set_yscale("log")
    ax.set_ylim(10, 6e5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.3, f"{v:,}", ha="center", fontsize=10)
    ax.set_ylabel("Adjacent-day pairs split across train/test (log)")
    ax.set_title("(b) 1,459x more near-duplicate pairs")
    ax.set_xlabel("chronological = 2 year boundaries x 50 stations", fontsize=9)

    ax = axes[2]
    names = [n for n, _ in LAG1][::-1]
    vals = [v for _, v in LAG1][::-1]
    cols = [RED if v > 0.7 else BLUE for v in vals]
    ax.barh(names, vals, color=cols)
    ax.axvline(0.7, color=GREY, ls=":", lw=1.2)
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Within-station lag-1 autocorrelation")
    ax.set_title("(c) Yesterday's row is a near-copy of today's")
    ax.legend(handles=[Patch(color=RED, label="> 0.7: test row ~ a training row"), Patch(color=BLUE, label="<= 0.7")],
              loc="lower right", frameon=True, fontsize=9)

    fig.suptitle("Figure 1.5  Split audit: why the 2000-2016 / 2017-2020 / 2021-2024 chronological split was chosen",
                 fontsize=13, y=1.02)
    savefig(fig, "fig01_split_audit")


# --------------------------------------------------------------------------------------
# Figure 4A.2 - outlier decision (item 4A)
# --------------------------------------------------------------------------------------
def fig_outliers() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.1, 1.0]})
    cases = [("same-day label\n(stages 1-3, 456,600 rows)", SAME_DAY),
             ("next-day label = the target\n(train, 310,500 rows)", NEXT_DAY_TRAIN)]

    ax = axes[0]
    x = np.arange(2)
    w = 0.36
    above = [c["rate_above"] for _, c in cases]
    below = [c["rate_below"] for _, c in cases]
    ax.bar(x - w / 2, above, w, color=RED, label=f"rainfall > {FENCE_MM:.2f} mm (flagged by the IQR rule)")
    ax.bar(x + w / 2, below, w, color=BLUE, label=f"rainfall <= {FENCE_MM:.2f} mm")
    for xi, v in zip(x - w / 2, above):
        ax.text(xi, v + 2, pct(v, 3), ha="center", fontsize=10)
    for xi, v in zip(x + w / 2, below):
        ax.text(xi, v + 2, pct(v, 3), ha="center", fontsize=10)
    ax.set_xticks(x, [n for n, _ in cases])
    ax.set_ylim(0, 118)
    ax.set_ylabel("Flood rate (%)")
    ax.set_title("(a) Flagged rows are flood days; tomorrow's lift is 4.26x")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    ax = axes[1]
    deleted = []
    for _, c in cases:
        n_above = round(c["flagged"] * c["rate_above"] / 100)
        deleted.append(n_above / c["floods"] * 100)
    y = np.arange(2)[::-1]
    ax.barh(y, deleted, color=RED, label="positives an IQR filter would delete")
    ax.barh(y, [100 - d for d in deleted], left=deleted, color=BLUE, label="positives that survive")
    for yi, d, (_, c) in zip(y, deleted, cases):
        n_above = round(c["flagged"] * c["rate_above"] / 100)
        ax.text(d / 2 if d > 20 else d + 1.5, yi, f"{d:.1f} %\n({n_above:,} of {c['floods']:,})",
                ha="center" if d > 20 else "left", va="center", fontsize=10,
                color="white" if d > 20 else INK, fontweight="bold")
    ax.set_yticks(y, [n for n, _ in cases])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of all positives (%)")
    ax.set_title("(b) Rows an IQR filter removes also remove positives")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=True, fontsize=9)

    fig.suptitle("Figure 4A.2  The 'outliers' are the event being forecast: 0 rows removed, 0 winsorised",
                 fontsize=13, y=1.02)
    savefig(fig, "fig40_outlier_decision")


# --------------------------------------------------------------------------------------
# Figure 5.1 - feature signal (stage 5)
# --------------------------------------------------------------------------------------
def fig_signal() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.0), gridspec_kw={"width_ratios": [1.15, 0.85]})

    ax = axes[0]
    names = [n for n, _, _ in AUC_NEXT][::-1]
    vals = [v for _, v, _ in AUC_NEXT][::-1]
    leaky = [k for _, _, k in AUC_NEXT][::-1]
    colors = [NAVY if n == "soil_moisture_percent" else BLUE for n in names]
    bars = ax.barh(names, vals, color=colors)
    for b, k in zip(bars, leaky):
        if k:
            b.set_hatch("//")
            b.set_facecolor("white")
            b.set_edgecolor(RED)
    ax.axvline(0.5, color=GREY, ls=":", lw=1.4)
    ax.text(0.503, 7.45, "chance", fontsize=9, color=GREY)
    for i, v in enumerate(vals):
        ax.text(v + 0.006, i, f"{v:.4f}", va="center", fontsize=10)
    ax.set_xlim(0.45, 0.93)
    ax.set_xlabel("ROC-AUC of one raw column against the next-day label (no model fitted)")
    ax.set_title("(a) Today's soil moisture is the best single predictor of tomorrow")
    ax.legend(handles=[Patch(color=NAVY, label="best honest column"), Patch(color=BLUE, label="other features"),
                       Patch(fc="white", ec=RED, hatch="//", label="flood_risk_score (excluded: leaky/redundant)")],
              loc="lower right", frameon=True, fontsize=9)

    ax = axes[1]
    labels = [d for d, _, _ in RAIN_DECILES]
    rates = [f / n * 100 for _, n, f in RAIN_DECILES]
    colors = [BLUE] * 5
    colors[int(np.argmax(rates))] = NAVY
    colors[-1] = RED
    bars = ax.bar(range(5), rates, color=colors)
    for i, (r, (_, n, f)) in enumerate(zip(rates, RAIN_DECILES)):
        ax.text(i, r + 0.18, f"{r:.3f} %", ha="center", fontsize=10)
    ax.set_xticks(range(5), [f"D{6 + i}\n{lab}" for i, lab in enumerate(labels)], fontsize=8.5)
    ax.set_ylim(0, 9.4)
    ax.set_xlabel("rainfall_mm decile (mm/day), upper five deciles")
    ax.set_ylabel("Next-day flood rate (%)")
    ax.set_title("(b) The wettest decile floods less than the 8th")
    ax.legend(handles=[Patch(color=NAVY, label="peak (D8)"), Patch(color=RED, label="top decile (D10)")],
              loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("Figure 5.1  Soil moisture is the standing precondition; rainfall is the same-day trigger",
                 fontsize=13, y=1.02)
    savefig(fig, "fig60_feature_signal")


# --------------------------------------------------------------------------------------
# Figure 5.2 - monsoon gate (stage 5)
# --------------------------------------------------------------------------------------
def fig_gate() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.0), gridspec_kw={"width_ratios": [1.1, 0.9]})

    ax = axes[0]
    order = sorted(MONTHS, key=lambda m: -(MONTHS[m][1] / MONTHS[m][0]))
    cum_rows = np.cumsum([MONTHS[m][0] for m in order]) / USABLE * 100
    cum_floods = np.cumsum([MONTHS[m][1] for m in order]) / FLOODS_NEXT_DAY * 100
    xs = np.r_[0, cum_rows]
    ys = np.r_[0, cum_floods]
    ax.plot([0, 100], [0, 100], color=GREY, ls=":", lw=1.4, label="no calendar information")
    ax.plot(xs, ys, color=RED, lw=2.4, marker="o", ms=5, label="months ranked by flood rate")
    for i, m in enumerate(order[:5]):
        ax.annotate(m, (cum_rows[i], cum_floods[i]), textcoords="offset points", xytext=(6, -14), fontsize=9)
    i3 = 2  # first three months of the ranking = Jul, Sep, Aug
    ax.annotate(f"Jul-Sep: {cum_rows[i3]:.1f} % of station-days\nhold {cum_floods[i3]:.2f} % of all floods",
                xy=(cum_rows[i3], cum_floods[i3]), xytext=(38, 52), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=INK), bbox=dict(boxstyle="round", fc="#fdf2d0", ec=GREY))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Cumulative share of station-days (%)")
    ax.set_ylabel("Cumulative share of next-day floods (%)")
    ax.set_title("(a) Cumulative gain by month: the calendar finds the floods")
    ax.legend(loc="lower right", frameon=True)

    ax = axes[1]
    gates = [("monsoon flag\n(Jun-Oct)", ["Jun", "Jul", "Aug", "Sep", "Oct"]),
             ("Jul-Sep only", ["Jul", "Aug", "Sep"])]
    x = np.arange(len(gates))
    w = 0.36
    rec, prec = [], []
    for _, ms in gates:
        rows = sum(MONTHS[m][0] for m in ms)
        fl = sum(MONTHS[m][1] for m in ms)
        rec.append(fl / FLOODS_NEXT_DAY * 100)
        prec.append(fl / rows * 100)
    ax.bar(x - w / 2, rec, w, color=NAVY, label="recall (floods caught)")
    ax.bar(x + w / 2, prec, w, color=AMBER, label="precision (flagged days that flood)")
    for xi, v in zip(x - w / 2, rec):
        ax.text(xi, v + 1.5, pct(v, 2), ha="center", fontsize=10)
    for xi, v in zip(x + w / 2, prec):
        ax.text(xi, v + 1.5, pct(v, 2), ha="center", fontsize=10)
    ax.axhline(100 * FLOODS_NEXT_DAY / USABLE, color=GREY, ls=":", lw=1.3, label="overall base rate 2.17 %")
    ax.set_xticks(x, [g for g, _ in gates])
    ax.set_ylim(0, 150)
    ax.set_ylabel("Metric (%)")
    ax.set_title("(b) The monsoon flag: 99.99 % recall at 5.17 % precision")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("Figure 5.2  The monsoon gate: the baseline every model must beat inside the window",
                 fontsize=13, y=1.02)
    savefig(fig, "fig60_monsoon_gate")


if __name__ == "__main__":
    check()
    fig_leakage()
    fig_split()
    fig_outliers()
    fig_signal()
    fig_gate()
    # numbers quoted in the report text for these figures
    n_above = round(SAME_DAY["flagged"] * 1.0)
    print(f"same-day floods above the fence : {n_above:,} of {SAME_DAY['floods']:,} = "
          f"{n_above / SAME_DAY['floods'] * 100:.2f} %")
    n_nd = round(NEXT_DAY_TRAIN["flagged"] * NEXT_DAY_TRAIN["rate_above"] / 100)
    print(f"next-day train positives flagged: {n_nd:,} of {NEXT_DAY_TRAIN['floods']:,} = "
          f"{n_nd / NEXT_DAY_TRAIN['floods'] * 100:.2f} %")
    jas = sum(MONTHS[m][1] for m in ("Jul", "Aug", "Sep"))
    print(f"Jul-Sep share of floods         : {jas:,} of {FLOODS_NEXT_DAY:,} = {jas / FLOODS_NEXT_DAY * 100:.2f} %")
    print(f"pairs ratio random/chronological: {RANDOM_SPLIT['straddling_pairs'] / CHRONO_SPLIT['straddling_pairs']:,.0f}x")
