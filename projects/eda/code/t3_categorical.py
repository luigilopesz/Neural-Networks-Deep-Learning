"""T3 - Stage 2B: univariate analysis of the categorical features.

Frequencies and cardinality for all seven categorical columns of the ASIA-FLOOD panel,
the rare / high-cardinality screen, and the redundancy + deterministic re-encoding checks
that decide which columns survive into the model.

Counting only: no classifier or regressor is fitted anywhere and nothing here is
stochastic. Run from this directory so that ``from data import ...`` resolves::

    python t3_categorical.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from data import (
    CATEGORICAL,
    DROPPED,
    NUMERIC,
    RANDOM_STATE,
    SEED,
    TARGET_NEXT,
    add_next_day_target,
    class_balance,
    configure_env,
    finish,
    load_raw,
    savefig,
)

# Must run before any BLAS/OMP pool spins up (named pipes are sandbox-blocked).
configure_env()

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

#: Every column analysed in item 2B, in report order.  ``monsoon_season`` is a 0/1 flag
#: and is listed under ``NUMERIC`` for the model, but for frequency purposes it is a
#: categorical with two levels - so it belongs in this table.
CAT_COLS: list[str] = [
    "station_id",
    "station_name",
    "basin",
    "country",
    "severity_level",
    "season",
    "monsoon_season",
]

#: A level holding less than this share of rows is "rare" by the usual rule of thumb.
RARE_PCT: float = 1.0

#: season is a deterministic function of month (stage 1B / note 01 section 3).
MONTH_TO_SEASON: dict[int, str] = {
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
    12: "Winter", 1: "Winter", 2: "Winter",
}

#: monsoon_season is a deterministic function of month (stage 1B / note 01 section 3).
MONSOON_MONTHS: set[int] = {6, 7, 8, 9, 10}

#: readability glosses for the two integer-valued categorical levels
LEVEL_GLOSS: dict[tuple[str, object], str] = {
    ("monsoon_season", 0): "0 (Nov-May)",
    ("monsoon_season", 1): "1 (Jun-Oct)",
}


# --------------------------------------------------------------------------------------
# Counting helpers
# --------------------------------------------------------------------------------------
def level_frame(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Levels of ``col`` with count and share of rows, most frequent first.

    Ties are broken alphabetically so the output is deterministic.
    """
    n = len(df)
    vc = df[col].value_counts()
    out = pd.DataFrame({"level": list(vc.index), "count": vc.to_numpy()})
    out = out.sort_values(["count", "level"], ascending=[False, True]).reset_index(drop=True)
    out["share_pct"] = out["count"] / n * 100.0
    return out


def label_levels(col: str, levels: list) -> str:
    """Readable name for a set of tied levels."""
    names = [LEVEL_GLOSS.get((col, v), str(v)) for v in levels]
    if len(names) == 1:
        return names[0]
    if len(names) <= 3:
        return " / ".join(names) + f" ({len(names)} tied)"
    return f"{len(names)} tied: " + ", ".join(names[:3]) + ", ..."


def summarise(df: pd.DataFrame, col: str) -> dict:
    """Cardinality, extremes and rare-level count for one categorical column."""
    n = len(df)
    lv = level_frame(df, col)
    top_n = int(lv["count"].iloc[0])
    bot_n = int(lv["count"].iloc[-1])
    top = lv.loc[lv["count"] == top_n, "level"].tolist()
    bot = lv.loc[lv["count"] == bot_n, "level"].tolist()
    return {
        "column": col,
        "n": n,
        "cardinality": len(lv),
        "most_label": label_levels(col, top),
        "most_count": top_n,
        "most_pct": top_n / n * 100.0,
        "least_label": label_levels(col, bot),
        "least_count": bot_n,
        "least_pct": bot_n / n * 100.0,
        "n_rare": int((lv["share_pct"] < RARE_PCT).sum()),
        "min_share": float(lv["share_pct"].min()),
        "max_share": float(lv["share_pct"].max()),
        "ratio": float(lv["share_pct"].max() / lv["share_pct"].min()),
        "levels": lv,
    }


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------
def section_frequencies(df: pd.DataFrame) -> list[dict]:
    n = len(df)
    print("=" * 108)
    print(f"1. FREQUENCIES AND CARDINALITY - all {len(CAT_COLS)} categorical columns, n = {n:,} rows")
    print("=" * 108)
    stats = []
    for col in CAT_COLS:
        s = summarise(df, col)
        stats.append(s)
        print(f"\n--- {col} --- cardinality {s['cardinality']}")
        print(f"    most frequent : {s['most_label']}  ->  {s['most_count']:,} rows "
              f"({s['most_pct']:.4f} % of rows)")
        print(f"    least frequent: {s['least_label']}  ->  {s['least_count']:,} rows "
              f"({s['least_pct']:.4f} % of rows)")
        print(f"    levels < {RARE_PCT:g} % of rows: {s['n_rare']}   |   "
              f"min share {s['min_share']:.4f} %   max share {s['max_share']:.4f} %   "
              f"max/min ratio {s['ratio']:.2f}x")
        lv = s["levels"]
        if len(lv) <= 10:
            for _, r in lv.iterrows():
                gloss = LEVEL_GLOSS.get((col, r["level"]), str(r["level"]))
                print(f"      {gloss:<24} {int(r['count']):>9,}  {r['share_pct']:>8.4f} %")
        else:
            print(f"      all {len(lv)} levels hold exactly {s['most_count']:,} rows "
                  f"({s['most_pct']:.4f} %) - panel is perfectly balanced")
    return stats


def section_markdown_table(stats: list[dict]) -> None:
    print("\n" + "=" * 108)
    print("2. FREQUENCY + CARDINALITY TABLE (markdown, for the report fragment)")
    print("=" * 108)
    print("| column | cardinality | most frequent level | count | % of rows | "
          "least frequent level | count | % of rows | levels < 1 % |")
    print("|---|---:|---|---:|---:|---|---:|---:|---:|")
    for s in stats:
        print(f"| `{s['column']}` | {s['cardinality']} | {s['most_label']} | {s['most_count']:,} | "
              f"{s['most_pct']:.4f} | {s['least_label']} | {s['least_count']:,} | "
              f"{s['least_pct']:.4f} | {s['n_rare']} |")


def section_rare_screen(stats: list[dict], usable: pd.DataFrame) -> None:
    print("\n" + "=" * 108)
    print("3. RARE-CATEGORY AND HIGH-CARDINALITY SCREEN")
    print("=" * 108)
    total_levels = sum(s["cardinality"] for s in stats)
    total_rare = sum(s["n_rare"] for s in stats)
    print(f"{'levels in total across the 7 columns':<50}: {total_levels}")
    print(f"{'levels holding < 1 % of rows':<50}: {total_rare}")

    def smallest(cols: list[str], label: str) -> None:
        cands = [s for s in stats if s["column"] in cols]
        share = min(s["min_share"] for s in cands)
        at_min = [s["column"] for s in cands if abs(s["min_share"] - share) < 1e-9]
        s0 = next(s for s in cands if s["column"] == at_min[0])
        lv = s0["levels"]
        levels = lv.loc[lv["count"] == lv["count"].min(), "level"].tolist()
        rows = int(lv["count"].min())
        print(f"{label:<50}: {share:7.4f} %  ({' / '.join(at_min)} -> "
              f"{label_levels(s0['column'], levels)}, {rows:,} rows)")

    smallest(CAT_COLS, "smallest level share anywhere")
    smallest([c for c in CAT_COLS if c not in ("station_id", "station_name")],
             "smallest share, station-identity columns excluded")
    smallest([c for c in CAT_COLS if c not in ("station_id", "station_name", "severity_level")],
             "smallest share, leaky column also excluded")
    by_card = sorted(stats, key=lambda s: (-s["cardinality"], s["column"]))
    print("\nhigh-cardinality screen (k > 20 is the usual trigger):")
    for s in by_card[:2]:
        print(f"  {s['column']:<14} k = {s['cardinality']:<3} "
              f"every level {s['most_count']:,} rows ({s['most_pct']:.4f} %) - "
              f"balanced, not skewed: the levels are not rare, there are simply many of them")
    print("  -> station_id   : kept as a grouping key (target shift, per-station windows), "
          "NOT a model input")
    print("  -> station_name : 1:1 with station_id -> dropped; one-hot on both would add "
          "50 + 50 = 100 columns for zero information")
    print("\nno level of any column falls below 1 %: this dataset has no rare category to "
          "collapse and no high-cardinality column to handle.")

    # Guard against the one confusion this item invites: the imbalance is in the TARGET,
    # not in the categorical levels (whose smallest share is 2.0000 %).
    bal = class_balance(usable[TARGET_NEXT].astype(int))
    pos = bal.loc[1]
    print(f"[context] next-day target: {int(pos['count']):,} positives of {len(usable):,} rows "
          f"({pos['share_pct']:.4f} %), imbalance {pos['imbalance_ratio']:.1f} : 1 - the imbalance "
          f"is in the target, not in the categorical levels")


def section_balance(df: pd.DataFrame) -> None:
    print("\n" + "=" * 108)
    print("4. PANEL BALANCE - rows per level")
    print("=" * 108)
    for key in ("station_id", "basin"):
        per = df.groupby(key).size()
        uniq = sorted(set(per.tolist()))
        print(f"rows per {key:<11}: n_levels {len(per):>3}  min {per.min():,}  max {per.max():,}  "
              f"unique values {uniq}")
    spb = df.groupby("basin")["station_id"].nunique()
    print(f"stations per basin : n_levels {len(spb):>3}  min {spb.min()}  max {spb.max()}  "
          f"unique values {sorted(set(spb.tolist()))}")
    print("-> balanced panel: 50 stations x 9,132 days = 456,600 rows; 10 basins x 45,660 rows.")


def section_redundancy(df: pd.DataFrame) -> None:
    n = len(df)
    print("\n" + "=" * 108)
    print("5. REDUNDANCY AND DETERMINISTIC RE-ENCODINGS")
    print("=" * 108)

    ids, names = df["station_id"].nunique(), df["station_name"].nunique()
    pairs = len(df[["station_id", "station_name"]].drop_duplicates())
    per_id = int(df.groupby("station_id")["station_name"].nunique().max())
    print(f"[1:1]      station_id {ids} levels, station_name {names} levels, "
          f"unique (station_id, station_name) pairs {pairs}, max names per station {per_id} "
          f"-> station_name is redundant (drop one)")

    basins, countries = df["basin"].nunique(), df["country"].nunique()
    cpb = int(df.groupby("basin")["country"].nunique().max())
    print(f"[function] basin {basins} levels, country {countries} levels, "
          f"max countries per basin {cpb} -> country is a function of basin (redundant)")
    print(pd.crosstab(df["basin"], df["country"]).to_string())

    mism_season = int((df["month"].map(MONTH_TO_SEASON) != df["season"]).sum())
    print(f"\n[re-encoding] month -> season          : {mism_season} mismatches of {n:,} rows")
    print(pd.crosstab(df["season"], df["month"]).to_string())

    expected_monsoon = df["month"].isin(MONSOON_MONTHS).astype(int)
    mism_monsoon = int((expected_monsoon != df["monsoon_season"].astype(int)).sum())
    print(f"\n[re-encoding] month -> monsoon_season  : {mism_monsoon} mismatches of {n:,} rows "
          f"(monsoon = month in {{6,7,8,9,10}})")
    print(pd.crosstab(df["month"], df["monsoon_season"]).to_string())

    uniq = df.groupby("station_id")[["latitude", "longitude"]].nunique()
    triples = len(df[["station_name", "latitude", "longitude"]].drop_duplicates())
    print(f"\n[static]   unique latitude / longitude per station: max "
          f"{int(uniq['latitude'].max())} / {int(uniq['longitude'].max())}; "
          f"unique (station_name, latitude, longitude) triples {triples} of {n:,} rows "
          f"-> geographic columns are station constants, not measurements")


def section_survivors(stats: list[dict]) -> None:
    print("\n" + "=" * 108)
    print("6. WHAT SURVIVES INTO THE MODEL")
    print("=" * 108)
    width = {s["column"]: s["cardinality"] for s in stats}
    kept = " + ".join(f"{c} ({width[c]})" for c in CATEGORICAL)
    print(f"CATEGORICAL kept : {CATEGORICAL}  -> one-hot width "
          f"{kept} = {sum(width[c] for c in CATEGORICAL)} indicator columns")
    print(f"NUMERIC kept     : {NUMERIC}")
    print("dropped, reason from data.DROPPED (see stage 1B for the full leakage proof):")
    for col in ("severity_level", "station_name", "country", "latitude", "longitude"):
        print(f"  {col:<15} {DROPPED[col]}")
    print("kept as a grouping key, not a feature: station_id (k = 50, balanced).")


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def figure_frequencies(df: pd.DataFrame) -> None:
    """Figure 2B.1 - level frequency of every categorical column."""
    fig = plt.figure(figsize=(15, 14))
    # NB: no hspace/wspace here - a GridSpec with locally modified spacing is reported
    # as tight_layout-incompatible by matplotlib (savefig calls fig.tight_layout()).
    gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.0, 1.0, 1.0])
    layout = {
        "station_id": (gs[0, :], True),
        "station_name": (gs[1, 0], False),
        "basin": (gs[1, 1], False),
        "country": (gs[2, 0], False),
        "severity_level": (gs[2, 1], False),
        "season": (gs[3, 0], False),
        "monsoon_season": (gs[3, 1], False),
    }
    for col, (cell, wide) in layout.items():
        ax = fig.add_subplot(cell)
        lv = level_frame(df, col)
        k = len(lv)
        counts = lv["count"].to_numpy(dtype=float)
        shares = lv["share_pct"].to_numpy(dtype=float)
        x = np.arange(k)
        leaky = col == "severity_level"
        ax.bar(x, counts, width=0.82, zorder=3,
               color="#c44e52" if leaky else "#4c72b0",
               hatch="//" if leaky else None, edgecolor="white")
        ax.set_yscale("log")
        ax.set_ylim(1_000, 900_000)

        if k <= 10:
            for xi, ci, si in zip(x, counts, shares):
                ax.annotate(f"{int(ci):,}\n({si:.2f} %)", (xi, ci), textcoords="offset points",
                            xytext=(0, 3), ha="center", va="bottom", fontsize=7.0)
        else:
            label = (f"all {k} levels = {int(counts[0]):,} rows ({shares[0]:.4f} %)\n"
                     f"perfectly balanced - no rare level")
            if col == "station_id":
                label += "\nstation_name has the identical 50 levels (1:1)"
            elif col == "station_name":
                label += "\nredundant with station_id"
            ax.text(0.5, 0.80, label, transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.5, bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.95})

        if col in ("station_id", "station_name"):
            ticks = [0, 9, 19, 29, 39, 49] if wide else [0, 24, 49]
            ax.set_xticks(ticks, [str(lv["level"].iloc[t]) for t in ticks], fontsize=7.5,
                          rotation=20, ha="right")
        elif col == "basin":
            ax.set_xticks(x, list(lv["level"]), fontsize=8, rotation=35, ha="right")
        elif col == "country":
            ax.set_xticks(x, list(lv["level"]), fontsize=8, rotation=25, ha="right")
        elif col == "monsoon_season":
            ax.set_xticks(x, [LEVEL_GLOSS[(col, v)] for v in lv["level"]], fontsize=9)
        else:
            ax.set_xticks(x, list(lv["level"]), fontsize=9)

        if leaky:
            ax.legend(handles=[Patch(facecolor="#c44e52", hatch="//", edgecolor="white",
                                     label="leaky column - dropped (stage 1B)")],
                      loc="upper right", fontsize=7.5, frameon=True)
        finish(ax, title=f"{col} - {k} levels", xlabel=f"{col} (category level)",
               ylabel="rows (count, log scale)", legend=False)

    fig.suptitle("Figure 2B.1 - level frequency of every categorical column "
                 "(456,600 rows, log count axis; every bar labelled with count and share)",
                 fontsize=13, y=0.975)
    savefig(fig, "fig20_category_frequencies")


def figure_cardinality(df: pd.DataFrame) -> None:
    """Figure 2B.2 - cardinality, and the level-share range per column."""
    stats = sorted((summarise(df, c) for c in CAT_COLS),
                   key=lambda s: (-s["cardinality"], s["column"]))
    names = [s["column"] for s in stats]
    ypos = np.arange(len(stats))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.4),
                                   gridspec_kw={"width_ratios": [1.0, 1.35]})

    # --- panel A: cardinality
    ks = [s["cardinality"] for s in stats]
    ax1.barh(ypos, ks, height=0.62, color="#4c72b0", zorder=3)
    ax1.set_xscale("log")
    ax1.set_xlim(1.4, 220)
    ax1.set_yticks(ypos, names, fontsize=10)
    for yv, s in zip(ypos, stats):
        ax1.annotate(f"{s['cardinality']}", (s["cardinality"], yv), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=9.5)
    finish(ax1, title="A. Cardinality of every categorical column",
           xlabel="distinct levels (count, log scale)", ylabel="categorical column",
           legend=False)

    # --- panel B: share range of the least / most frequent level
    disposition = {
        "station_id": "grouping key, not a feature",
        "station_name": "drop: 1:1 with station_id",
        "basin": "keep: one-hot (10)",
        "country": "drop: function of basin",
        "severity_level": "drop: leakage",
        "season": "keep: one-hot (4)",
        "monsoon_season": "keep: numeric flag",
    }
    for i, s in enumerate(stats):
        yv = ypos[i]
        ax2.plot([s["min_share"], s["max_share"]], [yv, yv], color="#9a9a9a", lw=2.2, zorder=2)
        ax2.scatter([s["max_share"]], [yv], marker="s", s=75, color="#c44e52", zorder=4,
                    label="most frequent level" if i == 0 else None)
        # open marker drawn on top: when min == max the ring sits inside the square and
        # the coincidence is visible instead of one marker hiding the other
        ax2.scatter([s["min_share"]], [yv], marker="o", s=75, zorder=5,
                    facecolor="white", edgecolor="#4c72b0", linewidths=1.6,
                    label="least frequent level" if i == 0 else None)
        tag = "x1.00 all levels identical" if s["ratio"] < 1.0001 else f"x{s['ratio']:.2f}"
        ax2.annotate(f"{tag}   k={s['cardinality']} - {disposition[s['column']]}",
                     (s["max_share"], yv), xytext=(7, 0), textcoords="offset points",
                     va="center", fontsize=8.5)
        ax2.annotate(f"{s['min_share']:.2f} %", (s["min_share"], yv), xytext=(-9, 0),
                     textcoords="offset points", va="center", ha="right", fontsize=8.5,
                     color="#4c72b0")
    ax2.axvline(RARE_PCT, color="red", ls="--", lw=1.5, zorder=1,
                label="rare-category threshold (< 1 % of rows)")
    ax2.set_xscale("log")
    ax2.set_xlim(0.55, 900)
    ax2.set_yticks(ypos, [f"{s['column']}  (k={s['cardinality']})" for s in stats], fontsize=10)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9.5,
               frameon=False)
    finish(ax2, title="B. Share of rows held by the least and most frequent level",
           xlabel="share of rows per level (%, log scale)", ylabel="categorical column",
           legend=False)
    fig.suptitle("Figure 2B.2 - cardinality and level balance: no level of any column "
                 "falls below 2.00 %, so no rare category exists",
                 fontsize=13, y=1.02)
    savefig(fig, "fig20_cardinality")


# --------------------------------------------------------------------------------------
def main() -> None:
    configure_env()
    raw = load_raw()
    frame = add_next_day_target(raw)
    usable = frame[frame[TARGET_NEXT].notna()].copy()

    print("=" * 108)
    print("T3 / STAGE 2B - UNIVARIATE ANALYSIS OF THE CATEGORICAL FEATURES")
    print("=" * 108)
    print(f"seed                 : {SEED} (RANDOM_STATE={RANDOM_STATE}) - counting only, "
          f"no stochastic step")
    print(f"full panel           : {len(raw):,} rows x {raw.shape[1]} columns "
          f"({raw['station_id'].nunique()} stations x {raw['date'].nunique():,} days, "
          f"{raw['date'].min():%Y-%m-%d} -> {raw['date'].max():%Y-%m-%d})")
    print(f"usable (t+1 target)  : {len(usable):,} rows")
    print(f"categorical columns  : {CAT_COLS}")

    stats = section_frequencies(raw)
    section_markdown_table(stats)
    section_rare_screen(stats, usable)
    section_balance(raw)
    section_redundancy(raw)
    section_survivors(stats)

    drift = 0.0
    for col in CAT_COLS:
        a = level_frame(raw, col).set_index("level")["share_pct"]
        b = level_frame(usable, col).set_index("level")["share_pct"]
        drift = max(drift, float((a - b).abs().max()))
    print(f"\n[frame check] max |share(full panel) - share(usable t+1 frame)| over the "
          f"{len(CAT_COLS)} columns = {drift:.4f} pp  ({len(raw) - len(usable)} rows dropped)")

    print("\n" + "=" * 108)
    print("7. FIGURES")
    print("=" * 108)
    figure_frequencies(raw)
    figure_cardinality(raw)


if __name__ == "__main__":
    main()
