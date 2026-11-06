"""Shared data layer for the EDA deliverable.

Every EDA script imports from here so that the dataset path, the target definition,
the split boundaries and the feature lists have exactly one definition.

    from data import (load_raw, add_next_day_target, train_test_frames,
                      NUMERIC, CATEGORICAL, DROPPED, TARGET, SEED, FIGDIR, savefig)

Run anything from anywhere; paths are resolved relative to this file.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: scripts are run offline and commit PNGs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------------------
# Seed
# --------------------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent          # docs/projects/eda/code
EDA_DIR = CODE_DIR.parent                           # docs/projects/eda
REPO_ROOT = EDA_DIR.parents[2]                      # repo root
FIGDIR = EDA_DIR / "figures"
DATA_DIR = REPO_ROOT / "docs" / "projects" / "data" / "ASIA_FLOOD_25YEAR_DATASET" / "data"

FULL_CSV = DATA_DIR / "station_daily_data_full.csv"
SAMPLE_CSV = DATA_DIR / "station_daily_data.csv"
VULN_CSV = DATA_DIR / "vulnerability_data.csv"
GEOJSON = DATA_DIR / "asia_flood_geospatial.geojson"

FIGDIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Column roles
# --------------------------------------------------------------------------------------
RAW_MEASUREMENTS = [
    "rainfall_mm",
    "river_level_m",
    "soil_moisture_percent",
    "temperature_celsius",
]

#: label for the deliverable
TARGET = "flood_event_occurred"

#: one-day-ahead label, built by :func:`add_next_day_target`
TARGET_NEXT = "flood_next_day"

#: columns kept as numeric model inputs
NUMERIC = [
    "rainfall_mm",
    "river_level_m",
    "soil_moisture_percent",
    "temperature_celsius",
    "day_of_year",
    "monsoon_season",
]

#: columns kept as categorical model inputs
CATEGORICAL = ["basin", "season"]

#: columns excluded from the feature matrix, with the reason that goes in the report.
#: This dict is the single source of truth for report section 1B / results-table row 4.
DROPPED: dict[str, str] = {
    "severity_level": (
        "target leakage - a deterministic binning of flood_risk_score: "
        "cut(risk, [1,3,5,7,10.01], right=False) reproduces it on all 456,600 rows "
        "(the top edge must be closed: the 31 rows at risk == 10.00 fall outside an "
        "open [7,10) bin, and the default right=True reproduces only 450,602)"
    ),
    "flood_risk_score": (
        "target leakage / redundant - risk > 7.0 reproduces the same-day label on "
        "456,600/456,600 rows (ROC-AUC 1.0000), and it correlates r=0.934 with "
        "soil_moisture_percent while being a weaker next-day predictor (AUC 0.8088)"
    ),
    "data_quality_score": (
        "noise - |r| <= 0.004 against every other column; next-day ROC-AUC 0.5072"
    ),
    "station_name": "redundant - 1:1 with station_id (50 levels each)",
    "country": "redundant - a function of basin (exactly one country per basin)",
    "day": "noise - day-of-month, r=0.0007 with the next-day target",
    "latitude": "constant within station - 1 unique value for each of the 50 stations",
    "longitude": "constant within station - 1 unique value for each of the 50 stations",
}

#: years per split - chronological, by date, identical dates for every station
SPLITS: dict[str, tuple[int, int]] = {
    "train": (2000, 2016),
    "validation": (2017, 2020),
    "test": (2021, 2024),
}

DATES = {"min": "2000-01-01", "max": "2024-12-31"}
N_STATIONS = 50
N_DAYS = 9132

#: statement + project plan both fix this
RANDOM_STATE = SEED


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the full daily panel, sorted by (station_id, date), dates parsed.

    456,600 rows x 21 columns. Nothing is dropped, imputed or transformed here - this is
    the raw frame every quality check in stage 1 runs against.
    """
    df = pd.read_csv(path or FULL_CSV, parse_dates=["date"])
    return df.sort_values(["station_id", "date"], kind="stable").reset_index(drop=True)


def add_next_day_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``flood_next_day`` = ``flood_event_occurred`` on the following day, per station.

    This is the deliverable's label. The last day of each station has no t+1 and is NaN
    (50 rows over the whole panel), so the usable frame has 456,550 rows.
    """
    out = df.copy()
    out[TARGET_NEXT] = out.groupby("station_id")[TARGET].shift(-1)
    return out


def label_split(years: pd.Series) -> pd.Series:
    """Map a year Series to 'train' / 'validation' / 'test'."""
    out = pd.Series(pd.NA, index=years.index, dtype="object")
    for name, (lo, hi) in SPLITS.items():
        out[(years >= lo) & (years <= hi)] = name
    return out


def train_test_frames(df: pd.DataFrame | None = None, target: str = TARGET_NEXT):
    """Return ``(X_train, X_test, y_train, y_test)`` with the raw feature columns.

    ``X`` holds ``NUMERIC + CATEGORICAL`` only: dropped columns never enter, and no
    statistic of any kind is fitted here. Use :mod:`pipeline` to transform.
    """
    if df is None:
        df = add_next_day_target(load_raw())
    frame = df.dropna(subset=[target]).copy()
    frame["split"] = label_split(frame["year"])
    cols = NUMERIC + CATEGORICAL
    tr = frame[frame["split"] == "train"]
    te = frame[frame["split"] == "test"]
    return tr[cols].copy(), te[cols].copy(), tr[target].astype(int).copy(), te[target].astype(int).copy()


def split_frames(df: pd.DataFrame | None = None, target: str = TARGET_NEXT):
    """Return the frame with a ``split`` column and no missing target rows."""
    if df is None:
        df = add_next_day_target(load_raw())
    frame = df.dropna(subset=[target]).copy()
    frame["split"] = label_split(frame["year"])
    return frame


def load_vulnerability() -> pd.DataFrame:
    """The 10-row per-basin static table (area, population, resilience, damage)."""
    return pd.read_csv(VULN_CSV)


def load_geojson() -> dict:
    """The basin polygons (10 features). Not station points."""
    import json

    with open(GEOJSON, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# Figure helpers - enforce rule G4 (numbered, titled, labelled axes, legend)
# --------------------------------------------------------------------------------------
def savefig(fig, name: str, *, tight: bool = True, dpi: int = 150) -> Path:
    """Save ``fig`` to ``figures/<name>.png`` and return the path."""
    if not name.endswith(".png"):
        name = f"{name}.png"
    path = FIGDIR / name
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] {path.relative_to(REPO_ROOT)}")
    return path


def finish(ax, *, title: str, xlabel: str, ylabel: str, legend: bool = True):
    """Apply the mandatory figure furniture to an Axes."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True)


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------
def configure_env() -> None:
    """Pin BLAS/OMP/joblib to one thread.

    The DSH file sandbox forbids named pipes, so any joblib ``Parallel`` pool (used by
    ``HistGradientBoosting*`` and anything with ``n_jobs > 1``) dies with
    ``PermissionError: [WinError 5]``. Must run before numpy/sklearn spin up threads.
    """
    for var in ("OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def class_balance(y: pd.Series) -> pd.DataFrame:
    """Counts, shares and imbalance ratio for a binary label."""
    vc = y.value_counts().sort_index()
    return pd.DataFrame(
        {
            "count": vc,
            "share_pct": (vc / len(y) * 100).round(4),
        }
    ).assign(imbalance_ratio=lambda d: round(d["count"].max() / d["count"].min(), 2))


if __name__ == "__main__":
    configure_env()
    df = load_raw()
    print("rows x cols:", df.shape)
    print("stations:", df.station_id.nunique(), "| days:", df.date.nunique())
    print("missing:", int(df.isna().sum().sum()))
    d = add_next_day_target(df)
    usable = d.dropna(subset=[TARGET_NEXT])
    print("panel rows:", len(d), "| usable rows (target present):", len(usable))
    # NOTE: dropna FIRST. Grouping the full frame would report the year-window size
    # (73,050 for test) while the mean already ignores the missing target - the two
    # numbers would then describe different row sets.
    print(usable.groupby(label_split(usable["year"]))[TARGET_NEXT].agg(["size", "sum", "mean"]).round(6))
    print("\nNUMERIC:", NUMERIC)
    print("CATEGORICAL:", CATEGORICAL)
    print("DROPPED:", len(DROPPED))
