"""Dump the numbers behind every figure into charts.json for the web page.

No statistics are recomputed here: ex1/ex2 are executed with runpy and their
globals are read straight out, so the interactive charts and the .png figures
are guaranteed to show the same data. Exercise 3's pipeline lives inside
main(), so the two columns Figure 6 needs (FoodCourt before/after log1p on the
train split) are rebuilt with the same seed, split and imputer.

    python export_charts.py
writes ../charts.json.
"""

import json
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "charts.json"


def r(a, nd=3):
    """Round for transport: JSON gets 3 decimals, not 17."""
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def hist(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return {"centers": r(centers), "counts": counts.tolist(), "width": float(edges[1] - edges[0])}


# --- Exercise 1 -------------------------------------------------------------
g1 = runpy.run_path(str(HERE / "ex1_point_clouds.py"))
means, scales = g1["CLASS_MEANS"], g1["SCALES"]

# Voronoi boundary: same dense-grid nearest-mean partition the figure draws,
# contoured once and read back as polylines instead of pixels.
X1 = g1["X1"]
xlim = (X1[:, 0].min() - 1.0, X1[:, 0].max() + 1.0)
ylim = (X1[:, 1].min() - 1.0, X1[:, 1].max() + 1.0)
xx, yy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
zz = np.argmin(np.sum((grid[:, None, :] - means[None, :, :]) ** 2, axis=-1), axis=1).reshape(xx.shape)
segments = []
_fig, _ax = plt.subplots()
for k in range(4):
    cs = _ax.contour(xx, yy, (zz == k).astype(float), levels=[0.5])
    segments += [r(seg) for seg in cs.allsegs[0] if len(seg) > 1]
plt.close(_fig)

ex1 = {
    "means": r(means),
    "stds": r(g1["CLASS_STDS"]),
    "colors": g1["CLASS_COLORS"],
    "scales": scales,
    "partA": r(np.column_stack([X1, g1["y1"]])),
    "points": {str(s): r(np.column_stack([g1["datasets"][s][0], g1["datasets"][s][1]])) for s in scales},
    "boundary": segments,
    "mixing": {str(s): round(g1["mixing_rates"][s], 4) for s in scales},
    "r_ij": {f"{i}{j}": round(v, 4) for (i, j), v in g1["r_ij"].items()},
    "r01": round(g1["smallest_r"], 4),
}

# --- Exercise 2 -------------------------------------------------------------
g2 = runpy.run_path(str(HERE / "ex2_nonlinearity_5d.py"))
ex2 = {
    "colors": {"A": g2["COLOR_A"], "B": g2["COLOR_B"], "C": g2["COLOR_C"], "D": g2["COLOR_D"]},
    "pcaI": r(g2["proj_I"]),
    "pcaII": r(g2["proj_II"]),
    "evrI": r(g2["evr_I"], 4),
    "evrII": r(g2["evr_II"], 4),
    "radius": {k: r(g2[f"radius_{k}"]) for k in "ABCD"},
    "threshold": round(float(g2["radius_threshold"]), 4),
    "centroidDist": {"I": round(float(g2["centroid_dist_I"]), 4), "II": round(float(g2["centroid_dist_II"]), 4)},
}

# --- Exercise 3 -------------------------------------------------------------
df = pd.read_csv(HERE / "data" / "train.csv")
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC = ["Age"] + SPEND

train_df, _ = train_test_split(df, test_size=0.2, stratify=df["Transported"], random_state=42)
imputed = SimpleImputer(strategy="median").fit_transform(train_df[NUMERIC])
fc_before = imputed[:, NUMERIC.index("FoodCourt")]
fc_after = np.log1p(fc_before)

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
ex3 = {
    "colors": {"before": g1["CLASS_COLORS"][0], "after": g1["CLASS_COLORS"][1]},
    "missing": {"cols": missing.index.tolist(), "pct": r(missing / len(df) * 100, 2)},
    "spend": {
        "cols": SPEND,
        "mean": r([df[c].mean() for c in SPEND], 2),
        "median": r([df[c].median() for c in SPEND], 2),
        "max": r([df[c].max() for c in SPEND], 2),
    },
    "foodcourtBefore": hist(fc_before, 40),
    "foodcourtAfter": hist(fc_after, 40),
    "foodcourtStats": {"mean": round(float(fc_before.mean()), 2), "median": round(float(np.median(fc_before)), 2)},
}

OUT.write_text(json.dumps({"ex1": ex1, "ex2": ex2, "ex3": ex3}, separators=(",", ":")), encoding="utf-8")

# The page's headline numbers must survive this round-trip unchanged.
assert ex1["mixing"]["2.0"] == 0.225 and ex1["r01"] == 1.3258
assert len(ex1["partA"]) == 400
assert len(ex1["boundary"]) >= 3, "Voronoi boundary polylines missing"
assert abs(sum(ex2["evrI"]) - 0.6597) < 1e-3 and abs(sum(ex2["evrII"]) - 0.4291) < 1e-3
assert ex3["foodcourtStats"]["median"] == 0.0 and ex3["foodcourtStats"]["mean"] > 400
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
