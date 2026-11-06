"""T6 - stage 4B (dimensionality reduction): evidence generator.

Produces every number quoted in ``sections/s4b-reduction.md`` and the seven stage-4B
figures (``fig50_*`` - ``fig53_*``).  **No model is trained.**  PCA is an exact linear
decomposition of the scaled training matrix; t-SNE and UMAP are nonlinear 2-D embeddings
of a stratified sample of it.  The label is never predicted - it is only ever used to
colour the projections.

    python t6_reduction.py        # cwd = docs/projects/eda/code

Contract
--------
* Features come from ``pipeline.fit_preprocessor(X_train)`` + ``pipeline.transform``,
  fitted on the **training window only** (2000-2016, 310,500 rows).  Validation and test
  are never touched by this script.
* PCA is fitted on the **full** training set (310,500 x 21).
* t-SNE / UMAP run on a **stratified 30,000-row sample**: all 6,566 training positives plus
  a random draw of 23,434 negatives (``random_state=42``).  The sample's positive rate is
  21.887 %, not the population's 2.166 % - see the sampling note in the report.
* ``random_state=42`` / ``SEED`` everywhere; ``n_jobs=1`` for UMAP (the sandbox forbids the
  named pipes a joblib ``Parallel`` pool opens).
"""

from __future__ import annotations

import os
import time

# Must precede the numpy / numba imports: these pin every thread pool to one thread.
# NUMBA_NUM_THREADS is what actually keeps UMAP deterministic and single-threaded.
for _var in (
    "OMP_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pipeline as pl
from data import (
    SEED,
    TARGET_NEXT,
    add_next_day_target,
    configure_env,
    finish,
    load_raw,
    savefig,
    train_test_frames,
)

# matplotlib is switch to the headless Agg backend by ``data``; import pyplot after it.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import umap  # noqa: E402  (umap-learn 0.5.12 - required by the statement)

configure_env()

# --------------------------------------------------------------------------------------
# Design constants
# --------------------------------------------------------------------------------------
SAMPLE_N: int = 30_000          # stratified t-SNE / UMAP sample size
PERPLEXITIES: tuple[int, ...] = (5, 30, 50)
N_NEIGHBORS: tuple[int, ...] = (5, 15, 50)
BEST_PERPLEXITY: int = 30       # balanced default -> the panel in the comparison figure
BEST_N_NEIGHBORS: int = 15      # balanced default -> the panel in the comparison figure
K_LOCAL: int = 25               # neighbourhood diagnostic: k nearest neighbours in 2-D
N_NEG_PLOT: int = 40_000        # negatives drawn in the full-training-set PCA scatter

NEG_COLOUR, POS_COLOUR = "#9AA5B1", "#C44E52"
PCA_COLOUR = "#8172B2"

TARGET_LEGEND = [
    Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=NEG_COLOUR,
           markeredgecolor="none", label="no flood on day t+1 (negative)"),
    Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=POS_COLOUR,
           markeredgecolor="none", label="flood on day t+1 (positive)"),
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def table(rows: list[tuple], header: tuple) -> None:
    """Print a fixed-width table so the fragment can cite it verbatim."""
    widths = [max(len(str(header[i])), *(len(str(r[i])) for r in rows)) for i in range(len(header))]
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def draw_classes(ax, coords, y, *, xlabel, ylabel, title, s_neg=3.0, s_pos=9.0,
                 alpha_neg=0.15, alpha_pos=0.80):
    """Scatter ``coords`` coloured by the target: negatives first, positives on top.

    Drawing the 21.9 %-positive *sample* in this order is what makes a class that is
    2.166 % of the population visible at all.
    """
    neg = y == 0
    ax.scatter(coords[neg, 0], coords[neg, 1], s=s_neg, c=NEG_COLOUR, alpha=alpha_neg,
               linewidths=0)
    ax.scatter(coords[~neg, 0], coords[~neg, 1], s=s_pos, c=POS_COLOUR, alpha=alpha_pos,
               linewidths=0)
    finish(ax, title=title, xlabel=xlabel, ylabel=ylabel, legend=False)


def local_stats(coords: np.ndarray, y: np.ndarray, k: int) -> dict[str, float]:
    """Label composition of each point's ``k`` nearest neighbours **in the embedding**.

    ``scipy.spatial.cKDTree`` is a geometric index, not a fitted estimator: this measures
    what the projection preserved, it does not predict anything.
    """
    dist, ind = cKDTree(coords).query(coords, k=k + 1)
    neigh_rate = y[ind[:, 1:]].mean(axis=1)          # drop self
    span = float(np.hypot(*(coords.max(axis=0) - coords.min(axis=0))))
    return {
        "pos": float(neigh_rate[y == 1].mean()),
        "neg": float(neigh_rate[y == 0].mean()),
        "global": float(y.mean()),
        "radius_pos": float(np.median(dist[y == 1][:, -1])),
        "radius_neg": float(np.median(dist[y == 0][:, -1])),
        "span": span,
    }


# ======================================================================================
print("=" * 88)
print("T6 - STAGE 4B  DIMENSIONALITY REDUCTION")
print("=" * 88)
T_SCRIPT = time.perf_counter()

# --------------------------------------------------------------------------------------
print("\n[1] FEATURE CONTRACT - the scaled training matrix")
raw = add_next_day_target(load_raw())
X_train, X_test, y_train, y_test = train_test_frames(raw)

pre = pl.fit_preprocessor(X_train)                 # fit on TRAIN ONLY
Z = pl.transform(pre, X_train)
NAMES = pl.feature_names(pre)
y_all = y_train.to_numpy().astype(int)

print(f"    raw panel            : {raw.shape[0]:,} rows x {raw.shape[1] - 1} cols (+{TARGET_NEXT})")
print(f"    X_train (2000-2016)  : {X_train.shape}  {len(pl.NUMERIC_RAW)} numeric + "
      f"{len(pl.CATEGORICAL_RAW)} categorical raw columns")
print(f"    X_test  (2021-2024)  : {X_test.shape}  - never transformed or fitted here")
print(f"    Z = transform(pre, X_train) : {Z.shape}  {Z.dtype}")
print(f"    np.isnan(Z).sum()    : {int(np.isnan(Z).sum())}")
print(f"    training positives   : {int(y_all.sum()):,} of {len(y_all):,} "
      f"({y_all.mean() * 100:.4f} %, imbalance {int((y_all == 0).sum()) / int(y_all.sum()):.1f} : 1)")
print(f"    features produced    : {len(NAMES)} (the pipeline, not this script, decides that)")
for i, name in enumerate(NAMES, start=1):
    print(f"      {i:>2}. {name}")

# --------------------------------------------------------------------------------------
print("\n[2] PCA - fitted on the FULL training set")
t0 = time.perf_counter()
pca = PCA(n_components=min(Z.shape), svd_solver="full", random_state=SEED)
SCORES = pca.fit_transform(Z)                      # (310500, 21) - exact full SVD
pca_seconds = time.perf_counter() - t0
EVR = pca.explained_variance_ratio_
CUM = np.cumsum(EVR)
pc12 = float(CUM[1] * 100)

print(f"    fitted on            : {Z.shape[0]:,} x {Z.shape[1]}  (all training rows, no sampling)")
print(f"    runtime              : {pca_seconds:.1f} s  (svd_solver='full', deterministic)")
rows = [(f"PC{i + 1}", f"{EVR[i] * 100:.4f}", f"{CUM[i] * 100:.4f}") for i in range(min(10, len(EVR)))]
table(rows, ("component", "variance %", "cumulative %"))
print(f"    PC11-PC{len(EVR)} together : {(CUM[-1] - CUM[9]) * 100:.4f} %  "
      f"(PC{len(EVR)} alone {EVR[-1] * 100:.4f} %)")
n90 = int(np.searchsorted(CUM, 0.90) + 1)
n95 = int(np.searchsorted(CUM, 0.95) + 1)
print(f"    components for 90 %  : {n90}   for 95 %: {n95}   (of {len(EVR)})")
print()
print(f"    *** PC1 + PC2 CUMULATIVE EXPLAINED VARIANCE = {pc12:.4f} % ***")
print(f"    *** (results-table row 9; PC1 = {EVR[0] * 100:.4f} %, PC2 = {EVR[1] * 100:.4f} %) ***")

print("\n    loadings - the 8 largest |coefficient| of PC1 and of PC2 (scaled-feature units)")
for comp in (0, 1):
    load = pd.Series(pca.components_[comp], index=NAMES)
    order = load.abs().sort_values(ascending=False).index
    print(f"    PC{comp + 1}:")
    for name in order[:8]:
        print(f"        {name:<34} {load[name]:+.4f}")
    share = {g: float((load[[n for n in NAMES if n.startswith(p)]] ** 2).sum())
             for g, p in (("numeric", "num__"), ("basin one-hot", "cat__basin_"),
                          ("season one-hot", "cat__season_"))}
    total = sum(share.values())
    print("        squared-loading share: "
          + ", ".join(f"{g} {v / total * 100:.1f} %" for g, v in share.items()))

print("\n    how far apart are the classes on PC1 / PC2?  (310,500 training rows)")
for comp in (0, 1):
    s = SCORES[:, comp]
    m0, m1 = float(s[y_all == 0].mean()), float(s[y_all == 1].mean())
    q99 = float(np.quantile(s[y_all == 0], 0.99))
    above = float((s[y_all == 1] > q99).mean() * 100)
    print(f"    PC{comp + 1}: negatives mean {m0:+.4f} (sd {s[y_all == 0].std():.4f}) | "
          f"positives mean {m1:+.4f} (sd {s[y_all == 1].std():.4f}) | shift {m1 - m0:+.4f} "
          f"| {above:.2f} % of positives beyond the negatives' 99th pct "
          f"({100 - above:.2f} % inside it)")

# --------------------------------------------------------------------------------------
print(f"\n[3] STRATIFIED {SAMPLE_N:,}-ROW SAMPLE for t-SNE / UMAP")
t0 = time.perf_counter()
pos_idx = np.flatnonzero(y_all == 1)
neg_idx = np.flatnonzero(y_all == 0)
rng = np.random.default_rng(SEED)                  # random_state=42
neg_draw = np.sort(rng.choice(neg_idx, size=SAMPLE_N - len(pos_idx), replace=False))
neg_plot = np.sort(rng.choice(neg_idx, size=N_NEG_PLOT, replace=False))
idx = np.sort(np.concatenate([pos_idx, neg_draw]))
S = np.ascontiguousarray(Z[idx])
y_s = y_all[idx]
sample_seconds = time.perf_counter() - t0

print(f"    training rows        : {len(y_all):,}  ({len(pos_idx):,} positives, {len(neg_idx):,} negatives)")
print(f"    positives retained   : {len(pos_idx):,} of {len(pos_idx):,}  (100.000 % - every training positive)")
print(f"    negatives drawn      : {len(neg_draw):,} of {len(neg_idx):,}  "
      f"({len(neg_draw) / len(neg_idx) * 100:.2f} %, np.random.default_rng({SEED}).choice without replacement)")
print(f"    S = Z[idx]           : {S.shape}  {S.dtype}")
print(f"    sample positive rate : {y_s.mean() * 100:.4f} %  ({int(y_s.sum()):,} positives) "
      f"- vs {y_all.mean() * 100:.4f} % in the training set "
      f"({y_s.mean() / y_all.mean():.2f}x - the sample is stratified, not representative)")
print("    Figure 4B.3 draws every positive but only "
      f"{len(neg_plot):,} negatives ({len(neg_plot) / len(neg_idx) * 100:.2f} %), so the red:grey "
      f"density ratio there overstates the class ratio by {len(neg_idx) / len(neg_plot):.2f}x")
print(f"    runtime              : {sample_seconds:.2f} s")

# --------------------------------------------------------------------------------------
print("\n[4] t-SNE SWEEP - 30,000 rows, random_state=42, n_jobs=1")
TSNE_EMB: dict[int, np.ndarray] = {}
TSNE_SECONDS: dict[int, float] = {}
for perp in PERPLEXITIES:
    model = TSNE(n_components=2, perplexity=perp, learning_rate="auto", init="pca",
                 max_iter=1000, method="barnes_hut", angle=0.5, random_state=SEED, n_jobs=1)
    t0 = time.perf_counter()
    emb = np.asarray(model.fit_transform(S), dtype=float)
    secs = time.perf_counter() - t0
    TSNE_EMB[perp], TSNE_SECONDS[perp] = emb, secs
    print(f"    perplexity={perp:<3} runtime {secs:7.1f} s | KL divergence {model.kl_divergence_:.4f} "
          f"| axis 1 [{emb[:, 0].min():+.2f}, {emb[:, 0].max():+.2f}] "
          f"| axis 2 [{emb[:, 1].min():+.2f}, {emb[:, 1].max():+.2f}] "
          f"| sum {float(np.abs(emb).sum()):.6f}")

# --------------------------------------------------------------------------------------
print("\n[5] UMAP SWEEP - 30,000 rows, random_state=42, n_jobs=1")
t0 = time.perf_counter()
umap.UMAP(n_components=2, n_neighbors=BEST_N_NEIGHBORS, random_state=SEED,
          n_jobs=1).fit_transform(S[:500])
jit_seconds = time.perf_counter() - t0
print(f"    numba JIT warm-up on 500 rows: {jit_seconds:.1f} s (excluded from the timings below)")

UMAP_EMB: dict[int, np.ndarray] = {}
UMAP_SECONDS: dict[int, float] = {}
for nn in N_NEIGHBORS:
    model = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.1, metric="euclidean",
                      random_state=SEED, n_jobs=1, verbose=False)
    t0 = time.perf_counter()
    emb = np.asarray(model.fit_transform(S), dtype=float)
    secs = time.perf_counter() - t0
    UMAP_EMB[nn], UMAP_SECONDS[nn] = emb, secs
    print(f"    n_neighbors={nn:<3} runtime {secs:7.1f} s | axis 1 [{emb[:, 0].min():+.2f}, "
          f"{emb[:, 0].max():+.2f}] | axis 2 [{emb[:, 1].min():+.2f}, {emb[:, 1].max():+.2f}] "
          f"| sum {float(np.abs(emb).sum()):.6f}")

# --------------------------------------------------------------------------------------
print(f"\n[6] NEIGHBOURHOOD DIAGNOSTIC - label composition of the {K_LOCAL} nearest neighbours in 2-D")
EMBEDDINGS: list[tuple[str, np.ndarray]] = [("PCA (PC1-PC2)", SCORES[idx][:, :2])]
EMBEDDINGS += [(f"t-SNE perplexity={p}", TSNE_EMB[p]) for p in PERPLEXITIES]
EMBEDDINGS += [(f"UMAP n_neighbors={n}", UMAP_EMB[n]) for n in N_NEIGHBORS]

t0 = time.perf_counter()
LOCAL = {label: local_stats(coords, y_s, K_LOCAL) for label, coords in EMBEDDINGS}
diag_seconds = time.perf_counter() - t0

rows = []
for label, _ in EMBEDDINGS:
    st = LOCAL[label]
    rows.append((label, f"{st['pos'] * 100:.2f}", f"{st['neg'] * 100:.2f}",
                 f"{st['pos'] / st['neg']:.2f}",
                 f"{st['radius_pos'] / st['span'] * 100:.3f}",
                 f"{st['radius_neg'] / st['span'] * 100:.3f}"))
table(rows, ("embedding", "pos kNN pos %", "neg kNN pos %", "ratio",
             "25-NN radius pos", "25-NN radius neg"))
print(f"    sample positive rate (reference) : {y_s.mean() * 100:.4f} %")
print(f"    runtime                          : {diag_seconds:.2f} s")
print("    reading: 'ratio' is how much more positive a positive point's neighbourhood is")
print("    than a negative point's. 1.00 = the label is invisible in the projection.")

# --------------------------------------------------------------------------------------
print("\n[7] FIGURES")

# --- Figure 4B.1: scree + cumulative --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))
k = len(EVR)
ax = axes[0]
ax.bar(np.arange(1, k + 1), EVR * 100, color=PCA_COLOUR, edgecolor="white")
for i in range(min(10, k)):
    ax.text(i + 1, EVR[i] * 100 + 0.35, f"{EVR[i] * 100:.2f}", ha="center", va="bottom",
            fontsize=7, rotation=90)
ax.set_ylim(0, float(EVR[:10].max() * 100) * 1.35)
finish(ax, title="(a) Scree - variance carried by each principal component",
       xlabel="principal component index", ylabel="variance explained by that component (%)",
       legend=False)

ax = axes[1]
ax.plot(np.arange(1, k + 1), CUM * 100, marker="o", color=POS_COLOUR, label="cumulative variance")
ax.axhline(pc12, ls="--", lw=1.2, color="black", label=f"PC1+PC2 = {pc12:.2f} %")
ax.annotate(f"PC1+PC2 = {pc12:.2f} %", xy=(2, pc12), xytext=(4.0, pc12 - 13),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
ax.set_ylim(0, 102)
finish(ax, title="(b) Cumulative variance explained",
       xlabel="number of principal components kept",
       ylabel="cumulative variance explained (%)")
fig.suptitle("Figure 4B.1 - PCA on the scaled training features "
             f"({Z.shape[0]:,} rows x {Z.shape[1]} features)", y=1.02, fontsize=13)
savefig(fig, "fig50_pca_variance")

# --- Figure 4B.2: loadings heatmap ----------------------------------------------------
N_SHOW = 8
LOAD = pca.components_[:N_SHOW]
fig, ax = plt.subplots(figsize=(15.0, 6.4))
vmax = float(np.abs(LOAD).max())
im = ax.imshow(LOAD, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
for i in range(N_SHOW):
    for j in range(len(NAMES)):
        ax.text(j, i, f"{LOAD[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(LOAD[i, j]) > 0.55 * vmax else "black")
ax.set_xticks(range(len(NAMES)))
ax.set_xticklabels(NAMES, rotation=90, fontsize=8)
ax.set_yticks(range(N_SHOW))
ax.set_yticklabels([f"PC{i + 1} ({EVR[i] * 100:.2f} %)" for i in range(N_SHOW)])
finish(ax, title=f"Figure 4B.2 - PCA loadings: the first {N_SHOW} components against the "
                 f"{len(NAMES)} scaled training features",
       xlabel="scaled training feature (pipeline output column)",
       ylabel="principal component (share of variance in brackets)", legend=False)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
             label="loading = correlation of the feature with the component")
savefig(fig, "fig50_pca_loadings")

# --- Figure 4B.3: PC1-PC2 coloured by the target (full training set) -------------------
pc1 = SCORES[:, 0]
q99_pc1 = float(np.quantile(pc1[y_all == 0], 0.99))
inside = float((pc1[y_all == 1] <= q99_pc1).mean() * 100)

fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.4),
                         gridspec_kw={"width_ratios": [1.45, 1.0]})
ax = axes[0]
ax.scatter(SCORES[neg_plot, 0], SCORES[neg_plot, 1], s=3, c=NEG_COLOUR, alpha=0.12,
           linewidths=0, label=f"no flood on day t+1 - {len(neg_plot):,} of "
                               f"{len(neg_idx):,} negatives drawn")
ax.scatter(SCORES[pos_idx, 0], SCORES[pos_idx, 1], s=9, c=POS_COLOUR, alpha=0.75,
           linewidths=0, label=f"flood on day t+1 - all {len(pos_idx):,} positives")
ax.scatter(*SCORES[y_all == 0][:, :2].mean(axis=0), marker="X", s=220, c=NEG_COLOUR,
           edgecolors="black", linewidths=1.4, zorder=5, label="class centroid (negatives)")
ax.scatter(*SCORES[y_all == 1][:, :2].mean(axis=0), marker="X", s=220, c=POS_COLOUR,
           edgecolors="black", linewidths=1.4, zorder=5, label="class centroid (positives)")
finish(ax, title="(a) PC1-PC2 scatter, coloured by the target",
       xlabel=f"PC1 ({EVR[0] * 100:.2f} % of variance, component 1)",
       ylabel=f"PC2 ({EVR[1] * 100:.2f} % of variance, component 2)")
ax.legend(loc="upper left", fontsize=9)
ax.text(0.98, 0.02, f"every positive drawn, but only {len(neg_plot) / len(neg_idx) * 100:.1f} % "
                    f"of the negatives:\nthe red:grey ratio here overstates the true class "
                    f"ratio by {len(neg_idx) / len(neg_plot):.1f}x",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

ax = axes[1]
bins = np.linspace(float(pc1.min()), float(pc1.max()), 121)
ax.hist(pc1[y_all == 0], bins=bins, histtype="step", lw=1.6, color=NEG_COLOUR,
        label=f"negatives, all {len(neg_idx):,}")
ax.hist(pc1[y_all == 1], bins=bins, histtype="step", lw=1.6, color=POS_COLOUR,
        label=f"positives, all {len(pos_idx):,}")
ax.axvline(q99_pc1, ls="--", lw=1.3, color="black",
           label=f"negatives' 99th percentile (PC1 = {q99_pc1:.2f})")
ax.set_yscale("log")
finish(ax, title="(b) PC1 distribution of each class (all 310,500 training rows)",
       xlabel=f"PC1 ({EVR[0] * 100:.2f} % of variance, component 1)",
       ylabel="training rows per bin (count, log scale)")
ax.legend(loc="upper right", fontsize=9)
ax.text(0.97, 0.50, f"{inside:.2f} % of positives\nlie left of the dashed line",
        transform=ax.transAxes, ha="right", va="center", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
fig.suptitle("Figure 4B.3 - PCA of the scaled training features, coloured by the next-day target",
             y=1.02, fontsize=13)
savefig(fig, "fig50_pca_scatter")

# --- Figure 4B.4 / 4B.5: parameter sweeps ---------------------------------------------
def sweep_figure(embeddings, runtimes, values, param, name, fig_number, title):
    """One panel per parameter value, all panels sharing one target legend."""
    axis_label = "t-SNE" if param == "perplexity" else "UMAP"
    fig, axes = plt.subplots(1, len(values), figsize=(6.0 * len(values), 5.8))
    for ax, value in zip(np.atleast_1d(axes), values):
        draw_classes(ax, embeddings[value], y_s,
                     xlabel=f"{axis_label} 1 (arbitrary units)",
                     ylabel=f"{axis_label} 2 (arbitrary units)",
                     # NOTE: runtimes are printed to stdout, never into the panel titles -
                     # a wall-clock string in the image makes the committed PNG unreproducible.
                     title=f"{param} = {value}")
    fig.legend(handles=TARGET_LEGEND, loc="lower center", ncol=2, frameon=True,
               bbox_to_anchor=(0.5, -0.035))
    fig.suptitle(f"Figure 4B.{fig_number} - {title}", y=1.02, fontsize=13)
    fig.tight_layout()
    savefig(fig, name)


sweep_figure(TSNE_EMB, TSNE_SECONDS, PERPLEXITIES, "perplexity",
             "fig51_tsne_perplexity", 4,
             f"t-SNE on the {SAMPLE_N:,}-row stratified sample "
             f"({int(y_s.sum()):,} positives), coloured by the next-day target")
sweep_figure(UMAP_EMB, UMAP_SECONDS, N_NEIGHBORS, "n_neighbors",
             "fig52_umap_nneighbors", 5,
             f"UMAP on the {SAMPLE_N:,}-row stratified sample "
             f"({int(y_s.sum()):,} positives), coloured by the next-day target")

# --- Figure 4B.6: PCA vs best t-SNE vs best UMAP, same rows ---------------------------
panels = [
    ("PCA (PC1-PC2)", "PCA - PC1 vs PC2", SCORES[idx][:, :2],
     f"PC1 ({EVR[0] * 100:.2f} % of variance, component 1)",
     f"PC2 ({EVR[1] * 100:.2f} % of variance, component 2)"),
    (f"t-SNE perplexity={BEST_PERPLEXITY}",
     f"t-SNE - perplexity = {BEST_PERPLEXITY}",
     TSNE_EMB[BEST_PERPLEXITY], "t-SNE 1 (arbitrary units)", "t-SNE 2 (arbitrary units)"),
    (f"UMAP n_neighbors={BEST_N_NEIGHBORS}",
     f"UMAP - n_neighbors = {BEST_N_NEIGHBORS}",
     UMAP_EMB[BEST_N_NEIGHBORS], "UMAP 1 (arbitrary units)", "UMAP 2 (arbitrary units)"),
]
fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.0))
for ax, (key, title, coords, xlabel, ylabel) in zip(axes, panels):
    draw_classes(ax, coords, y_s, xlabel=xlabel, ylabel=ylabel, title=title,
                 s_neg=2.5, s_pos=7.0)
    st = LOCAL[key]
    ax.text(0.02, 0.98, f"k={K_LOCAL} kNN positive rate\npositives {st['pos'] * 100:.1f} %  |  "
                        f"negatives {st['neg'] * 100:.1f} %",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
fig.legend(handles=TARGET_LEGEND, loc="lower center", ncol=2, frameon=True,
           bbox_to_anchor=(0.5, -0.035))
fig.suptitle(f"Figure 4B.6 - the same {SAMPLE_N:,} training rows under all three projections, "
             "coloured by the next-day target", y=1.02, fontsize=13)
fig.tight_layout()
fig.text(0.5, -0.055, f"stratified sample: all {int(y_s.sum()):,} training positives + "
                      f"{len(neg_draw):,} random negatives = {y_s.mean() * 100:.2f} % positive "
                      f"({y_s.mean() / y_all.mean():.1f}x the {y_all.mean() * 100:.2f} % training "
                      "rate) - the red density is inflated by construction; the three panels are "
                      "comparable with each other, not with the population",
         ha="center", fontsize=9)
savefig(fig, "fig53_comparison")

# --- Figure 4B.7: what the projection preserves ---------------------------------------
labels = [label for label, _ in EMBEDDINGS]
fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4))
xpos = np.arange(len(labels))
ax = axes[0]
ax.bar(xpos - 0.2, [LOCAL[l]["pos"] * 100 for l in labels], width=0.4, color=POS_COLOUR,
       label=f"neighbourhood of a positive point (n={int(y_s.sum()):,})")
ax.bar(xpos + 0.2, [LOCAL[l]["neg"] * 100 for l in labels], width=0.4, color=NEG_COLOUR,
       label=f"neighbourhood of a negative point (n={int((y_s == 0).sum()):,})")
ax.axhline(y_s.mean() * 100, ls="--", lw=1.2, color="black",
           label=f"sample positive rate {y_s.mean() * 100:.2f} % (the no-structure line)")
for i, l in enumerate(labels):
    ax.text(i, max(LOCAL[l]["pos"], LOCAL[l]["neg"]) * 100 + 0.5,
            f"{LOCAL[l]['pos'] / LOCAL[l]['neg']:.2f}x", ha="center", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_ylim(0, 85)  # headroom so the legend cannot cover the 7 ratio labels (bars peak ~51)
finish(ax, title=f"(a) Label composition of the {K_LOCAL} nearest neighbours in each 2-D embedding",
       xlabel="2-D projection (same 30,000 rows)", ylabel="positive share of the neighbourhood (%)")
ax.legend(fontsize=8, loc="upper left")

ax = axes[1]
ax.bar(xpos - 0.2, [LOCAL[l]["radius_pos"] / LOCAL[l]["span"] * 100 for l in labels],
       width=0.4, color=POS_COLOUR, label="positive points")
ax.bar(xpos + 0.2, [LOCAL[l]["radius_neg"] / LOCAL[l]["span"] * 100 for l in labels],
       width=0.4, color=NEG_COLOUR, label="negative points")
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_ylim(0, 1.12)
finish(ax, title="(b) Median distance to the 25th neighbour, as a share of the embedding "
                 "diagonal",
       xlabel="2-D projection (same 30,000 rows)",
       ylabel="median 25-NN radius (% of the embedding's diagonal extent)")
ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Figure 4B.7 - what each projection actually preserves: local label structure "
             "and local scale", y=1.02, fontsize=13)
fig.tight_layout()
savefig(fig, "fig53_local_enrichment")

# --------------------------------------------------------------------------------------
print("\n[8] RUNTIME BUDGET (wall clock, 1 thread)")
rows = [("PCA, full training set, all components", f"{pca_seconds:.1f} s"),
        ("stratified sample construction", f"{sample_seconds:.2f} s")]
rows += [(f"t-SNE, perplexity={p}", f"{TSNE_SECONDS[p]:.1f} s") for p in PERPLEXITIES]
rows += [(f"UMAP, n_neighbors={n}", f"{UMAP_SECONDS[n]:.1f} s") for n in N_NEIGHBORS]
rows += [("neighbourhood diagnostic, 7 embeddings", f"{diag_seconds:.2f} s"),
         ("UMAP numba JIT warm-up (one-off, 500 rows)", f"{jit_seconds:.1f} s"),
         ("whole script", f"{time.perf_counter() - T_SCRIPT:.1f} s")]
table(rows, ("step", "wall clock"))
print(f"    t-SNE sweep total    : {sum(TSNE_SECONDS.values()):.1f} s for {len(PERPLEXITIES)} values")
print(f"    UMAP sweep total     : {sum(UMAP_SECONDS.values()):.1f} s for {len(N_NEIGHBORS)} values")
print("\n" + "=" * 88)
print("T6 DONE - no model was trained; PCA, t-SNE and UMAP are projections only.")
print("=" * 88)
