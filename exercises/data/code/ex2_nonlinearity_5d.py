"""Exercise 2 -- Non-linearity in Higher Dimensions.

Two synthetic 5D, two-class datasets are built to contrast a setting where
linear structure dominates against one where it fundamentally cannot
separate the classes:

  * Dataset I  -- two correlated multivariate Gaussian blobs (classes A, B),
    500 points each, drawn directly with rng.multivariate_normal(mean, cov,
    size=500) from the specified means/covariances (Part A).
  * Dataset II -- two concentric spherical shells (classes C, D), 500 points
    each. Every point is rho * u: a uniformly random unit direction u on the
    5D sphere (v = rng.standard_normal(5), u = v / ||v||) scaled by a
    class-specific radius rho ~ N(radius_mean, 0.4) (Part B).

Both datasets are then reduced to 2D with a separately-fit PCA, and compared
via centroid distance and per-class radius ||x|| in the original 5D space
(Part C), to show that a near-zero centroid distance combined with cleanly
separated radius histograms means the classes sit on concentric shells, not
offset blobs -- a shape no hyperplane can cut, no matter how much data is
collected (Part D).

Run standalone:
    python ex2_nonlinearity_5d.py
from this file's own directory. Figures are written to ../figures/.

Reproducibility: a single rng = np.random.default_rng(42) is created once
and consumed in exactly this order -- Dataset I: class A then class B (one
rng.multivariate_normal call each); Dataset II: for each of class C then
class D, one rng.standard_normal(500, 5) direction draw followed by one
rng.normal(...) radius draw. PCA uses svd_solver="full" (exact LAPACK SVD,
not the randomized default), so it never touches rng -- all randomness in
the pipeline is confined to data generation above.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_PER_CLASS = 500
DIM = 5

# Validated, colorblind-safe categorical palette -- fixed roles, never
# reassigned or cycled.
COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
COLOR_C, COLOR_D = "#1baf7a", "#4a3aa7"

# Single, exclusive source of randomness for the whole script.
rng = np.random.default_rng(42)

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.color": "#000000",
        "grid.alpha": 0.08,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "font.size": 10.5,
    }
)

# ---------------------------------------------------------------------------
# Part A -- Dataset I: shifted correlated Gaussians (classes A, B)
# ---------------------------------------------------------------------------

mu_A = np.zeros(DIM)
Sigma_A = np.array(
    [
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0],
    ]
)

mu_B = np.full(DIM, 1.5)
Sigma_B = np.array(
    [
        [1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7, 1.5, 0.4, 0.0, 0.0],
        [0.2, 0.4, 1.5, 0.6, 0.0],
        [0.0, 0.0, 0.6, 1.5, 0.3],
        [0.0, 0.0, 0.0, 0.3, 1.5],
    ]
)

# Deterministic draw order: class A before class B.
X_A = rng.multivariate_normal(mu_A, Sigma_A, size=N_PER_CLASS)
X_B = rng.multivariate_normal(mu_B, Sigma_B, size=N_PER_CLASS)

X_I = np.vstack([X_A, X_B])
y_I = np.array(["A"] * N_PER_CLASS + ["B"] * N_PER_CLASS)

# ---------------------------------------------------------------------------
# Part B -- Dataset II: concentric shells (classes C, D)
# ---------------------------------------------------------------------------


def sample_shell(n: int, radius_mean: float, radius_std: float) -> np.ndarray:
    """n points uniform in direction on the 5D unit sphere, scaled by a
    Gaussian radius rho ~ N(radius_mean, radius_std). One direction draw
    (n x DIM) followed by one radius draw (n,), in that order."""
    v = rng.standard_normal((n, DIM))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    rho = rng.normal(radius_mean, radius_std, size=n)
    return rho[:, None] * u


# Deterministic draw order: core (C) before shell (D).
X_C = sample_shell(N_PER_CLASS, radius_mean=2.0, radius_std=0.4)
X_D = sample_shell(N_PER_CLASS, radius_mean=5.0, radius_std=0.4)

X_II = np.vstack([X_C, X_D])
y_II = np.array(["C"] * N_PER_CLASS + ["D"] * N_PER_CLASS)

# ---------------------------------------------------------------------------
# Part C.1 -- PCA(2), fit separately per dataset (svd_solver="full": exact,
# never touches rng)
# ---------------------------------------------------------------------------

pca_I = PCA(n_components=2, svd_solver="full")
proj_I = pca_I.fit_transform(X_I)
evr_I = pca_I.explained_variance_ratio_

pca_II = PCA(n_components=2, svd_solver="full")
proj_II = pca_II.fit_transform(X_II)
evr_II = pca_II.explained_variance_ratio_

# ---------------------------------------------------------------------------
# Part C.3 -- 5D geometric measures: centroid distance + per-class radius
# ---------------------------------------------------------------------------

centroid_A, centroid_B = X_A.mean(axis=0), X_B.mean(axis=0)
centroid_C, centroid_D = X_C.mean(axis=0), X_D.mean(axis=0)
centroid_dist_I = float(np.linalg.norm(centroid_A - centroid_B))
centroid_dist_II = float(np.linalg.norm(centroid_C - centroid_D))

radius_A, radius_B = np.linalg.norm(X_A, axis=1), np.linalg.norm(X_B, axis=1)
radius_C, radius_D = np.linalg.norm(X_C, axis=1), np.linalg.norm(X_D, axis=1)
mean_radius_A, mean_radius_B = radius_A.mean(), radius_B.mean()
mean_radius_C, mean_radius_D = radius_C.mean(), radius_D.mean()

# Explicit separating function for Dataset II, f(x) = ||x||^2, threshold set
# at the midpoint of the two class mean radii (Part D.3).
radius_threshold = (mean_radius_C + mean_radius_D) / 2.0
sq_radius_threshold = radius_threshold**2
pred_core = np.concatenate([radius_C, radius_D]) < radius_threshold
true_core = np.array([True] * N_PER_CLASS + [False] * N_PER_CLASS)
threshold_accuracy = float(np.mean(pred_core == true_core))

# ---------------------------------------------------------------------------
# Sanity checks -- fail loudly if the reproducible pipeline breaks
# ---------------------------------------------------------------------------

assert X_I.shape == (2 * N_PER_CLASS, DIM) and X_II.shape == (2 * N_PER_CLASS, DIM)
assert abs(centroid_dist_I - 3.2282) < 1e-3
assert abs(centroid_dist_II - 0.2662) < 1e-3
assert abs(evr_I.sum() - 0.6597) < 1e-3
assert abs(evr_II.sum() - 0.4291) < 1e-3
assert threshold_accuracy > 0.95  # simple ||x||^2 threshold should near-perfectly separate C/D

# ---------------------------------------------------------------------------
# Printed report -- every number the write-up needs also appears here
# ---------------------------------------------------------------------------

print("=== Dataset I: shifted correlated Gaussians (classes A / B) ===")
print(f"Empirical centroid A: {np.array2string(centroid_A, precision=4)}")
print(f"Empirical centroid B: {np.array2string(centroid_B, precision=4)}")
print(f"Distance between class centroids ||mu_A - mu_B||: {centroid_dist_I:.4f}")
print(f"PCA explained variance -- PC1: {evr_I[0]:.4f}, PC2: {evr_I[1]:.4f}, sum: {evr_I.sum():.4f}")
print(f"Mean radius ||x|| -- class A: {mean_radius_A:.4f}, class B: {mean_radius_B:.4f}")

print("\n=== Dataset II: concentric shells (classes C / D) ===")
print(f"Empirical centroid C: {np.array2string(centroid_C, precision=4)}")
print(f"Empirical centroid D: {np.array2string(centroid_D, precision=4)}")
print(f"Distance between class centroids ||mu_C - mu_D||: {centroid_dist_II:.4f}")
print(f"PCA explained variance -- PC1: {evr_II[0]:.4f}, PC2: {evr_II[1]:.4f}, sum: {evr_II.sum():.4f}")
print(f"Mean radius ||x|| -- class C (core): {mean_radius_C:.4f} (std {radius_C.std():.4f})")
print(f"Mean radius ||x|| -- class D (shell): {mean_radius_D:.4f} (std {radius_D.std():.4f})")
print(f"Radius gap (mean D - mean C): {mean_radius_D - mean_radius_C:.4f} vs. centroid distance: {centroid_dist_II:.4f}")
print(
    f"Separating function f(x) = ||x||^2, threshold T = {sq_radius_threshold:.2f} "
    f"(||x|| < {radius_threshold:.4f} -> core/C): {threshold_accuracy * 100:.2f}% accuracy on Dataset II"
)

# ---------------------------------------------------------------------------
# Figure 4 -- PCA(2) projections, side by side
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

ax = axes[0]
for label, color in [("A", COLOR_A), ("B", COLOR_B)]:
    mask = y_I == label
    ax.scatter(
        proj_I[mask, 0], proj_I[mask, 1], s=18, alpha=0.7, color=color,
        edgecolors="white", linewidths=0.2, label=f"Class {label}",
    )
ax.set_title("Dataset I: PCA(2) of shifted Gaussians", fontweight="bold")
ax.set_xlabel(f"PC1 ({evr_I[0] * 100:.1f}% var.)")
ax.set_ylabel(f"PC2 ({evr_I[1] * 100:.1f}% var.)")
ax.legend(title="Class", frameon=False, loc="best")

ax = axes[1]
for label, color in [("C", COLOR_C), ("D", COLOR_D)]:
    mask = y_II == label
    ax.scatter(
        proj_II[mask, 0], proj_II[mask, 1], s=18, alpha=0.7, color=color,
        edgecolors="white", linewidths=0.2, label=f"Class {label}",
    )
ax.set_title("Dataset II: PCA(2) of concentric shells", fontweight="bold")
ax.set_xlabel(f"PC1 ({evr_II[0] * 100:.1f}% var.)")
ax.set_ylabel(f"PC2 ({evr_II[1] * 100:.1f}% var.)")
ax.legend(title="Class", frameon=False, loc="best")
ax.set_aspect("equal", adjustable="datalim")

fig.suptitle("Figure 4 -- PCA(2) Projections: Dataset I vs. Dataset II (fit separately)", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG_DIR / "fig4.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5 -- per-class radius histograms, one panel per dataset, overlaid
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

ax = axes[0]
bins_I = np.linspace(min(radius_A.min(), radius_B.min()), max(radius_A.max(), radius_B.max()), 36)
ax.hist(radius_A, bins=bins_I, alpha=0.55, color=COLOR_A, label="Class A", edgecolor="white", linewidth=0.4)
ax.hist(radius_B, bins=bins_I, alpha=0.55, color=COLOR_B, label="Class B", edgecolor="white", linewidth=0.4)
ax.set_title("Dataset I: radius $\\|x\\|$ per class", fontweight="bold")
ax.set_xlabel(r"$\|x\|$ (Euclidean norm)")
ax.set_ylabel("Count")
ax.legend(title="Class", frameon=False, loc="best")

ax = axes[1]
bins_II = np.linspace(min(radius_C.min(), radius_D.min()), max(radius_C.max(), radius_D.max()), 36)
ax.hist(radius_C, bins=bins_II, alpha=0.55, color=COLOR_C, label="Class C (core)", edgecolor="white", linewidth=0.4)
ax.hist(radius_D, bins=bins_II, alpha=0.55, color=COLOR_D, label="Class D (shell)", edgecolor="white", linewidth=0.4)
ax.axvline(radius_threshold, color="black", linestyle="--", linewidth=1.3, zorder=4)
ax.annotate(
    f"threshold\n||x|| = {radius_threshold:.2f}",
    xy=(radius_threshold, 0.97), xycoords=("data", "axes fraction"),
    xytext=(radius_threshold + 0.15, 0.97), textcoords=("data", "axes fraction"),
    fontsize=8.5, va="top", ha="left",
)
ax.set_title("Dataset II: radius $\\|x\\|$ per class", fontweight="bold")
ax.set_xlabel(r"$\|x\|$ (Euclidean norm)")
ax.set_ylabel("Count")
ax.legend(title="Class", frameon=False, loc="upper left")

fig.suptitle("Figure 5 -- Per-class Radius Histograms: Dataset I vs. Dataset II", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG_DIR / "fig5.png", dpi=150)
plt.close(fig)

print(f"\nSaved: {FIG_DIR / 'fig4.png'}")
print(f"Saved: {FIG_DIR / 'fig5.png'}")
