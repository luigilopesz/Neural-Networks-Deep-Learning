"""Exercise 1 -- Point Clouds: Geometry and Spread in 2D.

Four synthetic 2D classes (100 points each, 400 total) are drawn as Gaussians
with fixed per-class means and per-axis standard deviations (Part A), then
regenerated four times over with every standard deviation scaled by a factor
s in {0.5, 1.0, 2.0, 4.0} while the means stay fixed (Part B), to see how
spread alone changes class separability.

Separation is measured two ways, both purely geometric -- no model is fit or
trained anywhere in this script:

  * Analytic ratio  r_ij = ||mu_i - mu_j|| / (sigmabar_i + sigmabar_j),
    computed once from the class parameters at s = 1.0 (does not touch rng).
  * Nearest-center mixing rate -- for every point, the fraction whose
    closest of the 4 true CLASS CENTERS (means, not other data points;
    Euclidean, vectorized point-to-center distance matrix) is not its own
    class's center. Purely geometric, nothing trained.

Figure 1's sketched boundary is a separate, independent construction: the
piecewise-linear (Voronoi) partition of the plane by nearest TRUE class
MEAN, the simplest straight-line rule a trained network could settle near
given these four fixed centers. It is not the rule behind the mixing rate
above.

Run standalone:
    python ex1_point_clouds.py
from this file's own directory. Figures are written to ../figures/.

Reproducibility: a single rng = np.random.default_rng(42) is created once and
consumed in exactly this order -- Part A's original dataset (classes 0..3 at
s = 1.0), then Part B's four scaled datasets in order s = 0.5, 1.0, 2.0, 4.0
(classes 0..3 each). Nothing else in the script draws from rng.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_PER_CLASS = 100
N_CLASSES = 4
SCALES = [0.5, 1.0, 2.0, 4.0]

CLASS_MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
CLASS_STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])

# Validated, colorblind-safe categorical palette -- fixed roles, never reassigned.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
CLASS_LABELS = [f"Class {k}" for k in range(N_CLASSES)]

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


def generate_dataset(scale: float, rng: np.random.Generator):
    """Draw 100 points per class from N(mean_k, (scale * std_k)^2), class
    order 0..3, via one rng.normal(mean, std, size=(100, 2)) call per class.

    Returns (X, y): X shape (400, 2), y shape (400,) integer labels.
    """
    X = np.empty((N_CLASSES * N_PER_CLASS, 2))
    y = np.empty(N_CLASSES * N_PER_CLASS, dtype=int)
    for k in range(N_CLASSES):
        sl = slice(k * N_PER_CLASS, (k + 1) * N_PER_CLASS)
        X[sl] = rng.normal(CLASS_MEANS[k], CLASS_STDS[k] * scale, size=(N_PER_CLASS, 2))
        y[sl] = k
    return X, y


def nearest_center_mixing_rate(X: np.ndarray, y: np.ndarray, means: np.ndarray) -> float:
    """Fraction of points whose nearest of the 4 true CLASS CENTERS (means,
    NOT nearest other data point) is not the center of their own class.
    Fully vectorized point-to-center distance matrix; purely geometric --
    nothing fit, nothing trained."""
    diff = X[:, None, :] - means[None, :, :]
    dist2 = np.sum(diff**2, axis=-1)
    nearest_center = np.argmin(dist2, axis=1)
    return float(np.mean(nearest_center != y))


# ---------------------------------------------------------------------------
# Part A -- generate the original clouds (s = 1.0), rng draws 1-4
# ---------------------------------------------------------------------------

X1, y1 = generate_dataset(scale=1.0, rng=rng)

# ---------------------------------------------------------------------------
# Part B.1 -- regenerate at 4 scales, rng draws 5-20 (same rng, no reseed)
# ---------------------------------------------------------------------------

datasets = {s: generate_dataset(scale=s, rng=rng) for s in SCALES}

# ---------------------------------------------------------------------------
# Part B.2 -- pairwise separation ratios r_ij at s = 1.0 (analytic, no rng)
# ---------------------------------------------------------------------------

sigma_bar = CLASS_STDS.mean(axis=1)  # sigmabar_k, shape (4,)
pairs = list(combinations(range(N_CLASSES), 2))
r_ij = {}
for i, j in pairs:
    dist_ij = np.linalg.norm(CLASS_MEANS[i] - CLASS_MEANS[j])
    r_ij[(i, j)] = dist_ij / (sigma_bar[i] + sigma_bar[j])

smallest_pair = min(r_ij, key=r_ij.get)
smallest_r = r_ij[smallest_pair]
# r_ij scales as 1/s (means fixed, sigmabar_i+sigmabar_j scales with s) --
# pure arithmetic on the s=1.0 value, nothing regenerated.
smallest_r_at_s2 = smallest_r / 2.0

# ---------------------------------------------------------------------------
# Part B.3 -- nearest-class-center mixing rate per scale
# ---------------------------------------------------------------------------

mixing_rates = {s: nearest_center_mixing_rate(*datasets[s], CLASS_MEANS) for s in SCALES}

# ---------------------------------------------------------------------------
# Sanity checks -- fail loudly if the reproducible pipeline breaks
# ---------------------------------------------------------------------------

assert X1.shape == (400, 2) and y1.shape == (400,)
for s in SCALES:
    X_s, y_s = datasets[s]
    assert X_s.shape == (400, 2) and y_s.shape == (400,)
    assert np.bincount(y_s, minlength=4).tolist() == [100, 100, 100, 100]
    assert 0.0 <= mixing_rates[s] <= 1.0
assert len(r_ij) == 6
assert smallest_pair == (0, 1)
assert abs(smallest_r - 1.325825) < 1e-4
assert abs(mixing_rates[0.5] - 0.0000) < 1e-9
assert abs(mixing_rates[1.0] - 0.0675) < 1e-9
assert abs(mixing_rates[2.0] - 0.2250) < 1e-9
assert abs(mixing_rates[4.0] - 0.4175) < 1e-9

# ---------------------------------------------------------------------------
# Printed report -- every number the write-up needs also appears here
# ---------------------------------------------------------------------------

print("=== Part A: class parameters ===")
for k in range(N_CLASSES):
    print(
        f"Class {k}: mean={CLASS_MEANS[k].tolist()}, std={CLASS_STDS[k].tolist()}, "
        f"sigma_bar={sigma_bar[k]:.4f}"
    )

print("\n=== Part B.2: pairwise separation ratios r_ij at s = 1.0 ===")
for (i, j), r in r_ij.items():
    print(f"r_{i}{j} = {r:.4f}")
print(
    f"Smallest: r_{smallest_pair[0]}{smallest_pair[1]} = {smallest_r:.4f} "
    f"(class {smallest_pair[0]} vs class {smallest_pair[1]})"
)
print(
    f"By pure arithmetic (r_ij scales as 1/s), at s=2.0 this becomes "
    f"r_{smallest_pair[0]}{smallest_pair[1]} = {smallest_r_at_s2:.4f}"
)

print("\n=== Part B.3: nearest-class-center mixing rate per scale ===")
for s in SCALES:
    n_wrong = int(round(mixing_rates[s] * N_CLASSES * N_PER_CLASS))
    print(f"s={s}: mixing rate = {mixing_rates[s]:.4f} ({n_wrong}/400 points)")

# ---------------------------------------------------------------------------
# Figure 1 -- original clouds (s = 1.0), means marked, decision boundary sketch
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6.5))

for k in range(N_CLASSES):
    mask = y1 == k
    ax.scatter(
        X1[mask, 0], X1[mask, 1], s=26, alpha=0.75, color=CLASS_COLORS[k],
        edgecolors="white", linewidths=0.3, label=CLASS_LABELS[k], zorder=3,
    )
    ax.scatter(
        CLASS_MEANS[k, 0], CLASS_MEANS[k, 1], marker="X", s=240,
        color=CLASS_COLORS[k], edgecolors="black", linewidths=1.4, zorder=5,
    )

xlim = (X1[:, 0].min() - 1.0, X1[:, 0].max() + 1.0)
ylim = (X1[:, 1].min() - 1.0, X1[:, 1].max() + 1.0)

# Part C.2: sketch a decision boundary directly onto the figure -- a fine
# grid classified by "nearest TRUE class center" (Voronoi partition of the 4
# fixed means), contoured to trace the straight-line boundary a
# piecewise-linear classifier -- the simplest thing a trained network could
# approximate -- would draw between the 4 classes. Same nearest-center rule
# as the Part B.3 mixing rate above, applied here to every point of a dense
# grid instead of just the 400 sampled points.
xx, yy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
grid_dist2 = np.sum((grid[:, None, :] - CLASS_MEANS[None, :, :]) ** 2, axis=-1)
zz = np.argmin(grid_dist2, axis=1).reshape(xx.shape)
for k in range(N_CLASSES):
    ax.contour(
        xx, yy, (zz == k).astype(float), levels=[0.5],
        colors="black", linewidths=1.4, linestyles="--", zorder=4,
    )
ax.plot([], [], color="black", linestyle="--", linewidth=1.4, label="Sketched decision boundary")

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_title("Figure 1 -- Original four Gaussian classes (s = 1.0)", fontweight="bold")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend(title="Class ('X' = true mean)", frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 -- four scales, 2x2 grid, shared axis limits
# ---------------------------------------------------------------------------

all_x = np.concatenate([datasets[s][0][:, 0] for s in SCALES])
all_y = np.concatenate([datasets[s][0][:, 1] for s in SCALES])
margin_x = 0.05 * (all_x.max() - all_x.min())
margin_y = 0.05 * (all_y.max() - all_y.min())
shared_xlim = (all_x.min() - margin_x, all_x.max() + margin_x)
shared_ylim = (all_y.min() - margin_y, all_y.max() + margin_y)

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), SCALES):
    X_s, y_s = datasets[s]
    for k in range(N_CLASSES):
        mask = y_s == k
        ax.scatter(
            X_s[mask, 0], X_s[mask, 1], s=16, alpha=0.7, color=CLASS_COLORS[k],
            edgecolors="white", linewidths=0.2, label=CLASS_LABELS[k],
        )
    ax.set_xlim(shared_xlim)
    ax.set_ylim(shared_ylim)
    ax.set_title(f"s = {s}")

for ax in axes[-1, :]:
    ax.set_xlabel("Feature 1")
for ax in axes[:, 0]:
    ax.set_ylabel("Feature 2")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title="Class", loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Figure 2 -- Same 4 classes regenerated at scale s (shared axes)", fontweight="bold")
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig(FIG_DIR / "fig2.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3 -- mixing rate vs. scale factor
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 5.2))
rates = [mixing_rates[s] for s in SCALES]
ax.plot(SCALES, rates, marker="o", markersize=9, linewidth=2, color="#2a78d6", zorder=3, label="Mixing rate")
for s, r in zip(SCALES, rates):
    ax.annotate(f"{r:.4f}", (s, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

# Selective direct label: the scale factor identified in Part B.4 as where
# straight-line (nearest-centroid) separation is decisively lost.
ax.annotate(
    "linear separability\nlost from here (r_01 < 1)",
    xy=(2.0, mixing_rates[2.0]), xytext=(1.05, 0.36),
    fontsize=9, ha="left", color="#4a3aa7",
    arrowprops=dict(arrowstyle="->", color="#4a3aa7", linewidth=1.2),
)

ax.set_xlim(0.42, 4.9)
ax.set_title("Figure 3 -- Nearest-center mixing rate vs. spread scale", fontweight="bold")
ax.set_xlabel("Scale factor s (std multiplier)")
ax.set_ylabel("Mixing rate (fraction with cross-class nearest center)")
ax.set_xscale("log", base=2)
ax.set_xticks(SCALES)
ax.set_xticklabels([str(s) for s in SCALES])
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98))

fig.tight_layout()
fig.savefig(FIG_DIR / "fig3.png", dpi=150)
plt.close(fig)

print(f"\nSaved: {FIG_DIR / 'fig1.png'}")
print(f"Saved: {FIG_DIR / 'fig2.png'}")
print(f"Saved: {FIG_DIR / 'fig3.png'}")
