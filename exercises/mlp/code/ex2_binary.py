"""Exercise 2 -- Binary Classification a Straight Line Cannot Do.

Class 0 is one Gaussian blob at the origin (500 points); Class 1 is two blobs on
opposite sides of it, at (3, 3) and (-3, -3) (250 points each) -- a shape no
single hyperplane can separate. `make_blobs` assigns blob index `c in {0,1,2}`;
`y = (c > 0).astype(int)` folds the two Class-1 blobs into one label. A linear
baseline (`LogisticRegression`) is fit first to show it fails below chance, then
the shared, from-scratch `MLP` (`mlp.py`, unmodified) is trained on the same
split with one hidden layer of 16 tanh units and a sigmoid output.

Run standalone:
    python ex2_binary.py
from this file's own directory. Figures are written to ../figures/.

Reproducibility: `rng = np.random.default_rng(42)` is created once and consumed
exactly once, by `MLP.__init__`'s Xavier weight init (training is full-batch, so
`fit` never draws from it). Data generation and the train/test split use their
own `random_state=42`, as the assignment statement writes them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlp import MLP

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Validated, colorblind-safe categorical pair -- fixed roles, never reassigned.
COLOR_0, COLOR_1 = "#2a78d6", "#eb6834"
LIGHT_0, LIGHT_1 = "#cfe3fa", "#fbdbc4"  # pastel tints of the same pair, for region shading

# Single, exclusive source of randomness for the whole script (MLP weight init only).
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
# Part A -- data generation
# ---------------------------------------------------------------------------

CENTERS = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
N_SAMPLES = [500, 250, 250]
CLUSTER_STD = 1.2

X, c = make_blobs(
    n_samples=N_SAMPLES, centers=CENTERS, cluster_std=CLUSTER_STD, random_state=42
)
y = (c > 0).astype(int)  # class 0: 1 cluster (blob 0) | class 1: 2 clusters (blobs 1, 2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# Part B -- linear baseline
# ---------------------------------------------------------------------------

baseline = LogisticRegression().fit(X_train, y_train)
baseline_acc = baseline.score(X_test, y_test)

# ---------------------------------------------------------------------------
# Part C -- MLP from scratch
# ---------------------------------------------------------------------------

LAYER_SIZES = [2, 16, 1]
LR = 0.1
EPOCHS = 2000

model = MLP(LAYER_SIZES, output="sigmoid", rng=rng)
losses = model.fit(X_train, y_train[:, None], epochs=EPOCHS, lr=LR)

pred_test = model.predict(X_test)
mlp_acc = float(np.mean(pred_test == y_test))
n_misclassified = int(np.sum(pred_test != y_test))

# ---------------------------------------------------------------------------
# Sanity checks -- fail loudly if the reproducible pipeline breaks
# ---------------------------------------------------------------------------

assert X.shape == (1000, 2) and y.shape == (1000,)
assert np.bincount(y).tolist() == [500, 500]
assert mlp_acc >= 0.85, f"MLP test accuracy regressed: {mlp_acc:.4f} < 0.85"
assert baseline_acc <= 0.60, f"linear baseline no longer below chance: {baseline_acc:.4f}"

# ---------------------------------------------------------------------------
# Printed report -- every number the write-up needs also appears here
# ---------------------------------------------------------------------------

print("=== Part A: data ===")
print(f"centers = {CENTERS.tolist()}")
print(f"n_samples = {N_SAMPLES}, cluster_std = {CLUSTER_STD}, random_state=42")
print(f"X.shape = {X.shape}, class counts = {np.bincount(y).tolist()}")
print(f"train/test split: {X_train.shape[0]}/{X_test.shape[0]} (80/20, stratified)")

print("\n=== Part B: linear baseline (LogisticRegression) ===")
print(f"test accuracy = {baseline_acc:.4f}")

print("\n=== Part C: MLP from scratch ===")
print(f"architecture = {LAYER_SIZES}  (hidden: tanh, output: sigmoid)")
print(f"learning rate = {LR}, epochs = {EPOCHS}, batch = full")
print(f"n_params = {model.n_params()}")
print(f"final training loss = {losses[-1]:.6f}")
print(f"test accuracy = {mlp_acc:.4f} ({n_misclassified}/{X_test.shape[0]} misclassified)")

print("\n=== Baseline vs. MLP ===")
print(f"{'model':<22}{'test accuracy':>15}")
print(f"{'logistic regression':<22}{baseline_acc:>15.4f}")
print(f"{'MLP [2, 16, 1]':<22}{mlp_acc:>15.4f}")

# ---------------------------------------------------------------------------
# Figure 1 -- scatter of all 1000 points, one color per class
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=16, alpha=0.65, color=COLOR_0, edgecolors="white",
           linewidths=0.2, label="Class 0", zorder=3)
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, alpha=0.65, color=COLOR_1, edgecolors="white",
           linewidths=0.2, label="Class 1", zorder=3)
ax.set_title("Figure 1 -- Binary data: Class 0 vs. Class 1", fontweight="bold")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 -- training loss vs. epoch
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 5.2))
epochs_axis = np.arange(1, len(losses) + 1)
ax.plot(epochs_axis, losses, color=COLOR_0, linewidth=1.6, label="Training loss (BCE)")
ax.set_title("Figure 2 -- MLP training loss vs. epoch", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Binary cross-entropy loss")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3 -- decision regions over the test points
# ---------------------------------------------------------------------------

margin = 1.5
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
GRID_N = 350
xx, yy = np.meshgrid(np.linspace(x_min, x_max, GRID_N), np.linspace(y_min, y_max, GRID_N))
grid_points = np.column_stack([xx.ravel(), yy.ravel()])
grid_proba = model.predict_proba(grid_points).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.contourf(xx, yy, grid_proba, levels=[0.0, 0.5, 1.0], colors=[LIGHT_0, LIGHT_1],
            alpha=0.75, zorder=1)
ax.contour(xx, yy, grid_proba, levels=[0.5], colors="black", linewidths=1.5, zorder=2)

correct = pred_test == y_test
for k, color in ((0, COLOR_0), (1, COLOR_1)):
    ok = (y_test == k) & correct
    ax.scatter(X_test[ok, 0], X_test[ok, 1], s=32, color=color, edgecolors="white",
               linewidths=0.3, marker="o", zorder=3, label=f"Class {k} (correct)")
    bad = (y_test == k) & ~correct
    if bad.any():
        ax.scatter(X_test[bad, 0], X_test[bad, 1], s=70, color=color, marker="x",
                   linewidths=2.0, zorder=4, label=f"Class {k} (misclassified)")

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title("Figure 3 -- MLP decision regions and test points", fontweight="bold")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3.png", dpi=150)
plt.close(fig)

print(f"\nSaved: {FIG_DIR / 'fig1.png'}")
print(f"Saved: {FIG_DIR / 'fig2.png'}")
print(f"Saved: {FIG_DIR / 'fig3.png'}")
