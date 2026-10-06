"""Exercise 1 -- Perceptron on Separable Data.

Two 2D Gaussian classes, 1000 points each (2000 total), with means far apart
(‖mu1-mu0‖ ≈ 4.95) relative to their spread (std ≈ 0.707/axis) -- practically
linearly separable. Imports the single shared `Perceptron` from perceptron.py
(unchanged, no pocket) and trains it twice on the identical data: once at the
spec's standard eta=0.01 (Part C), once at eta=1.0 with nothing else changed
(Part D.2), to see how the learning rate does -- and does not -- affect the
outcome on data this easy.

Run standalone:
    python ex1_separable.py
from this file's own directory. Figures are written to ../figures/.

Reproducibility: a single rng = np.random.default_rng(42) is created once and
consumed in exactly this order -- Class 0 draw, Class 1 draw (Part A), then
the eta=0.01 weight init + training (Part C), then the eta=1.0 weight init +
training (Part D.2). Nothing else in the script draws from rng.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_PER_CLASS = 1000
MEAN_0, COV_0 = [1.5, 1.5], [[0.5, 0.0], [0.0, 0.5]]
MEAN_1, COV_1 = [5.0, 5.0], [[0.5, 0.0], [0.0, 0.5]]

# Validated, colorblind-safe categorical pair -- fixed roles, never reassigned.
COLOR_0, COLOR_1 = "#2a78d6", "#eb6834"

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
# Part A -- data generation, rng draws 1-2
# ---------------------------------------------------------------------------

X0 = rng.multivariate_normal(MEAN_0, COV_0, size=N_PER_CLASS)
X1 = rng.multivariate_normal(MEAN_1, COV_1, size=N_PER_CLASS)
X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)])

# ---------------------------------------------------------------------------
# Part C -- train at eta=0.01 (rng draw 3: weight init, consumed inside fit)
# ---------------------------------------------------------------------------

model = Perceptron(eta=0.01, max_epochs=100).fit(X, y, rng)
w, b = model.w, model.b
final_acc = model.history[-1]
preds = model.predict(X)
misclassified = preds != y

# ---------------------------------------------------------------------------
# Part D.2 -- re-train at eta=1.0, nothing else changed (rng draw 4: weight init)
# ---------------------------------------------------------------------------

model_hi = Perceptron(eta=1.0, max_epochs=100).fit(X, y, rng)
w_hi, b_hi = model_hi.w, model_hi.b
final_acc_hi = model_hi.history[-1]

w_dir = w / np.linalg.norm(w)
w_hi_dir = w_hi / np.linalg.norm(w_hi)
cos_sim = float(np.dot(w_dir, w_hi_dir))

# ---------------------------------------------------------------------------
# Sanity checks -- fail loudly if the reproducible pipeline breaks
# ---------------------------------------------------------------------------

assert X.shape == (2000, 2) and y.shape == (2000,)
assert np.bincount(y).tolist() == [1000, 1000]
assert model.converged, "separable data is expected to converge well under 100 epochs"
assert final_acc > 0.99, "near-100% accuracy expected on this well-separated data"
assert cos_sim > 0.99, "eta should not change the converged direction of w on separable data"

# ---------------------------------------------------------------------------
# Printed report -- every number the write-up needs also appears here
# ---------------------------------------------------------------------------

print("=== Part A: data ===")
print(f"Class 0: n={N_PER_CLASS}, mean={MEAN_0}, cov={COV_0}")
print(f"Class 1: n={N_PER_CLASS}, mean={MEAN_1}, cov={COV_1}")

print("\n=== Part C: training at eta=0.01 ===")
print(f"final w = [{w[0]:.4f}, {w[1]:.4f}]")
print(f"final b = {b:.4f}")
print(f"epochs run = {model.n_epochs} (converged={model.converged})")
print(f"final accuracy = {final_acc:.4f} ({int(misclassified.sum())}/2000 misclassified)")

print("\n=== Part D.2: re-training at eta=1.0 ===")
print(f"final w = [{w_hi[0]:.4f}, {w_hi[1]:.4f}]")
print(f"final b = {b_hi:.4f}")
print(f"epochs run = {model_hi.n_epochs} (converged={model_hi.converged})")
print(f"final accuracy = {final_acc_hi:.4f}")
print(f"w/||w|| at eta=0.01 = [{w_dir[0]:.4f}, {w_dir[1]:.4f}]")
print(f"w/||w|| at eta=1.0  = [{w_hi_dir[0]:.4f}, {w_hi_dir[1]:.4f}]")
print(f"cosine similarity between directions = {cos_sim:.6f}")

# ---------------------------------------------------------------------------
# Figure 1 -- scatter of all 2000 points, one color per class
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.scatter(X0[:, 0], X0[:, 1], s=14, alpha=0.65, color=COLOR_0, edgecolors="white",
           linewidths=0.2, label="Class 0", zorder=3)
ax.scatter(X1[:, 0], X1[:, 1], s=14, alpha=0.65, color=COLOR_1, edgecolors="white",
           linewidths=0.2, label="Class 1", zorder=3)
ax.set_title("Figure 1 -- Separable data: Class 0 vs. Class 1", fontweight="bold")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 -- decision boundary + misclassified points (eta=0.01 model)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, alpha=0.6, color=COLOR_0, edgecolors="white",
           linewidths=0.2, label="Class 0", zorder=2)
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, alpha=0.6, color=COLOR_1, edgecolors="white",
           linewidths=0.2, label="Class 1", zorder=2)

xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
if abs(w[1]) > 1e-12:
    xs = np.array(xlim)
    ys = -(w[0] * xs + b) / w[1]
    ax.plot(xs, ys, color="black", linewidth=1.8, zorder=4, label="Decision boundary  w·x+b=0")
else:
    x_line = -b / w[0]
    ax.axvline(x_line, color="black", linewidth=1.8, zorder=4, label="Decision boundary  w·x+b=0")

if misclassified.any():
    ax.scatter(X[misclassified, 0], X[misclassified, 1], s=110, facecolors="none",
               edgecolors="black", linewidths=1.6, marker="o", zorder=5,
               label=f"Misclassified (n={int(misclassified.sum())})")

ax.set_xlim(xlim)
ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
ax.set_title("Figure 2 -- Decision boundary after training (eta=0.01)", fontweight="bold")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3 -- accuracy vs. epoch (eta=0.01 model)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 5.2))
epochs = np.arange(1, len(model.history) + 1)
ax.plot(epochs, model.history, color="#2a78d6", linewidth=1.6, marker="o", markersize=4,
        label="Training accuracy (eta=0.01)")
ax.set_ylim(0.0, 1.02)
ax.set_xlim(0.5, len(model.history) + 0.5)
ax.set_xticks(epochs)
ax.set_title("Figure 3 -- Accuracy vs. epoch (eta=0.01)", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (full dataset)")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3.png", dpi=150)
plt.close(fig)

print(f"\nSaved: {FIG_DIR / 'fig1.png'}")
print(f"Saved: {FIG_DIR / 'fig2.png'}")
print(f"Saved: {FIG_DIR / 'fig3.png'}")
