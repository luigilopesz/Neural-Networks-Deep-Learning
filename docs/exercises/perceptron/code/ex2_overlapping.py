"""Exercise 2 -- Perceptron on Overlapping Data (Pocket Algorithm).

Two 2D Gaussian classes, 1000 points each (2000 total), with means close
together (‖mu1-mu0‖ = sqrt(2) ~= 1.41) relative to their spread (std ~= 1.22
/axis) -- heavily overlapping, NOT linearly separable. Imports the single
shared `Perceptron` from perceptron.py UNCHANGED (same class as Exercise 1,
just instantiated with pocket=True here) and trains at the spec's eta=0.01
with a 100-epoch cap.

Because perceptron.py overwrites `self.w`/`self.b` with the pocket-best
snapshot at the end of `fit` when pocket=True (see its docstring), a single
fit() call only ever exposes the pocket weights afterwards -- the last-epoch
("final") weights are gone. To report BOTH, as the assignment requires, this
script runs `fit` twice on the identical initial weights: once with
pocket=False (final-epoch w/b) and once with pocket=True (pocket-best w/b),
by snapshotting and restoring the shared rng's bit-generator state around the
pair of calls. Both calls execute the exact same unchanged update rule on the
exact same data with the exact same per-sample order, so the two runs are
byte-for-byte the same trajectory -- this is one training run viewed two
ways, not two independent trainings (verified below: their per-epoch
histories are asserted equal).

A second wrinkle: `X = vstack([X0, X1])` lays out all 1000 Class-0 points
before all 1000 Class-1 points, and `fit()`'s per-sample loop is a plain,
unshuffled `for i in range(n_samples)` -- it never reorders its input. Fed
that block layout directly, the online update degenerates into an
artificial 2-updates-per-epoch cycle: one correction late in the Class-0
block flips the (near-origin-scaled, all-positive-quadrant) boundary to favor
Class 0, then the very first Class-1 sample flips it straight back -- so the
epoch-end snapshot always favors whichever class was seen *last*, forever,
independent of how separable the data actually is. That is a data-*ordering*
artifact, not a property of the classes' overlap. The fix -- standard
practice for online/SGD-style learners -- is a single upfront shuffle of
(X, y) before the (unmodified) 100-epoch training loop runs; this is a data
presentation choice made in this script, not a change to perceptron.py's
class or its update rule.

Run standalone:
    python ex2_overlapping.py
from this file's own directory. Figures are written to ../figures/.

Reproducibility: a single rng = np.random.default_rng(42) is created once.
Draw order: Class 0 (Part A), Class 1 (Part A), the one-time shuffle
permutation (pre-Part B), then the shared weight init consumed identically
by both the pocket=False and pocket=True fit() calls (Part B) via a
bit-generator state snapshot/restore -- nothing else in the script draws
from rng.
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
MEAN_0, COV_0 = [3.0, 3.0], [[1.5, 0.0], [0.0, 1.5]]
MEAN_1, COV_1 = [4.0, 4.0], [[1.5, 0.0], [0.0, 1.5]]

# Same validated, colorblind-safe categorical pair as Exercise 1 -- fixed roles.
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
# Part B -- pocket training at eta=0.01, 100-epoch cap
# ---------------------------------------------------------------------------

# rng draw 3: one-time shuffle, breaking the Class-0-block/Class-1-block
# layout of vstack([X0, X1]) so the unmodified, unshuffling fit() loop sees
# an interleaved presentation order every one of its (unchanging) 100 epochs.
perm = rng.permutation(len(X))
X_train, y_train = X[perm], y[perm]

init_state = rng.bit_generator.state  # snapshot so both fits share one init (rng draw 4: weight init)

model_final = Perceptron(eta=0.01, max_epochs=100, pocket=False).fit(X_train, y_train, rng)

rng.bit_generator.state = init_state  # restore: pocket run replays the same trajectory

model_pocket = Perceptron(eta=0.01, max_epochs=100, pocket=True).fit(X_train, y_train, rng)

w_final, b_final = model_final.w, model_final.b
final_acc = model_final.history[-1]
preds_final = model_final.predict(X)
mis_final = preds_final != y

w_pocket, b_pocket = model_pocket.w, model_pocket.b
pocket_acc = model_pocket.best_accuracy
pocket_epoch = int(np.argmax(model_pocket.history)) + 1  # first epoch hitting the max
preds_pocket = model_pocket.predict(X)
mis_pocket = preds_pocket != y

# ---------------------------------------------------------------------------
# Sanity checks -- fail loudly if the reproducible pipeline breaks
# ---------------------------------------------------------------------------

assert X.shape == (2000, 2) and y.shape == (2000,)
assert np.bincount(y).tolist() == [1000, 1000]
assert model_final.history == model_pocket.history, (
    "final-only and pocket runs must be the identical trajectory (same init, same updates)"
)
assert not model_pocket.converged, "overlapping, non-separable data should never converge"
assert model_pocket.n_epochs == 100, "non-separable data should run the full 100-epoch cap"
assert 0.35 < final_acc < 0.65, "final-epoch accuracy expected near a coin flip (~50%)"
assert 0.65 < pocket_acc < 0.80, "pocket accuracy expected near the ~73% best-line ceiling"
assert pocket_acc > final_acc, "pocket must not do worse than the final-epoch snapshot"

# ---------------------------------------------------------------------------
# Printed report -- every number the write-up needs also appears here
# ---------------------------------------------------------------------------

print("=== Part A: data ===")
print(f"Class 0: n={N_PER_CLASS}, mean={MEAN_0}, cov={COV_0}")
print(f"Class 1: n={N_PER_CLASS}, mean={MEAN_1}, cov={COV_1}")

print("\n=== Part B: pocket training at eta=0.01, max_epochs=100 ===")
print(f"epochs run = {model_pocket.n_epochs} (converged={model_pocket.converged})")
print(f"final   w = [{w_final[0]:.4f}, {w_final[1]:.4f}], b = {b_final:.4f}, "
      f"accuracy = {final_acc:.4f} ({int(mis_final.sum())}/2000 misclassified)")
print(f"pocket  w = [{w_pocket[0]:.4f}, {w_pocket[1]:.4f}], b = {b_pocket:.4f}, "
      f"accuracy = {pocket_acc:.4f} ({int(mis_pocket.sum())}/2000 misclassified), "
      f"first reached at epoch {pocket_epoch}")

# ---------------------------------------------------------------------------
# Figure 4 -- scatter of all 2000 points, one color per class
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.scatter(X0[:, 0], X0[:, 1], s=14, alpha=0.55, color=COLOR_0, edgecolors="white",
           linewidths=0.2, label="Class 0", zorder=3)
ax.scatter(X1[:, 0], X1[:, 1], s=14, alpha=0.55, color=COLOR_1, edgecolors="white",
           linewidths=0.2, label="Class 1", zorder=3)
ax.set_title("Figure 4 -- Overlapping data: Class 0 vs. Class 1", fontweight="bold")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5 -- both decision boundaries + misclassified points (pocket model)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, alpha=0.45, color=COLOR_0, edgecolors="white",
           linewidths=0.2, label="Class 0", zorder=2)
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, alpha=0.45, color=COLOR_1, edgecolors="white",
           linewidths=0.2, label="Class 1", zorder=2)

xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
xs = np.array(xlim)


def boundary_ys(w, b):
    return -(w[0] * xs + b) / w[1]


ax.plot(xs, boundary_ys(w_final, b_final), color="black", linewidth=1.8, linestyle="-",
        zorder=4, label=f"Final boundary (acc={final_acc:.2%})")
ax.plot(xs, boundary_ys(w_pocket, b_pocket), color="black", linewidth=1.8, linestyle="--",
        zorder=4, label=f"Pocket boundary (acc={pocket_acc:.2%})")

# Marking the final model's ~36% misclassified points would blanket most of
# the cloud (uninformative); the pocket model's ~29% error is the meaningful,
# more legible one to highlight.
ax.scatter(X[mis_pocket, 0], X[mis_pocket, 1], s=55, facecolors="none",
           edgecolors="black", linewidths=1.3, marker="^", zorder=5,
           label=f"Misclassified by pocket (n={int(mis_pocket.sum())})")

ax.set_xlim(xlim)
ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
ax.set_title("Figure 5 -- Final vs. pocket decision boundary", fontweight="bold")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 6 -- current (final-so-far) vs. best-so-far (pocket) accuracy / epoch
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 5.2))
epochs = np.arange(1, len(model_pocket.history) + 1)
history = np.array(model_pocket.history)
pocket_curve = np.maximum.accumulate(history)  # reproduces the class's own >best_accuracy rule

ax.plot(epochs, history, color="#2a78d6", linewidth=1.0, linestyle="-",
        label="Current (final-so-far) accuracy")
ax.plot(epochs, pocket_curve, color="#eb6834", linewidth=1.6, linestyle="--",
        label="Best-so-far (pocket) accuracy")
ax.set_ylim(0.0, 1.02)
ax.set_xlim(0.5, len(history) + 0.5)
ax.set_title("Figure 6 -- Accuracy vs. epoch: current vs. pocket", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (full dataset)")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6.png", dpi=150)
plt.close(fig)

print(f"\nSaved: {FIG_DIR / 'fig4.png'}")
print(f"Saved: {FIG_DIR / 'fig5.png'}")
print(f"Saved: {FIG_DIR / 'fig6.png'}")
