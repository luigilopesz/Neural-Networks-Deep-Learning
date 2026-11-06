"""Exercise 2 -- Perceptron on Overlapping Data (Pocket Algorithm).

Two 2D Gaussian classes, 1000 points each (2000 total), with means close
together (||mu1-mu0|| = sqrt(2) ~= 1.41) relative to their spread (std ~= 1.22
/axis) -- heavily overlapping, NOT linearly separable. Imports the single
shared `Perceptron` from perceptron.py UNCHANGED (the very file and the very
call used in Exercise 1: eta=0.01, 100-epoch cap) and trains it ONCE, on the
data in the order it was generated (all 1000 Class-0 points, then all 1000
Class-1 points), exactly as in Exercise 1 -- no shuffling.

One `fit` returns both sets of weights the assignment asks for:
  * final  -- `model.w, model.b`: whatever the loop holds after the last epoch;
  * pocket -- `model.pocket_w, model.pocket_b`: the best-so-far, copied every
    time an UPDATE produces a higher full-dataset accuracy than any seen before
    (the check runs after each update, as the assignment defines it, not once
    per epoch).

Part D needs a few extra numbers (step sizes, where each boundary sits relative
to the cloud, the accuracy ceiling of a straight line); they are computed here
from the data and the two weight sets so the write-up quotes only printed values.

Run standalone:
    python ex2_overlapping.py
from this file's own directory. Figures are written to ../figures/.

Reproducibility: a single rng = np.random.default_rng(42) is created once.
Draw order: Class 0 (Part A), Class 1 (Part A), then the weight init inside
fit() (Part B). Nothing else in the script draws from rng.
"""

import math
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
ETA = 0.01

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
# Part B -- ONE training run, eta=0.01, 100-epoch cap (rng draw 3: weight init)
# ---------------------------------------------------------------------------

model = Perceptron(eta=ETA, max_epochs=100).fit(X, y, rng)

w_final, b_final = model.w, model.b
final_acc = model.history[-1]
mis_final = model.predict(X) != y

w_pocket, b_pocket = model.pocket_w, model.pocket_b
pocket_acc, pocket_epoch = model.pocket_acc, model.pocket_epoch
mis_pocket = model.predict(X, w_pocket, b_pocket) != y

# ---------------------------------------------------------------------------
# Part D numbers -- where the boundaries sit and why the loop leaves them there
# ---------------------------------------------------------------------------

sigma = math.sqrt(COV_0[0][0])  # per-axis std of both classes
centre = X.mean(axis=0)  # centre of the whole cloud
norms = np.linalg.norm(X, axis=1)  # ||x|| of every point
step_w = ETA * norms.mean()  # how far ONE mistake moves w (b moves ETA)
score_shift = ETA * (norms**2 + 1).mean()  # change of w.x+b at the mistaken point: eta*(||x||^2 + 1)


def line_stats(w, b):
    """||w||, distance of the line from the origin, signed distance of the cloud
    centre from the line (+ = Class-1 side), share of points predicted Class 1."""
    nw = float(np.linalg.norm(w))
    return nw, abs(b) / nw, float((w @ centre + b) / nw), float(np.mean(X @ w + b >= 0))


nw_f, off_f, cdist_f, p1_f = line_stats(w_final, b_final)
nw_p, off_p, cdist_p, p1_p = line_stats(w_pocket, b_pocket)
pred_f, pred_p = model.predict(X), model.predict(X, w_pocket, b_pocket)
fp_f, fn_f = int(((pred_f == 1) & (y == 0)).sum()), int(((pred_f == 0) & (y == 1)).sum())
fp_p, fn_p = int(((pred_p == 1) & (y == 0)).sum()), int(((pred_p == 0) & (y == 1)).sum())
# angle between the two normals, and distance between the two lines at the cloud centre
angle = math.degrees(math.acos(float(w_final @ w_pocket / (nw_f * nw_p))))
gap_at_centre = abs(cdist_f - cdist_p)

# The update log: which samples fire, of which kind, and how much they change (w, b).
upd_pos = np.array([u.sample for u in model.log])
upd_err = np.array([u.error for u in model.log])  # -1 = false positive, +1 = false negative
acc_after = np.array([u.acc for u in model.log])  # accuracy after every single update
n_fp, n_fn = int((upd_err == -1).sum()), int((upd_err == +1).sum())  # = updates in the Class-0 / Class-1 block
n_blk0_early = int((upd_pos < 200).sum())  # first 200 samples of the Class-0 block
n_blk1_early = int(((upd_pos >= N_PER_CLASS) & (upd_pos < N_PER_CLASS + 200)).sum())
upd_hist = np.bincount(model.updates)  # upd_hist[k] = number of epochs with k updates
fired, fired_n = np.unique(upd_pos, return_counts=True)  # distinct samples that ever fire, and how often
top4 = np.argsort(-fired_n, kind="stable")[:4]
step_wb = ETA * upd_err[:, None] * np.c_[X[upd_pos], np.ones(len(upd_pos))]  # (dw, db) of every update
net_wb = np.zeros((model.n_epochs, 3))  # net (dw, db) of each whole epoch
np.add.at(net_wb, np.array([u.epoch for u in model.log]) - 1, step_wb)
net_norm = np.linalg.norm(net_wb, axis=1)

pocket_log = [u for u in model.log if u.epoch == pocket_epoch]  # the updates of the pocket epoch
pocket_k = next(k for k, u in enumerate(model.log) if u.acc == pocket_acc)  # first update reaching the best accuracy
pocket_sample = model.log[pocket_k].sample
start_of_epoch = [u for u in model.log if u.epoch < pocket_epoch][-1]  # state when the pocket epoch begins

# Accuracy ceilings of a straight line. Population (Bayes) value for two equal
# isotropic Gaussians: Phi(||mu1-mu0|| / (2 sigma)); and the best line on THIS
# sample, found by a dense sweep of directions with the exact best threshold.
delta = math.dist(MEAN_0, MEAN_1)
bayes_acc = 0.5 * (1 + math.erf((delta / (2 * sigma)) / math.sqrt(2)))


def best_line_accuracy(X, y, n_angles=3600):
    """Highest accuracy of any line `d.x >= t` (Class 1 on the >= side): sweep the
    direction d over [0, 2*pi) and, for each d, take the exact best threshold t by
    sorting the projections and scanning every cut. A lower bound on the true
    optimum (only n_angles directions are tried), but a tight one."""
    best = 0.0
    for a in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        ys = y[np.argsort(X @ np.array([np.cos(a), np.sin(a)]))]
        zeros_left = np.concatenate([[0], np.cumsum(ys == 0)])  # Class-0 among the first k (predicted 0)
        ones_right = ys.sum() - np.concatenate([[0], np.cumsum(ys)])  # Class-1 among the rest (predicted 1)
        best = max(best, float((zeros_left + ones_right).max()) / len(y))
    return best


best_line_acc = best_line_accuracy(X, y)

# ---------------------------------------------------------------------------
# Sanity checks -- fail loudly if the reproducible pipeline breaks
# ---------------------------------------------------------------------------

assert X.shape == (2000, 2) and y.shape == (2000,)
assert np.bincount(y).tolist() == [1000, 1000]
assert not model.converged, "overlapping, non-separable data should never converge"
assert model.n_epochs == 100, "non-separable data should run the full 100-epoch cap"
assert 0.45 < final_acc < 0.55, "final-iterate accuracy expected near a coin flip (~50%)"
assert 0.70 < pocket_acc < 0.76, "pocket accuracy expected near the ~73% best-line ceiling"
assert pocket_acc > final_acc, "pocket must not do worse than the final-epoch snapshot"
assert pocket_acc == acc_after.max() and model.accuracy(X, y, w_pocket, b_pocket) == pocket_acc
assert best_line_acc >= pocket_acc - 1e-9, "no line can beat the best line"
assert start_of_epoch.acc == model.history[pocket_epoch - 2]
# b only ever moves by +-eta per mistake: it is eta * (#false negatives - #false positives).
assert abs(b_final - ETA * (n_fn - n_fp)) < 1e-12
assert n_fp == int((upd_pos < N_PER_CLASS).sum()) and n_fn == int((upd_pos >= N_PER_CLASS).sum())

# ---------------------------------------------------------------------------
# Printed report -- every number the write-up needs also appears here
# ---------------------------------------------------------------------------

print("=== Part A: data ===")
print(f"Class 0: n={N_PER_CLASS}, mean={MEAN_0}, cov={COV_0}")
print(f"Class 1: n={N_PER_CLASS}, mean={MEAN_1}, cov={COV_1}")

print("\n=== Part B: training at eta=0.01, max_epochs=100 (one fit, dataset order) ===")
print(f"epochs run = {model.n_epochs} (converged={model.converged})")
print(f"final   w = [{w_final[0]:.4f}, {w_final[1]:.4f}], b = {b_final:.4f}, "
      f"accuracy = {final_acc:.4f} ({int(mis_final.sum())}/2000 misclassified)")
print(f"pocket  w = [{w_pocket[0]:.4f}, {w_pocket[1]:.4f}], b = {b_pocket:.4f}, "
      f"accuracy = {pocket_acc:.4f} ({int(mis_pocket.sum())}/2000 misclassified), "
      f"epoch {pocket_epoch} (update {pocket_k + 1} of {len(model.log)}, at sample index {pocket_sample})")
steps = [(e + 1, round(a, 4)) for e, a in enumerate(model.pocket_history) if e == 0 or a != model.pocket_history[e - 1]]
print(f"pocket accuracy steps (epoch, best-so-far): {steps}")

print("\n=== Part C: figure data ===")
print(f"best epoch-end accuracy over the run = {max(model.history):.4f} "
      f"(a pocket tested only at epoch ends would stop here); "
      f"epoch-end accuracy range = [{min(model.history):.4f}, {max(model.history):.4f}]")
print(f"accuracy right after a single update: min {acc_after.min():.4f}, max {acc_after.max():.4f}, "
      f"{int((acc_after > 0.7).sum())} of {len(acc_after)} updates give > 70%")
print(f"final : predicts Class 1 for {p1_f:.2%} of the points (FP={fp_f}, FN={fn_f})")
print(f"pocket: predicts Class 1 for {p1_p:.2%} of the points (FP={fp_p}, FN={fn_p})")

print("\n=== Part D.1: step sizes, boundary position, update pattern ===")
print(f"mean ||x|| = {norms.mean():.4f}; per mistake b moves eta = {ETA}, w moves eta*||x|| = {step_w:.4f} "
      f"({step_w / ETA:.1f}x)")
print(f"w.x+b at the mistaken point changes by eta*(||x||^2+1) = {score_shift:.4f} "
      f"(bias share {ETA / score_shift:.1%})")
print(f"cloud centre = ({centre[0]:.4f}, {centre[1]:.4f}), distance from origin = {np.linalg.norm(centre):.4f}, "
      f"per-axis std = {sigma:.4f}")
print(f"final : ||w|| = {nw_f:.4f}, one step of w = {step_w / nw_f:.0%} of ||w||, |b|/||w|| = {off_f:.4f}, "
      f"centre is {cdist_f:+.4f} from the line ({cdist_f / sigma:+.2f} sigma)")
print(f"pocket: ||w|| = {nw_p:.4f}, |b|/||w|| = {off_p:.4f}, "
      f"centre is {cdist_p:+.4f} from the line ({cdist_p / sigma:+.2f} sigma)")
print(f"the two lines are {gap_at_centre:.4f} apart at the cloud centre; their normals differ by {angle:.1f} degrees")
print(f"b = eta*(#FN - #FP) = {ETA}*({n_fn} - {n_fp}) = {ETA * (n_fn - n_fp):.2f}")
print(f"updates per epoch: {', '.join(f'{k} in {n} epochs' for k, n in enumerate(upd_hist) if n)}; "
      f"total = {sum(model.updates)}")
print(f"false-positive corrections (Class-0 block): {n_fp} ({n_blk0_early} within its first 200 samples); "
      f"false-negative corrections (Class-1 block): {n_fn} ({n_blk1_early} within its first 200 samples)")
print(f"distinct samples that ever fire an update: {len(fired)} of {len(X)}; the four most frequent "
      f"(sample: epochs in which it fires): {', '.join(f'{fired[k]}: {fired_n[k]}' for k in top4)} "
      f"= {int(fired_n[top4].sum())} of {len(model.log)} updates")
print(f"net change of (w, b) over one whole epoch: mean norm {net_norm.mean():.4f}, max {net_norm.max():.4f}; "
      f"one update changes (w, b) by {np.linalg.norm(step_wb, axis=1).mean():.4f} on average")
print(f"epoch {pocket_epoch} (the pocket epoch): state at its start and after each of its updates")
for label, kind, u in [("start", "  ", start_of_epoch)] + [
    (str(u.sample), "FP" if u.error < 0 else "FN", u) for u in pocket_log
]:
    nw = float(np.linalg.norm(u.w))
    print(f"   sample {label:>5} {kind}: w = [{u.w[0]:+.4f}, {u.w[1]:+.4f}], b = {u.b:+.2f}, ||w|| = {nw:.4f}, "
          f"|b|/||w|| = {abs(u.b) / nw:.2f}, accuracy = {u.acc:.4f}")

print("\n=== Part D.2/D.3: accuracy ceilings ===")
print(f"population Bayes-linear accuracy Phi(||mu1-mu0||/(2 sigma)) = {bayes_acc:.4f}")
print(f"best line on this sample (3600-direction sweep, exact thresholds) = {best_line_acc:.4f}")

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
# Figure 5 -- both decision boundaries; misclassified points of each model
# (left panel: final weights, right panel: pocket weights)
# ---------------------------------------------------------------------------

xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
xs = np.array(xlim)


def boundary_ys(w, b):
    return -(w[0] * xs + b) / w[1]


fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharex=True, sharey=True)
for ax, mis, name in ((axes[0], mis_final, "final"), (axes[1], mis_pocket, "pocket")):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, alpha=0.45, color=COLOR_0, edgecolors="white",
               linewidths=0.2, label="Class 0", zorder=2)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, alpha=0.45, color=COLOR_1, edgecolors="white",
               linewidths=0.2, label="Class 1", zorder=2)
    ax.plot(xs, boundary_ys(w_final, b_final), color="black", linewidth=1.8, linestyle="-",
            zorder=4, label=f"Final boundary (acc={final_acc:.2%})")
    ax.plot(xs, boundary_ys(w_pocket, b_pocket), color="black", linewidth=1.8, linestyle="--",
            zorder=4, label=f"Pocket boundary (acc={pocket_acc:.2%})")
    ax.scatter(X[mis, 0], X[mis, 1], s=45, facecolors="none", edgecolors="black",
               linewidths=0.9, marker="^", zorder=5,
               label=f"Misclassified by the {name} weights (n={int(mis.sum())})")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"Misclassified points of the {name} weights", fontsize=11)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="upper right", fontsize=8.5)
fig.suptitle("Figure 5 -- Final vs. pocket decision boundary", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 6 -- current (final-so-far) vs. best-so-far (pocket) accuracy / epoch
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.5, 5.4))
epochs = np.arange(1, len(model.history) + 1)
# Accuracy right after each single update, placed inside its epoch (x = epoch - 1 + position / n).
upd_x = [u.epoch - 1 + (u.sample + 1) / len(X) for u in model.log]

ax.axhline(0.5, color="#888888", linewidth=0.9, linestyle=":", zorder=1)
ax.axhline(best_line_acc, color="#888888", linewidth=0.9, linestyle=":", zorder=1)
ax.text(100.5, 0.497, "chance (50%)", ha="right", va="top", fontsize=8.5, color="#555555")
ax.text(100.5, best_line_acc, f"best line on this sample ({best_line_acc:.2%})", ha="right",
        va="bottom", fontsize=8.5, color="#555555")
ax.plot(upd_x, acc_after, linestyle="none", marker=".", markersize=3.5, color="#b0b0b0",
        label="Right after each update (inside the epoch)", zorder=2)
ax.plot(epochs, model.history, color=COLOR_0, linewidth=1.2, marker="o", markersize=2.8,
        label="Current weights, end of epoch", zorder=3)
ax.plot(epochs, model.pocket_history, color=COLOR_1, linewidth=1.8, linestyle="--",
        label="Best so far (pocket)", zorder=4)
ax.set_ylim(0.4, 0.8)
ax.set_xlim(0, len(model.history) + 0.5)
ax.set_title("Figure 6 -- Accuracy vs. epoch: current vs. pocket", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (full dataset)")
ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6.png", dpi=150)
plt.close(fig)

print(f"\nSaved: {FIG_DIR / 'fig4.png'}")
print(f"Saved: {FIG_DIR / 'fig5.png'}")
print(f"Saved: {FIG_DIR / 'fig6.png'}")
