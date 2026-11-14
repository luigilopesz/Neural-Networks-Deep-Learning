"""Exercise 3 -- Multi-class classification, same network (Exercise 2's MLP, untouched).

Extends the shared MLP (`mlp.py`) from 2 classes to 3. The output layer grows
to 3 units, softmax replaces sigmoid, categorical cross-entropy replaces
binary cross-entropy. `MLP.forward`, `.backward` and `.update` do not change:
`mlp.py`'s docstring shows `dL/dU = (P - Y) / N` is the same expression for
sigmoid+BCE and softmax+CCE, so one code path handles both (the derivation
for the softmax+CCE case is worked out in the report). Only the constructor
arguments differ: `layer_sizes=[4, HIDDEN, 3]`, `output="softmax"`.

HIDDEN, LR and EPOCHS are module-level constants with no import side effects,
so Exercise 4 can do `from ex3_multiclass import HIDDEN, LR, EPOCHS` and train
a deeper network `[4, HIDDEN, HIDDEN, 3]` on the identical data, split and
training budget.

Run from this directory:
    python ex3_multiclass.py
Figures are written to ../figures/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mlp import MLP

# Exercise 4 imports these three constants and trains [4, HIDDEN, HIDDEN, 3]
# on the same data, split and training budget -- do not rename.
HIDDEN = 16
LR = 0.1
EPOCHS = 4000

CLASS_NAMES = ["Class 0", "Class 1", "Class 2"]


def main() -> None:
    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

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

    # ------------------------------------------------------------------
    # Part A -- data, exactly as the statement writes it
    # ------------------------------------------------------------------
    X, y = make_classification(
        n_samples=1500, n_features=4, n_informative=4, n_redundant=0,
        n_repeated=0, n_classes=3, n_clusters_per_class=2, class_sep=1.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Y_train = np.eye(3)[y_train]  # one-hot for the network, plain numpy

    print("=== Part A: data ===")
    print(f"X: {X.shape}, y: {y.shape}, class counts: {np.bincount(y).tolist()}")
    print(f"train: {X_train.shape[0]}, test: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # Part B -- linear baseline (this is the "extend to 3 classes" item;
    # the actual softmax+CCE derivation lives in the report, not here)
    # ------------------------------------------------------------------
    baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)
    print("\n=== Part B: baseline ===")
    print(f"LogisticRegression test accuracy = {baseline_acc:.4f}")

    # ------------------------------------------------------------------
    # Part C -- train and evaluate the MLP
    # ------------------------------------------------------------------
    model = MLP([4, HIDDEN, 3], output="softmax", rng=rng)
    losses = model.fit(X_train, Y_train, epochs=EPOCHS, lr=LR)  # full-batch (batch_size=None)

    y_pred = model.predict(X_test)
    mlp_acc = float(np.mean(y_pred == y_test))

    print("\n=== Part C: MLP ===")
    print(f"architecture = [4, {HIDDEN}, 3], hidden=tanh, output=softmax")
    print(f"lr = {LR}, epochs = {EPOCHS}")
    print(f"final training loss = {losses[-1]:.4f}")
    print(f"test accuracy = {mlp_acc:.4f}")
    print(f"n_params = {model.n_params()}")

    assert mlp_acc >= 0.78, f"MLP accuracy {mlp_acc:.4f} below 0.78"
    assert baseline_acc <= 0.75, f"baseline accuracy {baseline_acc:.4f} above 0.75"

    # ------------------------------------------------------------------
    # Figure 4 -- training loss vs epoch
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(np.arange(1, EPOCHS + 1), losses, color="#2a78d6", linewidth=1.3,
            label="Training loss (softmax + categorical cross-entropy)")
    ax.set_title("Figure 4 -- Training loss vs. epoch", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean categorical cross-entropy")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5 -- confusion matrix on the test set
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    print("\n=== Part C/D: confusion matrix (rows=true, cols=predicted) ===")
    print(cm)

    off_diag = cm.copy()
    np.fill_diagonal(off_diag, -1)
    i, j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
    pair_total = int(cm[i, j] + cm[j, i])
    print(
        f"most-confused pair: true={CLASS_NAMES[i]}, predicted={CLASS_NAMES[j]} "
        f"({int(cm[i, j])} points); symmetric total (both directions) = {pair_total}"
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(3))
    ax.set_yticklabels(CLASS_NAMES)
    for r in range(3):
        for c in range(3):
            color = "white" if cm[r, c] > cm.max() / 2 else "black"
            ax.text(c, r, str(int(cm[r, c])), ha="center", va="center", color=color)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(
        "Figure 5 -- Confusion matrix (test set)\nClasses: Class 0, Class 1, Class 2",
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved: {fig_dir / 'fig4.png'}")
    print(f"Saved: {fig_dir / 'fig5.png'}")


if __name__ == "__main__":
    main()
