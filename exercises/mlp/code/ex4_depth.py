"""Exercise 4 -- Does a second hidden layer help? Controlled comparison against Exercise 3.

Same data, same split, same training budget, same learning rate as Exercise 3. The only
thing that changes is depth: shallow `[4, HIDDEN, 3]` (Exercise 3's network) versus deep
`[4, HIDDEN, HIDDEN, 3]`. `HIDDEN`, `LR` and `EPOCHS` are imported from `ex3_multiclass`,
not retyped, so the budget cannot silently drift between the two exercises.

`ex3_multiclass.py` does not expose a separate data-building function -- its `make_classification`
and `train_test_split` calls live inline in `main()` -- so they are copied verbatim below,
unchanged, to reproduce the identical dataset and split.

For each seed in [42, 43, 44], one `rng = np.random.default_rng(seed)` is created and the
shallow network is built and trained from it FIRST, then the deep network SECOND, both
consuming the same rng stream. Because Exercise 3 built exactly one `MLP([4, HIDDEN, 3], ...)`
from `np.random.default_rng(42)` and trained it full-batch (which never touches the rng again),
the seed-42 shallow run here draws its weights in exactly the same order Exercise 3 did and
must reproduce its numbers exactly -- checked below with an assert, not just eyeballed.

Run from this directory:
    python ex4_depth.py
Figure is written to ../figures/fig6.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ex3_multiclass import HIDDEN, LR, EPOCHS
from mlp import MLP

SEEDS = [42, 43, 44]
EX3_TEST_ACC = 0.8333
EX3_FINAL_LOSS = 0.2811

COLOR_SHALLOW = "#2a78d6"
COLOR_DEEP = "#eb6834"


def main() -> None:
    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

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
    # Data -- copied verbatim from ex3_multiclass.main(), see module docstring.
    # ------------------------------------------------------------------
    X, y = make_classification(
        n_samples=1500, n_features=4, n_informative=4, n_redundant=0,
        n_repeated=0, n_classes=3, n_clusters_per_class=2, class_sep=1.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Y_train = np.eye(3)[y_train]

    print("=== Data (identical to Exercise 3) ===")
    print(f"train: {X_train.shape[0]}, test: {X_test.shape[0]}")
    print(f"architectures: shallow=[4, {HIDDEN}, 3]  deep=[4, {HIDDEN}, {HIDDEN}, 3]")
    print(f"lr = {LR} (imported from ex3_multiclass), epochs = {EPOCHS} (imported), full-batch")

    # ------------------------------------------------------------------
    # Train both nets for every seed -- shallow first, then deep, same rng.
    # ------------------------------------------------------------------
    results = {"shallow": [], "deep": []}  # each entry: dict(seed, loss, acc, curve)
    n_params = {}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        shallow = MLP([4, HIDDEN, 3], output="softmax", rng=rng)
        shallow_losses = shallow.fit(X_train, Y_train, epochs=EPOCHS, lr=LR)
        shallow_acc = float(np.mean(shallow.predict(X_test) == y_test))
        results["shallow"].append(
            {"seed": seed, "loss": shallow_losses[-1], "acc": shallow_acc, "curve": shallow_losses}
        )
        n_params["shallow"] = shallow.n_params()

        deep = MLP([4, HIDDEN, HIDDEN, 3], output="softmax", rng=rng)
        deep_losses = deep.fit(X_train, Y_train, epochs=EPOCHS, lr=LR)
        deep_acc = float(np.mean(deep.predict(X_test) == y_test))
        results["deep"].append(
            {"seed": seed, "loss": deep_losses[-1], "acc": deep_acc, "curve": deep_losses}
        )
        n_params["deep"] = deep.n_params()

        print(f"\nseed {seed}: shallow loss={shallow_losses[-1]:.4f} acc={shallow_acc:.4f} | "
              f"deep loss={deep_losses[-1]:.4f} acc={deep_acc:.4f}")

    # ------------------------------------------------------------------
    # Reproduction check: seed-42 shallow run must equal Exercise 3 exactly.
    # ------------------------------------------------------------------
    seed42_shallow = results["shallow"][0]
    assert seed42_shallow["seed"] == 42
    loss_match = round(seed42_shallow["loss"], 4) == EX3_FINAL_LOSS
    acc_match = round(seed42_shallow["acc"], 4) == EX3_TEST_ACC
    print("\n=== Reproduction check (seed-42 shallow vs. Exercise 3) ===")
    print(f"test accuracy: {seed42_shallow['acc']:.4f} (expected {EX3_TEST_ACC}) -> "
          f"{'MATCH' if acc_match else 'MISMATCH'}")
    print(f"final train loss: {seed42_shallow['loss']:.4f} (expected {EX3_FINAL_LOSS}) -> "
          f"{'MATCH' if loss_match else 'MISMATCH'}")
    assert acc_match, f"seed-42 shallow test accuracy {seed42_shallow['acc']:.4f} != {EX3_TEST_ACC}"
    assert loss_match, f"seed-42 shallow final loss {seed42_shallow['loss']:.4f} != {EX3_FINAL_LOSS}"

    # ------------------------------------------------------------------
    # Per-seed table and mean +/- std summary.
    # ------------------------------------------------------------------
    print("\n=== Per-seed results ===")
    print(f"{'seed':>5} | {'shallow loss':>13} {'shallow acc':>12} | "
          f"{'deep loss':>10} {'deep acc':>9} | {'diff (deep-shallow)':>20}")
    diffs = []
    for s, d in zip(results["shallow"], results["deep"]):
        diff = d["acc"] - s["acc"]
        diffs.append(diff)
        print(f"{s['seed']:>5} | {s['loss']:>13.4f} {s['acc']:>12.4f} | "
              f"{d['loss']:>10.4f} {d['acc']:>9.4f} | {diff:>+20.4f}")

    shallow_losses_final = [r["loss"] for r in results["shallow"]]
    shallow_accs = [r["acc"] for r in results["shallow"]]
    deep_losses_final = [r["loss"] for r in results["deep"]]
    deep_accs = [r["acc"] for r in results["deep"]]

    print("\n=== Mean +/- std over seeds ===")
    print(f"shallow: train loss = {np.mean(shallow_losses_final):.4f} +/- {np.std(shallow_losses_final):.4f}, "
          f"test acc = {np.mean(shallow_accs):.4f} +/- {np.std(shallow_accs):.4f}")
    print(f"deep:    train loss = {np.mean(deep_losses_final):.4f} +/- {np.std(deep_losses_final):.4f}, "
          f"test acc = {np.mean(deep_accs):.4f} +/- {np.std(deep_accs):.4f}")
    print(f"accuracy difference (deep - shallow): mean = {np.mean(diffs):+.4f}, "
          f"per-seed = {[round(d, 4) for d in diffs]}")
    print(f"n_params: shallow = {n_params['shallow']}, deep = {n_params['deep']}")

    # ------------------------------------------------------------------
    # Figure 6 -- training loss curves, seed-42 solid, other seeds faint.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    epochs_axis = np.arange(1, EPOCHS + 1)
    for r in results["shallow"]:
        if r["seed"] == 42:
            ax.plot(epochs_axis, r["curve"], color=COLOR_SHALLOW, linewidth=1.6,
                     label=f"Shallow [4, {HIDDEN}, 3] (seed 42)")
        else:
            ax.plot(epochs_axis, r["curve"], color=COLOR_SHALLOW, linewidth=1.0, alpha=0.35)
    for r in results["deep"]:
        if r["seed"] == 42:
            ax.plot(epochs_axis, r["curve"], color=COLOR_DEEP, linewidth=1.6,
                     label=f"Deep [4, {HIDDEN}, {HIDDEN}, 3] (seed 42)")
        else:
            ax.plot(epochs_axis, r["curve"], color=COLOR_DEEP, linewidth=1.0, alpha=0.35)
    ax.set_title("Figure 6 -- Training loss vs. epoch: shallow vs. deep", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean categorical cross-entropy")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved: {fig_dir / 'fig6.png'}")


if __name__ == "__main__":
    main()
