"""Steps 0-4: everything that is DECIDED, using train (2000-2016) and validation (2017-2020) only.

The test split (2021-2024) is never loaded here.

    0  pipeline fitted on train; main metric and success criterion printed BEFORE any training
    1  baselines on validation: majority class, monsoon gate, logistic regression
    2  the network: architecture and parameter count
    3  training of the chosen configuration: curves per epoch (Figure 3.1, animation 3.2)
    4  search over 16 configurations (4 regularisations x 2 architectures x 2 learning rates),
       sorted by validation PR-AUC; regularisation comparison (Figure 4.1)

Writes results_select.json, read by test_once.py.
Run: python select_on_val.py   (about one minute on a laptop CPU)
"""
from __future__ import annotations

import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from common import (FIG_DIR, HERE, SEED, TARGET_RECALL, at_threshold, finish, fmt, load_splits,
                    savefig, scores, threshold_for_recall)
from model import train

CRITERION = ("success = the MLP's test PR-AUC (mean of 3 seeds) is at least 10 % higher "
             "(relative) than the logistic regression's test PR-AUC")

# Regularisation settings compared (step 4). AdamW's weight decay is decoupled: each step
# shrinks every weight by lr * weight_decay.
REGS = {"none": dict(dropout=0.0, weight_decay=0.0),
        "weight decay": dict(dropout=0.0, weight_decay=1e-2),
        "dropout": dict(dropout=0.2, weight_decay=0.0),
        "dropout + weight decay": dict(dropout=0.2, weight_decay=1e-2)}
ARCHS = {"21-32-1": (32,), "21-64-32-1": (64, 32)}
LRS = [1e-3, 3e-3]
BATCH = 1024


def main():
    # ---------------------------------------------------------------- step 0
    d = load_splits(("train", "validation"))
    tr, va = d["train"], d["validation"]
    mon_va = va["frame"]["monsoon_season"].to_numpy()
    print("== step 0: starting point")
    print(f"train {tr['X'].shape}  validation {va['X'].shape}  (pipeline fitted on train only)")
    print(f"positives: train {tr['y'].sum()} ({tr['y'].mean():.4%})  validation {va['y'].sum()} ({va['y'].mean():.4%})")
    print(f"validation monsoon rows {int(mon_va.sum())}, base rate inside the monsoon {va['y'][mon_va == 1].mean():.4%}")
    print("main metric: PR-AUC (average precision); also ROC-AUC and PR-AUC inside the monsoon")
    print(f"threshold rule: highest threshold with validation recall >= {TARGET_RECALL:.0%}")
    print(CRITERION)

    # ---------------------------------------------------------------- step 1
    print("\n== step 1: baselines on validation")
    base = {}
    dummy = DummyClassifier(strategy="most_frequent").fit(tr["X"], tr["y"])
    p = dummy.predict_proba(va["X"])[:, 1]
    base["majority"] = dict(**scores(va["y"], p, mon_va), **at_threshold(va["y"], p, 0.5))
    p = mon_va.astype(float)                                  # "flood whenever monsoon_season = 1"
    base["monsoon gate"] = dict(**scores(va["y"], p, mon_va), **at_threshold(va["y"], p, 1.0))
    logit = LogisticRegression(class_weight="balanced", max_iter=2000).fit(tr["X"], tr["y"])
    p_log = logit.predict_proba(va["X"])[:, 1]
    t_log = threshold_for_recall(va["y"], p_log)
    base["logistic"] = dict(**scores(va["y"], p_log, mon_va), **at_threshold(va["y"], p_log, t_log))
    for name, r in base.items():
        print(f"{name:<13} {fmt(r)}  | at t={r['threshold']:.4f}: precision={r['precision']:.4f} "
              f"recall={r['recall']:.4f} accuracy={r['accuracy']:.4f}")

    # ---------------------------------------------------------------- step 4 (search)
    print("\n== step 4: search, 16 configurations, chosen on validation PR-AUC")
    runs = []
    for (reg, rk), (arch, hidden), lr in itertools.product(REGS.items(), ARCHS.items(), LRS):
        _, hist, info = train(tr["X"], tr["y"], va["X"], va["y"], hidden=hidden, lr=lr, batch=BATCH,
                              seed=SEED, **rk)
        best = hist[info["best_epoch"] - 1]
        runs.append(dict(reg=reg, arch=arch, hidden=list(hidden), lr=lr, **rk,
                         n_params=info["n_params"], best_epoch=info["best_epoch"],
                         stopped_epoch=info["stopped_epoch"], val_pr_auc=info["best_val_pr_auc"],
                         gap=best["val_loss"] - best["train_loss"], history=hist))
    runs.sort(key=lambda r: -r["val_pr_auc"])
    print(f"{'#':>2} {'regularisation':<23} {'arch':<11} {'lr':>6} {'params':>6} {'best ep':>7} "
          f"{'val PR-AUC':>10} {'val-train loss':>14}")
    for i, r in enumerate(runs, 1):
        print(f"{i:>2} {r['reg']:<23} {r['arch']:<11} {r['lr']:>6.0e} {r['n_params']:>6} "
              f"{r['best_epoch']:>7} {r['val_pr_auc']:>10.4f} {r['gap']:>14.4f}")
    chosen = runs[0]
    print(f"chosen: {chosen['reg']}, {chosen['arch']}, lr={chosen['lr']:.0e}")

    # regularisation comparison at the chosen architecture and learning rate
    print("\nregularisation comparison (chosen architecture and learning rate):")
    same = {r["reg"]: r for r in runs if r["arch"] == chosen["arch"] and r["lr"] == chosen["lr"]}
    for reg in REGS:
        r = same[reg]
        print(f"  {reg:<23} val PR-AUC={r['val_pr_auc']:.4f}  best epoch={r['best_epoch']:>3}  "
              f"val-train loss at best epoch={r['gap']:.4f}  "
              f"vs none: {r['val_pr_auc'] - same['none']['val_pr_auc']:+.4f}")

    # ---------------------------------------------------------------- steps 2-3 (chosen network)
    rk = dict(dropout=chosen["dropout"], weight_decay=chosen["weight_decay"])
    model, hist, info = train(tr["X"], tr["y"], va["X"], va["y"], hidden=tuple(chosen["hidden"]),
                              lr=chosen["lr"], batch=BATCH, seed=SEED, keep_val_probs=True, **rk)
    assert info["best_val_pr_auc"] == chosen["val_pr_auc"], "re-run must reproduce the search"
    print("\n== step 2: the network")
    print(model)
    print(f"parameters {info['n_params']}, training rows {len(tr['y'])}, "
          f"rows per parameter {len(tr['y']) / info['n_params']:.1f}, pos_weight {info['pos_weight']:.2f}")
    print("\n== step 3: training of the chosen configuration")
    print(f"AdamW lr={chosen['lr']:.0e} batch={BATCH} max_epochs=60 patience=8; "
          f"best epoch {info['best_epoch']}, stopped at epoch {info['stopped_epoch']}")
    print(f"steps per epoch {int(np.ceil(len(tr['y']) / BATCH))}, positives per batch ~{BATCH * tr['y'].mean():.1f}")
    print(f"  {'epoch':>5} {'train loss':>10} {'val loss':>9} {'val-train':>9} {'val PR-AUC':>10}")
    for h in hist:
        print(f"  {h['epoch']:>5} {h['train_loss']:>10.4f} {h['val_loss']:>9.4f} "
              f"{h['val_loss'] - h['train_loss']:>9.4f} {h['val_pr_auc']:>10.4f}")
    late = [h["val_pr_auc"] for h in hist if h["epoch"] >= 11]
    print(f"  val PR-AUC from epoch 11 on: {min(late):.4f} to {max(late):.4f}; "
          f"above logistic ({base['logistic']['pr_auc']:.4f}) from epoch "
          f"{min(h['epoch'] for h in hist if all(g['val_pr_auc'] > base['logistic']['pr_auc'] for g in hist[h['epoch'] - 1:]))}")
    min_val = min(hist, key=lambda h: h["val_loss"])
    print(f"  lowest val loss at epoch {min_val['epoch']} ({min_val['val_loss']:.4f})")

    p_mlp = info["val_probs"][info["best_epoch"] - 1]
    t_mlp = threshold_for_recall(va["y"], p_mlp)
    mlp_val = dict(**scores(va["y"], p_mlp, mon_va), **at_threshold(va["y"], p_mlp, t_mlp))
    print(f"MLP on validation: {fmt(mlp_val)}  | at t={t_mlp:.4f}: precision={mlp_val['precision']:.4f} "
          f"recall={mlp_val['recall']:.4f}")
    print(f"logistic on validation at its own 80 % threshold: precision={base['logistic']['precision']:.4f}")
    print(f"MLP / logistic validation PR-AUC: {mlp_val['pr_auc'] / base['logistic']['pr_auc']:.4f} "
          f"({mlp_val['pr_auc'] / base['logistic']['pr_auc'] - 1:+.2%})")

    # ---------------------------------------------------------------- figures
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(ep, [h["train_loss"] for h in hist], label="train")
    a.plot(ep, [h["val_loss"] for h in hist], label="validation")
    a.axvline(info["best_epoch"], color="grey", ls="--", label=f"best epoch ({info['best_epoch']})")
    finish(a, title="(a) Weighted BCE loss per epoch", xlabel="epoch", ylabel="loss (nats per row, pos_weight 46.3)")
    b.plot(ep, [h["val_pr_auc"] for h in hist], color="C2", label="MLP, validation")
    b.axhline(base["logistic"]["pr_auc"], color="C3", ls=":", label="logistic, validation")
    b.axvline(info["best_epoch"], color="grey", ls="--", label="best epoch (weights restored)")
    finish(b, title="(b) Validation PR-AUC per epoch", xlabel="epoch", ylabel="PR-AUC (average precision)")
    fig.suptitle(f"Figure 3.1 - Training curves of the chosen MLP ({chosen['arch']}, {chosen['reg']}, lr {chosen['lr']:.0e})")
    savefig(fig, "fig31_training_curves")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for reg in REGS:
        h = same[reg]["history"]
        a.plot([x["epoch"] for x in h], [x["val_pr_auc"] for x in h], label=reg)
        b.plot([x["epoch"] for x in h], [x["val_loss"] - x["train_loss"] for x in h], label=reg)
    finish(a, title="(a) Validation PR-AUC per epoch", xlabel="epoch", ylabel="PR-AUC (average precision)")
    b.axhline(0, color="grey", lw=0.8)
    finish(b, title="(b) Validation minus training loss", xlabel="epoch", ylabel="loss gap (nats per row)")
    fig.suptitle(f"Figure 4.1 - Regularisation compared with none ({chosen['arch']}, lr {chosen['lr']:.0e})")
    savefig(fig, "fig41_regularisation")

    # Animation 3.2: the validation precision-recall curve, epoch by epoch
    fig, ax = plt.subplots(figsize=(6, 4.5))
    pl, rl, _ = precision_recall_curve(va["y"], p_log)
    ax.plot(rl, pl, color="C3", ls=":", label=f"logistic (PR-AUC {base['logistic']['pr_auc']:.3f})")
    ax.axhline(va["y"].mean(), color="grey", ls="--", label=f"base rate {va['y'].mean():.3f}")
    line, = ax.plot([], [], color="C0", label="MLP")
    finish(ax, title="", xlabel="recall", ylabel="precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.3)

    def draw(i):
        pr, rc, _ = precision_recall_curve(va["y"], info["val_probs"][i])
        line.set_data(rc, pr)
        line.set_label(f"MLP epoch {i + 1} (PR-AUC {hist[i]['val_pr_auc']:.3f})")
        ax.legend(loc="upper right")
        ax.set_title(f"Animation 3.2 - Validation PR curve, epoch {i + 1}/{len(hist)}")
        return (line,)

    anim = FuncAnimation(fig, draw, frames=len(hist))
    anim.save(FIG_DIR / "anim32_pr_curve.gif", writer=PillowWriter(fps=3), dpi=80)
    plt.close(fig)
    print("[figure] figures/anim32_pr_curve.gif")

    # ---------------------------------------------------------------- save
    out =dict(criterion=CRITERION, target_recall=TARGET_RECALL, batch=BATCH,
               baselines_val=base, mlp_val=mlp_val, logistic_threshold=t_log, mlp_threshold=t_mlp,
               chosen={k: v for k, v in chosen.items() if k != "history"},
               chosen_info={k: v for k, v in info.items() if k != "val_probs"}, chosen_history=hist,
               search=[{k: v for k, v in r.items() if k != "history"} for r in runs],
               regularisation={k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in same.items()})
    (HERE / "results_select.json").write_text(json.dumps(out, indent=1))
    print("[json] results_select.json")


if __name__ == "__main__":
    main()
