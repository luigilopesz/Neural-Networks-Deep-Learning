"""Steps 5-6: the ONLY script that reads the test split (2021-2024). Nothing is chosen here.

Everything frozen comes from results_select.json (written by select_on_val.py): the chosen
configuration and the rule "threshold = highest value with validation recall >= 80 %".

    5  baselines and the chosen MLP (retrained with seeds 42, 43, 44, early stopping on
       validation, threshold on validation) evaluated once on test; success criterion
    6  error per subgroup (basin, month), worst examples, permutation importance

Writes results_test.json.
Run: python test_once.py   (after select_on_val.py; about one minute)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve

from common import (HERE, SEED, at_threshold, finish, fmt, load_splits, pr_auc, savefig, scores,
                    threshold_for_recall)
from model import predict_proba, train

SEEDS = [SEED, SEED + 1, SEED + 2]
MEAS = ["rainfall_mm", "river_level_m", "soil_moisture_percent", "temperature_celsius"]


def main():
    sel = json.loads((HERE / "results_select.json").read_text())
    ch = sel["chosen"]
    d = load_splits(("train", "validation", "test"))
    tr, va, te = d["train"], d["validation"], d["test"]
    mon_va = va["frame"]["monsoon_season"].to_numpy()
    mon_te = te["frame"]["monsoon_season"].to_numpy()
    print(f"test {te['X'].shape}, positives {te['y'].sum()} ({te['y'].mean():.4%}), "
          f"inside the monsoon {te['y'][mon_te == 1].mean():.4%}")
    print(f"chosen configuration (frozen): {ch['reg']}, {ch['arch']}, lr={ch['lr']:.0e}")

    # ---------------------------------------------------------------- step 5: baselines
    res, probs = {}, {}
    p = DummyClassifier(strategy="most_frequent").fit(tr["X"], tr["y"]).predict_proba(te["X"])[:, 1]
    res["majority"], probs["majority"] = dict(**scores(te["y"], p, mon_te), **at_threshold(te["y"], p, 0.5)), p
    p = mon_te.astype(float)
    res["monsoon gate"], probs["monsoon gate"] = dict(**scores(te["y"], p, mon_te), **at_threshold(te["y"], p, 1.0)), p
    logit = LogisticRegression(class_weight="balanced", max_iter=2000).fit(tr["X"], tr["y"])
    t_log = threshold_for_recall(va["y"], logit.predict_proba(va["X"])[:, 1])     # from VALIDATION
    assert abs(t_log - sel["logistic_threshold"]) < 1e-12
    p = logit.predict_proba(te["X"])[:, 1]
    res["logistic"], probs["logistic"] = dict(**scores(te["y"], p, mon_te), **at_threshold(te["y"], p, t_log)), p

    # ---------------------------------------------------------------- step 5: MLP, 3 seeds
    seed_runs, models = [], {}
    for s in SEEDS:
        model, hist, info = train(tr["X"], tr["y"], va["X"], va["y"], hidden=tuple(ch["hidden"]),
                                  dropout=ch["dropout"], weight_decay=ch["weight_decay"], lr=ch["lr"],
                                  batch=sel["batch"], seed=s)
        t = threshold_for_recall(va["y"], predict_proba(model, va["X"]))              # from VALIDATION
        p = predict_proba(model, te["X"])
        seed_runs.append(dict(seed=s, best_epoch=info["best_epoch"], stopped_epoch=info["stopped_epoch"],
                              val_pr_auc=info["best_val_pr_auc"],
                              **scores(te["y"], p, mon_te), **at_threshold(te["y"], p, t)))
        models[s], probs[f"MLP seed {s}"] = model, p
    assert seed_runs[0]["val_pr_auc"] == ch["val_pr_auc"], "seed 42 must reproduce the selected model"
    res["MLP (seed 42)"] = seed_runs[0]

    print("\n== step 5: test set, thresholds from validation")
    print(f"{'model':<14} {'PR-AUC':>7} {'ROC-AUC':>7} {'PR-AUC mons.':>12} {'thr':>7} {'prec':>6} "
          f"{'recall':>6} {'F1':>6} {'macroF1':>7} {'TP':>5} {'FP':>6} {'FN':>5} {'TN':>6}")
    for name, r in list(res.items()) + [(f"MLP seed {r['seed']}", r) for r in seed_runs[1:]]:
        print(f"{name:<14} {r['pr_auc']:>7.4f} {r['roc_auc']:>7.4f} {r['pr_auc_monsoon']:>12.4f} "
              f"{r['threshold']:>7.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1']:>6.4f} "
              f"{r['macro_f1']:>7.4f} {r['tp']:>5} {r['fp']:>6} {r['fn']:>5} {r['tn']:>6}")

    agg = {k: (float(np.mean([r[k] for r in seed_runs])), float(np.std([r[k] for r in seed_runs], ddof=1)))
           for k in ("pr_auc", "roc_auc", "pr_auc_monsoon", "precision", "recall", "f1", "macro_f1",
                     "val_pr_auc", "best_epoch")}
    print("\nMLP over 3 seeds (mean ± sample std):")
    for k, (m, s) in agg.items():
        print(f"  {k:<15} {m:.4f} ± {s:.4f}")
    r0 = seed_runs[0]
    print(f"\nper-class report, MLP seed 42 at t={r0['threshold']:.4f}:")
    print(f"  class 0 (no flood): precision {r0['precision_0']:.4f}  recall {r0['recall_0']:.4f}  F1 {r0['f1_0']:.4f}  support {r0['tn'] + r0['fp']}")
    print(f"  class 1 (flood)   : precision {r0['precision']:.4f}  recall {r0['recall']:.4f}  F1 {r0['f1']:.4f}  support {r0['tp'] + r0['fn']}")
    print(f"  macro F1 {r0['macro_f1']:.4f}  accuracy {r0['accuracy']:.4f}")

    lg = res["logistic"]["pr_auc"]
    gap = agg["pr_auc"][0] - lg
    met = agg["pr_auc"][0] >= 1.10 * lg
    print(f"\ncriterion: {sel['criterion']}")
    print(f"  MLP {agg['pr_auc'][0]:.4f} vs logistic {lg:.4f}: ratio {agg['pr_auc'][0] / lg:.4f} "
          f"(needs >= 1.10), relative gain {agg['pr_auc'][0] / lg - 1:+.2%} -> {'MET' if met else 'NOT MET'}")
    print(f"  gap to logistic {gap:.4f} = {gap / agg['pr_auc'][1]:.1f} x the seed std ({agg['pr_auc'][1]:.4f})")
    print(f"  precision at ~80 % recall: MLP {agg['precision'][0]:.4f} vs logistic {res['logistic']['precision']:.4f}")
    for name, r in (("MLP seed 42", r0), ("logistic", res["logistic"])):
        print(f"  {name}: {r['fp'] / r['tp']:.2f} false alarms per flood caught")
    print(f"  MLP seed 42 vs logistic: {res['logistic']['fp'] - r0['fp']} fewer false alarms, "
          f"{r0['fn'] - res['logistic']['fn']} more missed floods")
    off = mon_te == 0
    print(f"  off-season test rows flagged by the MLP (seed 42): {int((probs['MLP seed 42'][off] >= r0['threshold']).sum())} of {int(off.sum())}")

    # ---------------------------------------------------------------- step 5: figures
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    show = [("MLP seed 42", "MLP (seed 42)", "C0"), ("logistic", "logistic", "C3"),
            ("monsoon gate", "monsoon gate", "C1"), ("majority", "majority", "grey")]
    for key, rkey, c in show:
        pr, rc, _ = precision_recall_curve(te["y"], probs[key])
        if key == "majority":        # a constant score: precision = base rate at every recall
            pr, rc = np.full(2, te["y"].mean()), np.array([0.0, 1.0])
        a.plot(rc, pr, color=c, drawstyle="steps-post", label=f"{key} (PR-AUC {res[rkey]['pr_auc']:.3f})")
        fpr, tpr, _ = roc_curve(te["y"], probs[key])
        b.plot(fpr, tpr, color=c, label=f"{key} (ROC-AUC {res[rkey]['roc_auc']:.3f})")
    a.scatter([r0["recall"]], [r0["precision"]], color="C0", zorder=3, label="MLP at the validation threshold")
    a.set_ylim(0, 0.3)
    finish(a, title="(a) Precision-recall, test", xlabel="recall (share of floods caught)", ylabel="precision (share of alarms that are floods)")
    b.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":")
    finish(b, title="(b) ROC, test", xlabel="false positive rate", ylabel="true positive rate (recall)")
    fig.suptitle("Figure 5.1 - Test-set curves: the MLP and the three baselines")
    savefig(fig, "fig51_pr_roc_test")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.subplots_adjust(wspace=0.4)
    for ax, (name, r) in zip(axes, (("MLP (seed 42)", r0), ("logistic", res["logistic"]))):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%} of row)", ha="center", va="center",
                        color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
        ax.set_xticks([0, 1], ["no flood", "flood"]); ax.set_yticks([0, 1], ["no flood", "flood"])
        finish(ax, title=f"{name}, threshold {r['threshold']:.3f}", xlabel="predicted", ylabel="actual", legend=False)
        ax.grid(False)
    fig.colorbar(im, ax=axes, label="share of the actual class (row)")
    fig.suptitle("Figure 5.2 - Test confusion matrices at the validation threshold (80 % recall)")
    fig.savefig(HERE.parent / "figures" / "fig52_confusion_test.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[figure] figures/fig52_confusion_test.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for key, c in (("MLP seed 42", "C0"), ("logistic", "C3")):
        fy, fx = calibration_curve(te["y"], probs[key], n_bins=10, strategy="quantile")
        ax.plot(fx, fy, "o-", color=c, label=key)
    ax.plot([0, 1], [0, 1], "k:", label="perfect calibration")
    ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01)
    finish(ax, title="Figure 5.3 - Calibration on test (10 quantile bins)",
           xlabel="mean predicted probability", ylabel="observed flood rate")
    savefig(fig, "fig53_calibration")
    bins_mlp = calibration_curve(te["y"], probs["MLP seed 42"], n_bins=10, strategy="quantile")
    print(f"calibration, MLP top bin: predicted {bins_mlp[1][-1]:.4f} observed {bins_mlp[0][-1]:.4f}; "
          f"mean predicted {probs['MLP seed 42'].mean():.4f} vs base rate {te['y'].mean():.4f} "
          f"({probs['MLP seed 42'].mean() / te['y'].mean():.1f}x)")

    # ---------------------------------------------------------------- step 6: subgroups
    print("\n== step 6: error analysis (MLP seed 42, test, validation threshold)")
    f = te["frame"].assign(p=probs["MLP seed 42"], y=te["y"])
    f["pred"] = (f["p"] >= r0["threshold"]).astype(int)
    f["month"] = f["date"].dt.month

    def group_table(key):
        rows = []
        for g, s in f.groupby(key):
            pos = s["y"].sum()
            alarms = s["pred"].sum()
            rows.append(dict(group=g, rows=len(s), floods=int(pos),
                             miss_rate=float(1 - (s["pred"] & s["y"]).sum() / pos) if pos else np.nan,
                             precision=float((s["pred"] & s["y"]).sum() / alarms) if alarms else np.nan,
                             alarms=int(alarms),
                             pr_auc=pr_auc(s["y"], s["p"]) if pos else np.nan))
        return pd.DataFrame(rows)

    by_basin, by_month = group_table("basin"), group_table("month")
    with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
        print("by basin:\n", by_basin.sort_values("miss_rate", ascending=False).to_string(index=False))
        print("by month:\n", by_month.to_string(index=False))
    # rank only groups with >= 100 floods: a miss rate over 1 or 22 floods is not a pattern
    wb = by_basin.loc[by_basin["miss_rate"].idxmax()]
    big = by_month[by_month["floods"] >= 100]
    wm = big.loc[big["miss_rate"].idxmax()]
    lb = by_basin.loc[by_basin["pr_auc"].idxmin()]
    print(f"largest miss rate by basin: {wb['group']} {wb['miss_rate']:.4f} ({wb['floods']} floods); "
          f"by month (>= 100 floods): {int(wm['group'])} {wm['miss_rate']:.4f} ({int(wm['floods'])} floods)")
    print(f"lowest PR-AUC by basin: {lb['group']} {lb['pr_auc']:.4f}")

    f["outcome"] = np.select([(f.pred == 1) & (f.y == 1), (f.pred == 0) & (f.y == 1),
                              (f.pred == 1) & (f.y == 0)], ["TP", "FN", "FP"], "TN")
    prof = f[f.monsoon_season == 1].groupby("outcome")[MEAS + ["p"]].mean()
    prof["rows"] = f[f.monsoon_season == 1].groupby("outcome").size()
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print("\nmean of today's measurements by outcome, monsoon test rows:\n", prof.to_string())
        cols = ["station_id", "date", "basin"] + MEAS + ["p"]
        print("\n5 worst false negatives (floods with the lowest probability):\n",
              f[f.outcome == "FN"].nsmallest(5, "p")[cols].to_string(index=False))
        print("\n5 worst false positives (no flood, highest probability):\n",
              f[f.outcome == "FP"].nlargest(5, "p")[cols].to_string(index=False))
    fn, edge = f[f.outcome == "FN"], f["month"].isin([5, 6, 10])
    print(f"\nfalse negatives: {len(fn)}; in May, June or October: {int(fn.month.isin([5, 6, 10]).sum())} "
          f"({fn.month.isin([5, 6, 10]).mean():.4f}); those months hold {int(f.y[edge].sum())} of "
          f"{int(f.y.sum())} test floods ({f.y[edge].sum() / f.y.sum():.4f})")
    tr_m = tr["frame"]["date"].dt.month
    print(f"train floods in June/October: {int(tr['y'][tr_m.isin([6, 10])].sum())} of {int(tr['y'].sum())} "
          f"({tr['y'][tr_m.isin([6, 10])].sum() / tr['y'].sum():.4f}); "
          f"in July-September: {tr['y'][tr_m.isin([7, 8, 9])].sum() / tr['y'].sum():.4f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    xb = np.arange(len(by_basin))
    a.bar(xb - 0.2, by_basin["miss_rate"], 0.4, label="miss rate (1 - recall)")
    a.bar(xb + 0.2, by_basin["precision"], 0.4, label="precision of the alarms")
    a.set_xticks(xb, by_basin["group"], rotation=45, ha="right")
    a.set_ylim(0, 0.42)
    finish(a, title="(a) By basin", xlabel="basin", ylabel="share (0-1)")
    mm = by_month[by_month["floods"] > 0]
    xm = np.arange(len(mm))
    b.bar(xm - 0.2, mm["miss_rate"], 0.4, label="miss rate (1 - recall)")
    b.bar(xm + 0.2, mm["precision"], 0.4, label="precision of the alarms")
    b.set_xticks(xm, [f"{int(m)} ({n})" for m, n in zip(mm["group"], mm["floods"])])
    b.set_ylim(0, 1.35)
    finish(b, title="(b) By month (months with floods; flood count in brackets)", xlabel="month (floods)", ylabel="share (0-1)")
    fig.suptitle("Figure 6.1 - Errors of the MLP by subgroup, test set, threshold from validation")
    savefig(fig, "fig61_subgroups")

    # ---------------------------------------------------------------- step 6: permutation importance
    names = d["names"]
    groups = {n.replace("num__", ""): [i] for i, n in enumerate(names) if n.startswith("num__")}
    groups["basin (10 one-hot)"] = [i for i, n in enumerate(names) if n.startswith("cat__basin")]
    groups["season (4 one-hot)"] = [i for i, n in enumerate(names) if n.startswith("cat__season")]
    m42 = models[SEED]
    base_ap = pr_auc(va["y"], predict_proba(m42, va["X"]))
    rng = np.random.default_rng(SEED)
    imp = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(5):
            Xp = va["X"].copy()
            Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]    # shuffle the group's columns together
            drops.append(base_ap - pr_auc(va["y"], predict_proba(m42, Xp)))
        imp[g] = (float(np.mean(drops)), float(np.std(drops, ddof=1)))
    imp = dict(sorted(imp.items(), key=lambda kv: -kv[1][0]))
    print(f"\npermutation importance on VALIDATION (seed-42 MLP, base PR-AUC {base_ap:.4f}, 5 repeats):")
    for g, (m, s) in imp.items():
        print(f"  {g:<24} PR-AUC drop {m:.4f} ± {s:.4f}  ({m / base_ap:.1%} of the PR-AUC)")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = list(imp)[::-1]
    ax.barh(labels, [imp[k][0] for k in labels], xerr=[imp[k][1] for k in labels], color="C0",
            label="mean drop over 5 shuffles (± std)")
    finish(ax, title="Figure 6.2 - Permutation importance (validation, MLP seed 42)",
           xlabel="drop in validation PR-AUC when the feature is shuffled", ylabel="feature (group)")
    savefig(fig, "fig62_permutation_importance")

    # ---------------------------------------------------------------- save
    out = dict(test_rows=int(len(te["y"])), test_positives=int(te["y"].sum()), baselines_test=res,
               mlp_seeds=seed_runs, mlp_mean_std=agg, criterion_met=bool(met),
               by_basin=by_basin.to_dict("records"), by_month=by_month.to_dict("records"),
               permutation_importance_val=imp)
    (HERE / "results_test.json").write_text(json.dumps(out, indent=1, default=float))
    print("[json] results_test.json")


if __name__ == "__main__":
    main()
