"""Shared helpers for the classification deliverable: data, metrics and figures.

The data layer is the EDA's, imported unchanged:
    data.py     -> load_raw, add_next_day_target, split_frames (target, chronological split)
    pipeline.py -> fit_preprocessor, transform, feature_names (8 raw columns -> 21 features)

The pipeline is fitted on the TRAINING split only (2000-2016). Validation (2017-2020) and
test (2021-2024) are only transformed.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # docs/projects/classification/code
EDA_CODE = HERE.parents[1] / "eda" / "code"            # docs/projects/eda/code
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(EDA_CODE))

# The EDA modules pin BLAS/OMP to one thread on import (data.configure_env).
from data import (CATEGORICAL, NUMERIC, SEED, TARGET_NEXT, add_next_day_target,  # noqa: E402
                  finish, load_raw, split_frames)
from pipeline import feature_names, fit_preprocessor, transform  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score  # noqa: E402

RAW = NUMERIC + CATEGORICAL          # the 8 raw columns the pipeline expects
TARGET_RECALL = 0.80                 # operating point, fixed before training


def load_splits(names=("train", "validation")) -> dict:
    """Return {split: dict(frame, X, y)} for the requested splits.

    `frame` keeps station_id, date and the raw columns (for error analysis);
    `X` is the 21-column float32 matrix from the EDA pipeline; `y` is the 0/1 next-day label.
    The pipeline is always fitted on the training split, whatever splits are requested.
    """
    df = split_frames(add_next_day_target(load_raw()))
    train = df[df["split"] == "train"]
    pre = fit_preprocessor(train[RAW])                       # fit on TRAIN ONLY
    out = {"names": feature_names(pre)}
    for name in names:
        part = df[df["split"] == name].reset_index(drop=True)
        out[name] = dict(frame=part,
                         X=transform(pre, part[RAW]).astype(np.float32),
                         y=part[TARGET_NEXT].astype(int).to_numpy())
    return out


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def pr_auc(y, p) -> float:
    """PR-AUC as average precision (the area under the precision-recall step curve)."""
    return float(average_precision_score(y, p))


def scores(y, p, monsoon) -> dict:
    """The ranking metrics: PR-AUC (headline), ROC-AUC, and PR-AUC inside the monsoon only."""
    m = monsoon == 1
    return dict(pr_auc=pr_auc(y, p), roc_auc=float(roc_auc_score(y, p)),
                pr_auc_monsoon=pr_auc(y[m], p[m]))


def threshold_for_recall(y, p, target=TARGET_RECALL) -> float:
    """The HIGHEST threshold whose recall is still >= target (most precision for that recall)."""
    _, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]     # rec[i] is the recall of the rule p >= thr[i]
    return float(thr[ok[-1]])


def at_threshold(y, p, t) -> dict:
    """Confusion counts, precision, recall and F1 for both classes, macro F1, at p >= t."""
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())

    def prf(tp_, fp_, fn_):
        p_ = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r_ = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return p_, r_, (2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)

    p1, r1, f1 = prf(tp, fp, fn)          # class 1 = flood
    p0, r0, f0 = prf(tn, fn, fp)          # class 0 = no flood
    return dict(threshold=float(t), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=p1, recall=r1, f1=f1, precision_0=p0, recall_0=r0, f1_0=f0,
                macro_f1=(f1 + f0) / 2, accuracy=(tp + tn) / len(y))


# ----------------------------------------------------------------------------------------------
# Figures (numbered, titled, labelled axes, legend: the statement's rule)
# ----------------------------------------------------------------------------------------------
def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] figures/{name}.png")


def fmt(d: dict, keys=("pr_auc", "roc_auc", "pr_auc_monsoon")) -> str:
    return "  ".join(f"{k}={d[k]:.4f}" for k in keys)
