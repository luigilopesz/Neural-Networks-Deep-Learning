"""The network and its training loop (PyTorch).

Network: an MLP, 21 inputs -> hidden ReLU layers -> 1 logit.
    * ReLU hidden layers with He (Kaiming) initialisation: the init that keeps the activation
      variance constant through ReLU layers; biases start at 0.
    * Optional Dropout after every hidden ReLU (regularisation technique 1).
    * The output is ONE logit (no sigmoid inside the model). The loss is BCEWithLogitsLoss,
      which applies the sigmoid itself in a numerically stable way.

Imbalance: pos_weight = (#negatives / #positives) on the TRAINING split (about 46.3), so a
missed flood costs as much in the loss as ~46 false alarms. Nothing is resampled.

Optimiser: AdamW (Adam with decoupled weight decay = regularisation technique 2).

Early stopping: after every epoch the validation PR-AUC is measured; the best epoch's weights
are kept in memory and restored at the end; training stops after `patience` epochs without
improvement.

Run `python model.py` for a small self-check (shapes, parameter count, determinism).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from common import pr_auc

# One thread + deterministic kernels: the same seed gives the same numbers on every run.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden=(64, 32), dropout: float = 0.0):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            lin = nn.Linear(prev, h)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")   # He init for ReLU
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        out = nn.Linear(prev, 1)
        nn.init.xavier_uniform_(out.weight)                              # no ReLU after it
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)                                   # logits, shape (N,)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def train(X_tr, y_tr, X_va, y_va, *, hidden=(64, 32), dropout=0.0, weight_decay=0.0,
          lr=1e-3, batch=1024, max_epochs=60, patience=8, seed=42, keep_val_probs=False):
    """Train one configuration; return (model with the best weights, history, info).

    history: one dict per epoch with train_loss, val_loss (same weighted BCE) and val_pr_auc.
    info: best_epoch, stopped_epoch, best_val_pr_auc, n_params, pos_weight.
    keep_val_probs=True also stores the validation probabilities of every epoch (for the GIF).
    """
    torch.manual_seed(seed)                       # seeds the init and the dropout masks
    gen = torch.Generator().manual_seed(seed)     # seeds the mini-batch order

    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))
    Xv, yv = torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.float32))
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

    model = MLP(X_tr.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    history, val_probs = [], []
    best, best_state, best_epoch, wait = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for idx in torch.randperm(len(Xt), generator=gen).split(batch):
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        model.eval()                              # dropout off for the measurements
        with torch.no_grad():
            train_loss = loss_fn(model(Xt), yt).item()
            logits_v = model(Xv)
            val_loss = loss_fn(logits_v, yv).item()
            pv = torch.sigmoid(logits_v).numpy()
        ap = pr_auc(y_va, pv)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_pr_auc=ap))
        if keep_val_probs:
            val_probs.append(pv)

        if ap > best:                             # early stopping on validation PR-AUC
            best, best_epoch, wait = ap, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait == patience:
                break

    model.load_state_dict(best_state)             # restore the best weights
    info = dict(best_epoch=best_epoch, stopped_epoch=epoch, best_val_pr_auc=best,
                n_params=n_params(model), pos_weight=pos_weight)
    if keep_val_probs:
        info["val_probs"] = val_probs
    return model, history, info


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 21)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=2000) > 1.6).astype(int)
    m = MLP(21, (64, 32))
    print("parameters of 21-64-32-1:", n_params(m), "(= 21*64+64 + 64*32+32 + 32+1 = 3521)")
    assert n_params(m) == 3521
    a = train(X[:1500], y[:1500], X[1500:], y[1500:], max_epochs=5)[2]["best_val_pr_auc"]
    b = train(X[:1500], y[:1500], X[1500:], y[1500:], max_epochs=5)[2]["best_val_pr_auc"]
    print(f"same seed, two runs: {a:.6f} {b:.6f}")
    assert a == b
    print("ok")
