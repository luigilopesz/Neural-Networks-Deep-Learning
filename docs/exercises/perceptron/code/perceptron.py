"""Perceptron -- single, from-scratch, shared implementation (Exercises 1 & 2).

No scikit-learn or third-party model -- numpy only.

Prediction:  yhat = step(w . x + b),  step(z) = 1 if z >= 0 else 0.
Update (online, per-sample, NOT batched):
    w <- w + eta * (y - yhat) * x
    b <- b + eta * (y - yhat)
Init:  w = rng.normal(0, 0.01, size=n_features);  b = 0.
Stopping: a full epoch with zero updates, OR `max_epochs` epochs -- whichever
first. Full-dataset accuracy at the current (w, b) is recorded after every
epoch (`history`).

Pocket (the one thing Exercise 2 adds to the loop): every time an update
produces a higher full-dataset accuracy than any previously seen, (w, b) is
copied into the pocket. The test runs right after EACH UPDATE, not once per
epoch, so a good line that exists for only a few samples in the middle of an
epoch is still caught. The pocket only READS (w, b): the update lines are
executed identically with or without it, so the trajectory (and every Exercise 1
number) is the same. `w`, `b` always hold the FINAL weights (the state after the
last epoch) and the pocket is kept apart in `pocket_w`, `pocket_b`, so a single
`fit` returns both sets -- the same class, called the same way, in both exercises.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Update(NamedTuple):
    """One entry of `Perceptron.log`: what a single update did."""

    epoch: int  # 1-based epoch in which it happened
    sample: int  # index of the sample that caused it
    error: int  # y - yhat: -1 = a false positive was corrected, +1 = a false negative
    acc: float  # full-dataset accuracy right after the update
    w: np.ndarray  # weights right after the update
    b: float  # bias right after the update


class Perceptron:
    """Binary perceptron classifier trained with the classic online update rule.

    Set by `fit`:
        w, b           -- FINAL weights / bias (the state after the last epoch)
        pocket_w/_b    -- pocket weights / bias: the (w, b) with the best full-dataset
                          accuracy seen right after any update (the initial weights
                          count as "seen" too, so the pocket is never worse than the start)
        pocket_acc     -- that best accuracy
        pocket_epoch   -- 1-based epoch in which the pocket was last updated (0 = initial weights)
        history        -- full-dataset accuracy of (w, b) after each epoch, len == n_epochs
        pocket_history -- pocket accuracy at the end of each epoch (best so far)
        updates        -- number of updates made in each epoch
        log            -- one `Update` (epoch, sample, error, accuracy, w, b after it) per update
        n_epochs       -- number of epochs actually run
        converged      -- True iff stopped because an epoch had zero updates
    """

    def __init__(self, eta: float = 0.01, max_epochs: int = 100):
        self.eta = eta
        self.max_epochs = max_epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.pocket_w: np.ndarray | None = None
        self.pocket_b: float = 0.0
        self.pocket_acc: float = -1.0
        self.pocket_epoch: int = 0
        self.history: list[float] = []
        self.pocket_history: list[float] = []
        self.updates: list[int] = []
        self.log: list[Update] = []
        self.n_epochs: int = 0
        self.converged: bool = False

    def predict(self, X: np.ndarray, w: np.ndarray | None = None, b: float | None = None) -> np.ndarray:
        """Vectorized step(w.x + b) over a batch of points (default: the final w, b);
        used only for accuracy bookkeeping and plotting -- never to decide an update."""
        w = self.w if w is None else w
        b = self.b if b is None else b
        return (X @ w + b >= 0).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, b: float | None = None) -> float:
        return float(np.mean(self.predict(X, w, b) == y))

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> "Perceptron":
        """Train in place. `rng` must be the single shared generator instance
        so weight init consumes from, and stays in sync with, the caller's draw
        order."""
        n_samples, n_features = X.shape
        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0
        self.history, self.pocket_history, self.updates, self.log = [], [], [], []
        self.converged = False
        # The pocket starts as the initial weights.
        self.pocket_w, self.pocket_b = self.w.copy(), self.b
        self.pocket_acc, self.pocket_epoch = self.accuracy(X, y), 0

        for epoch in range(self.max_epochs):
            n_updates = 0
            for i in range(n_samples):
                xi, yi = X[i], y[i]
                yhat = 1 if (self.w @ xi + self.b) >= 0 else 0
                error = yi - yhat
                if error != 0:
                    self.w = self.w + self.eta * error * xi
                    self.b = self.b + self.eta * error
                    n_updates += 1

                    # Pocket: right after the update, score the new (w, b) on the full dataset.
                    acc = self.accuracy(X, y)
                    self.log.append(Update(epoch + 1, i, int(error), acc, self.w.copy(), self.b))
                    if acc > self.pocket_acc:  # strictly higher than any accuracy seen so far
                        self.pocket_w, self.pocket_b = self.w.copy(), self.b
                        self.pocket_acc, self.pocket_epoch = acc, epoch + 1

            self.history.append(self.accuracy(X, y))
            self.pocket_history.append(self.pocket_acc)
            self.updates.append(n_updates)
            self.n_epochs = epoch + 1

            if n_updates == 0:
                self.converged = True
                break

        return self


if __name__ == "__main__":
    # Minimal self-check: a trivially separable toy set must converge to 100% and
    # leave the pocket equal to the final weights; on an overlapping toy set the
    # pocket must be exactly the best accuracy seen at the start or after any update.
    _rng = np.random.default_rng(0)
    _X = np.vstack(
        [_rng.normal([0, 0], 0.1, size=(20, 2)), _rng.normal([5, 5], 0.1, size=(20, 2))]
    )
    _y = np.array([0] * 20 + [1] * 20)

    _m = Perceptron(eta=0.1, max_epochs=50).fit(_X, _y, _rng)
    assert _m.converged, "toy separable set should converge"
    assert _m.history[-1] == 1.0, "toy separable set should reach 100% accuracy"
    assert len(_m.history) == _m.n_epochs == len(_m.updates) == len(_m.pocket_history)
    assert sum(_m.updates) == len(_m.log)
    assert _m.pocket_acc == 1.0 and np.array_equal(_m.pocket_w, _m.w) and _m.pocket_b == _m.b

    _Xo = np.vstack(
        [_rng.normal([0, 0], 1.0, size=(50, 2)), _rng.normal([1, 1], 1.0, size=(50, 2))]
    )
    _yo = np.array([0] * 50 + [1] * 50)
    _o = Perceptron(eta=0.1, max_epochs=30).fit(_Xo, _yo, np.random.default_rng(1))
    _init_acc = _o.accuracy(_Xo, _yo, np.random.default_rng(1).normal(0, 0.01, size=2), 0.0)
    assert not _o.converged and _o.n_epochs == 30, "overlapping set must run the full cap"
    assert _o.pocket_acc == max([_init_acc] + [u.acc for u in _o.log])
    assert _o.accuracy(_Xo, _yo, _o.pocket_w, _o.pocket_b) == _o.pocket_acc
    assert _o.pocket_acc >= max(_o.history)

    print(f"self-check passed: separable {_m.n_epochs} epochs, final accuracy {_m.history[-1]:.4f}; "
          f"overlapping pocket {_o.pocket_acc:.2f} (epoch {_o.pocket_epoch}) vs final {_o.history[-1]:.2f}")
