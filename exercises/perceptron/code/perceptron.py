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

Pocket-algorithm support (for Exercise 2's non-separable data) is an
optional, purely additive flag: with `pocket=True`, after each epoch the
(w, b) achieving the best accuracy-so-far across the whole run is snapshotted
and, at the end, returned in place of the last epoch's (w, b). This
bookkeeping wraps the epoch loop only -- the per-sample update line above is
executed identically, byte-for-byte, whether pocket is on or off.
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    """Binary perceptron classifier trained with the classic online update rule.

    Set by `fit`:
        w, b          -- final weights / bias (pocket-best if pocket=True, else last epoch's)
        history       -- full-dataset accuracy after each epoch, len == n_epochs
        n_epochs      -- number of epochs actually run
        converged     -- True iff stopped because an epoch had zero updates
        best_accuracy -- best epoch-end accuracy seen over the run (pocket bookkeeping)
    """

    def __init__(self, eta: float = 0.01, max_epochs: int = 100, pocket: bool = False):
        self.eta = eta
        self.max_epochs = max_epochs
        self.pocket = pocket
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.history: list[float] = []
        self.n_epochs: int = 0
        self.converged: bool = False
        self.best_accuracy: float = -1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorized step(w.x + b) over a batch of points; used only for
        accuracy bookkeeping and plotting -- never inside the training loop."""
        return (X @ self.w + self.b >= 0).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> "Perceptron":
        """Train in place. `rng` must be the single shared generator instance
        so weight init consumes from, and stays in sync with, the caller's draw
        order."""
        n_samples, n_features = X.shape
        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0
        self.history = []
        self.converged = False
        best_w, self.best_accuracy = self.w.copy(), -1.0
        best_b = self.b

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

            acc = self.accuracy(X, y)
            self.history.append(acc)
            self.n_epochs = epoch + 1

            if self.pocket and acc > self.best_accuracy:  # pocket bookkeeping only
                self.best_accuracy = acc
                best_w, best_b = self.w.copy(), self.b

            if n_updates == 0:
                self.converged = True
                break

        if self.pocket:
            self.w, self.b = best_w, best_b

        return self


if __name__ == "__main__":
    # Minimal self-check: a trivially separable toy dataset must converge to
    # 100% accuracy well under max_epochs, and pocket=True must not change
    # that outcome (its best-so-far snapshot equals the converged, last-epoch
    # result once accuracy hits 100%).
    _rng = np.random.default_rng(0)
    _X = np.vstack(
        [_rng.normal([0, 0], 0.1, size=(20, 2)), _rng.normal([5, 5], 0.1, size=(20, 2))]
    )
    _y = np.array([0] * 20 + [1] * 20)

    _m = Perceptron(eta=0.1, max_epochs=50).fit(_X, _y, _rng)
    assert _m.converged, "toy separable set should converge"
    assert _m.history[-1] == 1.0, "toy separable set should reach 100% accuracy"
    assert len(_m.history) == _m.n_epochs

    _m_pocket = Perceptron(eta=0.1, max_epochs=50, pocket=True).fit(_X, _y, _rng)
    assert _m_pocket.converged
    assert _m_pocket.best_accuracy == 1.0
    assert np.array_equal(_m_pocket.predict(_X), _y)

    print(f"self-check passed: {_m.n_epochs} epochs, final accuracy {_m.history[-1]:.4f}")
