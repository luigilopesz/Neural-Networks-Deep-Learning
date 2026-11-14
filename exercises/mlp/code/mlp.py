"""MLP -- from-scratch, shared implementation (Exercises 2, 3 & 4).

No scikit-learn or third-party model -- numpy only. Arbitrary depth: `layer_sizes`
is `[n_in, n_hidden_1, ..., n_hidden_k, n_out]`, tanh on every hidden layer, and
either a sigmoid or a softmax output.

Architecture, per weight matrix `l` (`W[l]` shape `(n_in, n_out)`, `b[l]` shape `(1, n_out)`):
    z[l] = h[l-1] @ W[l] + b[l]
    h[l] = tanh(z[l])                       (hidden layers only)
    P    = sigmoid(z[L]) or softmax(z[L])   (last layer)
Init: Xavier-uniform, U(-a, a) with a = sqrt(6 / (n_in + n_out)), drawn from `rng`;
biases start at zero.

Why `dL/dU = (P - Y) / N` for BOTH output types (`U` = output pre-activation):

sigmoid + binary cross-entropy, per sample (Y in {0, 1}, P = sigmoid(U)):
    dL/dP = -(Y/P) + (1-Y)/(1-P) = (P - Y) / (P(1-P))
    dP/dU = P(1-P)                                          (sigmoid derivative)
    dL/dU = dL/dP * dP/dU = P - Y                            -> average over N: (P-Y)/N

softmax + categorical cross-entropy, per sample (Y one-hot, P = softmax(U)):
    L = -sum_k Y_k log(P_k),  dP_k/dU_j = P_k(delta_kj - P_j)   (softmax Jacobian)
    dL/dU_j = sum_k dL/dP_k * dP_k/dU_j = P_j - Y_j           -> average over N: (P-Y)/N
Both reduce to the same expression, so `backward` needs no branch on output type:
one chain of matmuls with `W` and the tanh derivative handles either case.
"""

from __future__ import annotations

import numpy as np


class MLP:
    """Multi-layer perceptron trained with plain (mini-batch or full-batch) gradient descent.

    Set by `__init__`:
        W, b     -- list of weight matrices / bias rows, one pair per layer transition
        output   -- "sigmoid" (binary, BCE) or "softmax" (multi-class, CCE)
        rng      -- shared generator, consumed by init and by `fit`'s mini-batch shuffling
    Set by `forward` (consumed by `backward`, overwritten on the next call):
        _h_cache -- [X, h after each hidden layer], _P_cache, _n_cache -- batch size used
    """

    _EPS = 1e-12

    def __init__(
        self,
        layer_sizes: list[int],
        output: str = "sigmoid",
        rng: np.random.Generator | None = None,
    ) -> None:
        if output not in ("sigmoid", "softmax"):
            raise ValueError(f"output must be 'sigmoid' or 'softmax', got {output!r}")
        self.layer_sizes = list(layer_sizes)
        self.output = output
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.n_layers = len(layer_sizes) - 1  # number of weight matrices

        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            a = np.sqrt(6.0 / (n_in + n_out))  # Xavier-uniform bound
            self.W.append(self.rng.uniform(-a, a, size=(n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))

        self._h_cache: list[np.ndarray] = []
        self._P_cache: np.ndarray | None = None
        self._n_cache: int = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """One forward pass; caches every hidden activation and the output
        probabilities for `backward`."""
        h = np.asarray(X, dtype=np.float64)
        self._h_cache = [h]
        for l in range(self.n_layers - 1):  # hidden layers
            z = h @ self.W[l] + self.b[l]
            h = np.tanh(z)
            self._h_cache.append(h)

        z_out = h @ self.W[-1] + self.b[-1]
        if self.output == "sigmoid":
            P = 1.0 / (1.0 + np.exp(-np.clip(z_out, -500, 500)))  # avoid exp overflow
        else:
            z_shift = z_out - z_out.max(axis=1, keepdims=True)  # numerically stable softmax
            exp_z = np.exp(z_shift)
            P = exp_z / exp_z.sum(axis=1, keepdims=True)

        self._P_cache = P
        self._n_cache = h.shape[0]
        return P

    def loss(self, probs: np.ndarray, Y: np.ndarray) -> float:
        """Mean BCE (sigmoid) or mean CCE (softmax) over the batch, eps-clipped."""
        P = np.clip(probs, self._EPS, 1.0 - self._EPS)
        if self.output == "sigmoid":
            return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))
        return float(-np.mean(np.sum(Y * np.log(P), axis=1)))

    def backward(self, Y: np.ndarray) -> dict[str, list[np.ndarray]]:
        """Hand-derived gradients from the cache left by the last `forward` call.
        Same code path for sigmoid+BCE and softmax+CCE -- see module docstring."""
        N = self._n_cache
        dU = (self._P_cache - Y) / N  # dL/dU, identical for both output types

        grads_W: list[np.ndarray | None] = [None] * self.n_layers
        grads_b: list[np.ndarray | None] = [None] * self.n_layers

        h_prev = self._h_cache[-1]  # last hidden activation (or X if no hidden layer)
        grads_W[-1] = h_prev.T @ dU
        grads_b[-1] = dU.sum(axis=0, keepdims=True)
        dh = dU @ self.W[-1].T  # dL/dh = dU @ W.T, propagate to the layer below

        for l in range(self.n_layers - 2, -1, -1):
            h_l = self._h_cache[l + 1]  # cached tanh activation at this layer
            dz = dh * (1 - h_l**2)  # tanh derivative from the cached activation
            h_prev = self._h_cache[l]
            grads_W[l] = h_prev.T @ dz
            grads_b[l] = dz.sum(axis=0, keepdims=True)
            if l > 0:
                dh = dz @ self.W[l].T

        return {"W": grads_W, "b": grads_b}

    def update(self, grads: dict[str, list[np.ndarray]], lr: float) -> None:
        """Plain gradient descent step: no momentum, no regularisation."""
        for l in range(self.n_layers):
            self.W[l] -= lr * grads["W"][l]
            self.b[l] -= lr * grads["b"][l]

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int,
        lr: float,
        batch_size: int | None = None,
    ) -> list[float]:
        """Train in place. `batch_size=None` is full-batch GD (one forward/backward/
        update per epoch). An integer `batch_size` shuffles indices with `self.rng`
        each epoch and steps per mini-batch. Either way the recorded loss is
        `loss(forward(X), Y)` on the full training set, once per epoch."""
        N = X.shape[0]
        losses: list[float] = []
        for _ in range(epochs):
            if batch_size is None:
                self.forward(X)
                grads = self.backward(Y)
                self.update(grads, lr)
            else:
                order = self.rng.permutation(N)
                for start in range(0, N, batch_size):
                    idx = order[start : start + batch_size]
                    self.forward(X[idx])
                    grads = self.backward(Y[idx])
                    self.update(grads, lr)
            losses.append(self.loss(self.forward(X), Y))
        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Threshold 0.5 for sigmoid, argmax for softmax; returns integer labels."""
        P = self.forward(X)
        if self.output == "sigmoid":
            return (P >= 0.5).astype(int).ravel()
        return np.argmax(P, axis=1)

    def n_params(self) -> int:
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)


if __name__ == "__main__":
    # Self-check: central finite-difference gradient check, both output types.
    # Every entry of every analytic W/b gradient must match the numeric one.
    def _max_rel_error(model: MLP, X: np.ndarray, Y: np.ndarray, h: float = 1e-6) -> float:
        model.forward(X)
        analytic = model.backward(Y)
        worst = 0.0
        for params, grads in ((model.W, analytic["W"]), (model.b, analytic["b"])):
            for p, g in zip(params, grads):
                it = np.nditer(p, flags=["multi_index"])
                for _ in it:
                    idx = it.multi_index
                    orig = p[idx]
                    p[idx] = orig + h
                    loss_plus = model.loss(model.forward(X), Y)
                    p[idx] = orig - h
                    loss_minus = model.loss(model.forward(X), Y)
                    p[idx] = orig
                    numeric = (loss_plus - loss_minus) / (2 * h)
                    rel_err = abs(g[idx] - numeric) / max(abs(g[idx]), abs(numeric), 1e-12)
                    worst = max(worst, rel_err)
        return worst

    _rng = np.random.default_rng(0)
    _X = _rng.normal(size=(5, 3))

    _sig = MLP([3, 4, 1], output="sigmoid", rng=_rng)
    _y_sig = _rng.integers(0, 2, size=(5, 1)).astype(np.float64)
    _err_sig = _max_rel_error(_sig, _X, _y_sig)

    _soft = MLP([3, 4, 3], output="softmax", rng=_rng)
    _labels = _rng.integers(0, 3, size=5)
    _y_soft = np.eye(3)[_labels]
    _err_soft = _max_rel_error(_soft, _X, _y_soft)

    print(f"max relative error (sigmoid): {_err_sig:.3e}")
    print(f"max relative error (softmax): {_err_soft:.3e}")
    assert _err_sig < 1e-6, f"sigmoid gradient check failed: {_err_sig:.3e}"
    assert _err_soft < 1e-6, f"softmax gradient check failed: {_err_soft:.3e}"
    print("ok")
