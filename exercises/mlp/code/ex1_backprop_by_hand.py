"""
Exercise 1 - One step of backpropagation by hand, verified numerically.

Purpose: recompute, in plain numpy, every quantity of the by-hand derivation
(forward pass, backward pass, parameter update) so the report's numbers are
checked against a real computation instead of arithmetic done on paper alone.

Convention: z(1) = W(1) x + b(1), u(2) = W(2) h(1) + b(2). Row i of W(1)
holds the weights feeding hidden unit i, i.e. W(1)[i, j] connects input
feature j to hidden unit i. W(2) is a (2,) row vector (one output unit),
so u(2) is the dot product of W(2) with h(1) plus b(2).

Run from this directory:
    python ex1_backprop_by_hand.py
"""
import numpy as np

# ---- given values -------------------------------------------------------
x = np.array([0.5, -0.2])
y = 1.0

W1 = np.array([[0.3, -0.1],
               [0.2, 0.4]])
b1 = np.array([0.1, -0.2])

W2 = np.array([0.5, -0.3])  # shape (2,): one output unit, two hidden units
b2 = 0.2

eta = 0.3


def p(label, val):
    if isinstance(val, np.ndarray):
        print(f"{label} = {np.round(val, 6).tolist()}")
    else:
        print(f"{label} = {val:.6f}")


# ---- Part A: forward pass ------------------------------------------------
print("=== Part A: forward pass ===")
z1 = W1 @ x + b1
p("z(1)", z1)

h1 = np.tanh(z1)
p("h(1)", h1)

u2 = W2 @ h1 + b2
p("u(2)", u2)

yhat = np.tanh(u2)
p("yhat", yhat)

L = (y - yhat) ** 2
p("L", L)

# ---- Part B: backward pass ------------------------------------------------
print("\n=== Part B: backward pass ===")
dL_dyhat = -2 * (y - yhat)
p("dL/dyhat", dL_dyhat)

dyhat_du2 = 1 - yhat ** 2
dL_du2 = dL_dyhat * dyhat_du2
p("dL/du2", dL_du2)

dL_dW2 = dL_du2 * h1  # dL/du2 is a scalar, h1 has shape (2,)
p("dL/dW2", dL_dW2)

dL_db2 = dL_du2
p("dL/db2", dL_db2)

dL_dh1 = dL_du2 * W2  # backprop through u2 = W2 . h1 + b2
p("dL/dh1", dL_dh1)

dh1_dz1 = 1 - h1 ** 2
dL_dz1 = dL_dh1 * dh1_dz1
p("dL/dz1", dL_dz1)

dL_dW1 = np.outer(dL_dz1, x)  # outer product: shape (2,2), matches W1's shape
p("dL/dW1", dL_dW1)

dL_db1 = dL_dz1
p("dL/db1", dL_db1)

# ---- Part C: parameter update ---------------------------------------------
print("\n=== Part C: parameter update (eta = 0.3) ===")
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1
p("W2_new", W2_new)
p("b2_new", b2_new)
p("W1_new", W1_new)
p("b1_new", b1_new)

# ---- second forward pass with the updated parameters -----------------------
print("\n=== Second forward pass, updated parameters ===")
z1_2 = W1_new @ x + b1_new
h1_2 = np.tanh(z1_2)
u2_2 = W2_new @ h1_2 + b2_new
yhat_2 = np.tanh(u2_2)
L_2 = (y - yhat_2) ** 2
p("z(1)_2", z1_2)
p("h(1)_2", h1_2)
p("u(2)_2", u2_2)
p("yhat_2", yhat_2)
p("L_2", L_2)

print(f"\nLoss went {'down' if L_2 < L else 'up'}: L={L:.6f} -> L_2={L_2:.6f}")

# ---- pin the headline numbers (fails loudly if the derivation breaks) -----
assert round(float(yhat), 4) == 0.3672
assert round(float(L), 4) == 0.4004
assert round(float(W2_new[0]), 4) == 0.5866
assert round(float(W2_new[1]), 4) == -0.3585
assert round(float(b2_new), 4) == 0.5284
assert round(float(W1_new[0, 0]), 4) == 0.3764
assert round(float(W1_new[0, 1]), 4) == -0.1306
assert round(float(W1_new[1, 0]), 4) == 0.1523
assert round(float(W1_new[1, 1]), 4) == 0.4191
assert round(float(b1_new[0]), 4) == 0.2528
assert round(float(b1_new[1]), 4) == -0.2954
assert round(float(L_2), 4) == 0.0834
assert round(float(L_2), 4) < round(float(L), 4)
print("\nAll assertions passed.")
