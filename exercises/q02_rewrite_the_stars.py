"""Exercise 02 — the "star operation", built on the engine.

YOUR TASK
=========
Build a few small layers as ``nn.Module`` subclasses that use the **star
operation** — element-wise multiplication of two learned affine branches — and
compare it with the usual additive form. You implement only ``forward``: because
each layer is composed out of ops the engine already differentiates (``@``,
``+``, ``*``, ``relu`` ...), **the engine gives you the backward pass for free.**

This is the key contrast with ``exercises/q01_activations.py``: there you added a
*brand-new* op and so had to hand-write its ``_backward`` (the local derivative).
Here you add *no* new op — you only compose existing ones — so there is no
backward to write. The autograd graph (each result tensor remembering its
operands in ``_prev``) takes care of it. The gradient check at the bottom proves
it: the analytic gradients from ``backward()`` match finite differences, even
though you never wrote a derivative.

The maths
---------
A traditional two-branch layer *adds* the branches::

    u = W1ᵀ @ [1; x]            (branch 1, the bias folded into row 0)
    v = W2ᵀ @ [1; x]            (branch 2)
    y = act(u) + v             (SumLinear)

The "star" form *multiplies* them element-wise::

    y = act(u) * v             (StarLinear)

Expanding one output unit [1], ``(w1ᵀx + b1)(w2ᵀx + b2)`` contains terms
``x_i x_j``, ``x_i`` and constants — so the star layer represents implicit
second-order feature interactions without ever building the ``x_i x_j`` pairs.

Required reading
----------------
[1] Ma, X., Dai, X., Bai, Y., Wang, Y., & Fu, Y. (2024). "Rewrite the Stars."
CVPR 2024, pp. 5694-5703.


Conventions
-----------
Like the rest of the library, inputs are **column-oriented**: ``x`` has shape
``(in_features, batch)`` and each ``nn.Linear`` computes ``Wᵀ @ [1; x]`` (the
bias-trick). So ``u``, ``v`` and ``y`` have shape ``(out_features, batch)`` and
``*`` multiplies them element-wise.

HOW TO WORK
===========
1. Fill in the four ``forward`` methods (remove the ``raise NotImplementedError``).
2. Run the file to grade yourself — it gradient-checks your layers against finite
   differences and runs a tiny demo of the star op's second-order behaviour::

       python -m exercises.q02_rewrite_the_stars

TIP: a layer is just another graph. Look at ``bert_cpu/nn.py::Linear`` — its
``forward`` builds the output from engine ops and returns it; nothing else. Yours
do the same, just combining two branches with ``+`` or ``*``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from bert_cpu import engine as cpu
from bert_cpu import nn


# An activation is just a function Tensor -> Tensor; ReLU by default. Pass
# ``lambda t: t`` for the identity (handy for the gradient check).
Activation = Callable[[cpu.Tensor], cpu.Tensor]
_RELU: Activation = lambda t: t.relu()
_IDENTITY: Activation = lambda t: t


# ============================================================================ #
# Exercise 1 — additive two-branch layer
# ============================================================================ #
class SumLinear(nn.Module):
    """Two affine branches, combined by addition: ``y = act(W1ᵀ @ [1;x]) + W2ᵀ @ [1;x]``.

    The two ``nn.Linear`` branches are stored as attributes, so ``Module``
    collects their parameters automatically (``parameters()`` returns both
    weights).

    Questions to ponder
    -------------------
    1. With the identity activation, can the two branches be collapsed into one
       ``Linear``? (What does that say about additive fusion?)
    2. What does ReLU on one branch buy you?
    """

    def __init__(self, in_features: int, out_features: int, activation: Activation = _RELU) -> None:
        self.activation = activation
        self.branch1 = nn.Linear(in_features, out_features)
        self.branch2 = nn.Linear(in_features, out_features)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # TODO: 
        raise NotImplementedError("implement SumLinear.forward")


# ============================================================================ #
# Exercise 2 — the star-operation layer
# ============================================================================ #
class StarLinear(nn.Module):
    """Two affine branches, combined by element-wise product (the star op):

        ``y = act(W1ᵀ @ [1;x]) * (W2ᵀ @ [1;x])``

    You do not implement a backward pass. ``*`` (``Tensor.__mul__``) already
    carries the product rule, ``@`` carries the matmul rule, and so on — so the
    engine differentiates ``y`` for you. The product rule it applies is exactly
    ``dL/da = dL/dy * v`` and ``dL/dv = dL/dy * a`` (with ``a = act(u)``).
    """

    def __init__(self, in_features: int, out_features: int, activation: Activation = _RELU) -> None:
        self.activation = activation
        self.branch1 = nn.Linear(in_features, out_features)
        self.branch2 = nn.Linear(in_features, out_features)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # TODO: 
        raise NotImplementedError("implement StarLinear.forward")


# ============================================================================ #
# Exercise 3 — the "identity branch" special case
# ============================================================================ #
class IdentityStarLinear(nn.Module):
    """One transformed branch times the input itself: ``y = act(W ᵀ @ [1;x]) * x``.

    Requires ``out_features == in_features`` so the shapes line up for ``*``.
    Note that ``x`` reaches ``y`` through *two* paths (directly, and through the
    transformed branch); the engine sums both contributions into ``x.grad``
    automatically — you do not manage that.
    """

    def __init__(self, features: int, activation: Activation = _RELU) -> None:
        self.activation = activation
        self.linear = nn.Linear(features, features)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # TODO: 
        raise NotImplementedError("implement IdentityStarLinear.forward")


# ============================================================================ #
# Exercise 4 — the simplest star: square the features
# ============================================================================ #
class SquareFeature(nn.Module):
    """The simplest star case, ``y = x * x`` — no parameters, but non-linear.

    With ``parameters()`` empty, this layer is pure structure; it still produces
    a differentiable node (``x * x``) whose gradient ``2x`` the engine derives.
    """

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # TODO: 
        raise NotImplementedError("implement SquareFeature.forward")


# ============================================================================ #
# GIVEN — you do not edit below. Once your forwards work, this grades them.
# ============================================================================ #
def numeric_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a scalar ``f()`` w.r.t. the array ``x``.

    Identical in spirit to ``test/test_engine.py::numeric_gradient``: it knows
    nothing about the engine, so agreeing with ``backward()`` is real evidence
    the layer is correct. ``f`` must re-read ``x`` (the same array object we
    perturb) on every call.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        original = x[i]
        x[i] = original + eps
        plus = f()
        x[i] = original - eps
        minus = f()
        x[i] = original
        grad[i] = (plus - minus) / (2.0 * eps)
        it.iternext()
    return grad


def gradient_check(layer: nn.Module, x: cpu.Tensor, target: cpu.Tensor) -> float:
    """Backward() vs finite differences for ``loss = mean((layer(x) - target)**2)``.

    Returns the largest absolute mismatch over the input and every parameter.
    """
    def loss() -> cpu.Tensor:
        return ((layer(x) - target) ** 2).mean()

    for t in (x, *layer.parameters()):
        t.zero_grad()
    loss().backward()

    worst = 0.0
    for name, t in [("x", x), *layer.named_parameters()]:
        numeric = numeric_gradient(lambda: float(loss().data), t.data)
        err = float(np.max(np.abs(numeric - t.grad)))
        worst = max(worst, err)
        print(f"    {name:16s} max|analytic - numeric| = {err:.2e}")
    return worst


def grade() -> None:
    """Gradient-check every layer (identity activation keeps it smooth)."""
    cpu.set_seed(0)
    n_in, n_out, batch = 3, 2, 4
    x = cpu.Tensor(np.random.randn(n_in, batch))
    y2 = cpu.Tensor(np.random.randn(n_out, batch))
    yk = cpu.Tensor(np.random.randn(n_in, batch))   # square-shaped target

    cases = [
        ("SumLinear",         SumLinear(n_in, n_out, activation=_IDENTITY),  x, y2),
        ("StarLinear",        StarLinear(n_in, n_out, activation=_IDENTITY), x, y2),
        ("IdentityStarLinear", IdentityStarLinear(n_in, activation=_IDENTITY), x, yk),
        ("SquareFeature",     SquareFeature(),                                x, yk),
    ]
    print("Gradient check (analytic backward vs finite differences):\n")
    all_ok = True
    for name, layer, xin, target in cases:
        print(f"  {name}:")
        worst = gradient_check(layer, xin, target)
        ok = worst < 1e-5
        all_ok = all_ok and ok
        print(f"    -> {'PASS' if ok else 'FAIL'} (worst {worst:.2e})\n")
    print("All layers correct — and you wrote no backward pass!"
          if all_ok else "Some layers are off; revisit their forward().")


def demonstrate_second_order() -> None:
    """Show the star op carries an ``x_i x_j`` term that the sum op cannot.

    With the identity activation and a single output, ``StarLinear`` computes
    ``(w1ᵀx + b1)(w2ᵀx + b2)``. We freeze a tiny example and read off the mixed
    second-order coefficient via a second finite difference,
    ``∂²y / ∂x0 ∂x1`` — non-zero for the star op, exactly zero for the sum op.
    """
    cpu.set_seed(1)
    star = StarLinear(2, 1, activation=_IDENTITY)
    summ = SumLinear(2, 1, activation=_IDENTITY)

    def mixed_second_derivative(layer: nn.Module, h: float = 1e-3) -> float:
        def f(a: float, b: float) -> float:
            xv = cpu.Tensor(np.array([[a], [b]], dtype=float))
            return float(layer(xv).data.reshape(()))
        # ∂²/∂a∂b via the standard 4-point stencil.
        return (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)

    print("\nSecond-order interaction  ∂²y / ∂x0∂x1  (identity activation):")
    print(f"    StarLinear : {mixed_second_derivative(star):+.4f}   (non-zero -> learns x0*x1)")
    print(f"    SumLinear  : {mixed_second_derivative(summ):+.4f}   (zero      -> stays additive)")


def main() -> None:
    print("=" * 70)
    print("Rewrite the Stars — star operation on the bert_cpu engine")
    print("=" * 70 + "\n")
    grade()
    demonstrate_second_order()


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"Implement the forwards first ({exc}).")
