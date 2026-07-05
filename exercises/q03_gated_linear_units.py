"""Exercise 03 — Gated Linear Units (GLU & friends), built on the engine.

YOUR TASK
=========
Build the GLU family as ``nn.Module`` subclasses. A *gate* is a second learned
branch that multiplies the first one element-wise, deciding how much of each
feature passes through::

    content = W_cᵀ @ [1; x]
    gate    = σ(W_gᵀ @ [1; x])          # σ = sigmoid, values in (0, 1)
    y       = content * gate

As in the other exercises you implement **only ``forward``**. Every piece is an
op the engine already differentiates (``@``, ``*``, ``exp`` ...), so once the
forward is built the engine produces the backward pass for you — no derivative to
write. The gradient check at the bottom (reused from Exercise 02) proves it.

Relation to the other exercises
-------------------------------
- ``exercises/q01_activations.py`` (Ex. 01): there you *added* ``sigmoid`` and
  ``swish`` as primitive ops with hand-written ``_backward``. Here the
  ``sigmoid`` gate is instead **composed** from engine ops (no ``_backward``
  needed). The ``SwiGLU`` class goes the other way and **reuses the ``swish`` you
  implemented in Ex. 01** as its gate — closing the loop between the two files.
- ``exercises/q02_rewrite_the_stars.py`` (Ex. 02): the ``BilinearUnit`` below
  (``left * right``, two unrestricted branches) is exactly ``StarLinear`` with
  the identity activation. A GLU is the same "star" shape, but with one branch
  squashed into ``(0, 1)`` so it acts as a gate. We even reuse that file's
  ``gradient_check``.

The maths (backward, for understanding — you do not code it)
-----------------------------------------------------------
With ``y = c * g`` the engine applies the product rule it already knows:
``dL/dc = dL/dy * g`` and ``dL/dg = dL/dy * c``; the sigmoid/tanh/... factors come
from those ops' own backward rules, chained automatically.

Required reading
----------------
[1] Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). "Language Modeling
    with Gated Convolutional Networks." ICML 2017, PMLR 70, pp. 933-941.
[2] Shazeer, N. (2020). "GLU Variants Improve Transformer." (ReGLU/GEGLU/SwiGLU.)

Conventions
-----------
Column-oriented, like the rest of the library: ``x`` is ``(in_features, batch)``,
each ``nn.Linear`` computes ``Wᵀ @ [1; x]``, and ``*`` is element-wise.

HOW TO WORK
===========
1. Fill in the ``forward`` methods (remove each ``raise NotImplementedError``).
2. Run the file to grade yourself and see the gates in action::

       python -m exercises.q03_gated_linear_units

TIP: the gate activations you need are given just below as plain functions
``Tensor -> Tensor``. Your forwards only have to combine two ``nn.Linear``
branches with ``*`` — look at ``StarLinear`` in Exercise 02 for the shape.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from bert_cpu import engine as cpu
from bert_cpu import nn

# Reuse Exercise 02's finite-difference gradient checker so the exercises share a
# single, well-documented harness (it knows nothing about which layer it checks).
from exercises.q02_rewrite_the_stars import gradient_check
# The SwiGLU below gates with the ``swish`` YOU implement in Exercise 01.
from exercises.q01_activations import ExTensor


# ---------------------------------------------------------------------------- #
# Gate activations, as compositions of existing engine ops (GIVEN).
# These need no hand-written backward: the engine differentiates exp/*/+ for us.
# (Contrast exercises/q01_activations.py, where you add them as fused primitive ops.)
# ---------------------------------------------------------------------------- #
Activation = Callable[[cpu.Tensor], cpu.Tensor]


def sigmoid(t: cpu.Tensor) -> cpu.Tensor:
    """Logistic sigmoid σ(t) = 1 / (1 + e^{-t}), built from engine ops only."""
    return 1.0 / (1.0 + (-t).exp())


# ============================================================================ #
# Exercise 1 — the Gated Linear Unit
# ============================================================================ #
class GatedLinearUnit(nn.Module):
    """GLU: a linear *content* branch gated by a sigmoid branch.

        ``y = (W_cᵀ @ [1;x]) * σ(W_gᵀ @ [1;x])``

    The content branch stays linear (a direct, un-squashed path for gradients,
    the paper's main argument), while the sigmoid gate in ``(0, 1)`` attenuates
    or preserves each content feature.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.content = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # TODO: 
        raise NotImplementedError("implement GatedLinearUnit.forward")


# ============================================================================ #
# Exercise 2 — related gated/bilinear units
# ============================================================================ #
class BilinearUnit(nn.Module):
    """Pure bilinear unit: ``y = (W_lᵀ @ [1;x]) * (W_rᵀ @ [1;x])``.

    Neither branch is squashed, so this is *not* a gate — it is exactly the
    ``StarLinear`` of Exercise 02 with the identity activation. Comparing it with
    the GLU shows what the sigmoid actually adds: a bounded, interpretable gate.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.left = nn.Linear(in_features, out_features)
        self.right = nn.Linear(in_features, out_features)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # TODO: 
        raise NotImplementedError("implement BilinearUnit.forward")


class GatedTanhUnit(nn.Module):
    """GTU: a tanh *value* branch gated by a sigmoid branch.

        ``y = tanh(W_vᵀ @ [1;x]) * σ(W_gᵀ @ [1;x])``

    The GLU paper compares against this and argues GLU's linear content branch
    propagates gradients more directly than GTU's tanh-squashed one.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.value = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # TODO: 
        raise NotImplementedError("implement GatedTanhUnit.forward")


# ============================================================================ #
# Exercise 3 — modern GLU variants (ReGLU / GEGLU / SwiGLU)
# ============================================================================ #
class ActivatedGatedUnit(nn.Module):
    """Generalised gated unit: ``y = content * gate_activation(gate_pre)``.

    Choosing ``gate_activation`` recovers each named variant. ReGLU and GEGLU
    gate with activations the engine already provides as methods:

        - GLU   : ``sigmoid``        (see ``GatedLinearUnit`` above)
        - ReGLU : ``Tensor.relu``    (``build_reglu``)
        - GEGLU : ``Tensor.gelu``    (``build_geglu``)

    SwiGLU gates with ``swish``, which is *not* a built-in method — so it gets its
    own class (``SwiGLU``) that reuses the ``swish`` you wrote in Exercise 01.
    """

    def __init__(self, in_features: int, out_features: int, gate_activation: Activation) -> None:
        self.gate_activation = gate_activation
        self.content = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # TODO: return self.content(x) * self.gate_activation(self.gate(x))
        raise NotImplementedError("implement ActivatedGatedUnit.forward")


def build_reglu(in_features: int, out_features: int) -> ActivatedGatedUnit:
    return ActivatedGatedUnit(in_features, out_features, lambda t: t.relu())


def build_geglu(in_features: int, out_features: int) -> ActivatedGatedUnit:
    return ActivatedGatedUnit(in_features, out_features, lambda t: t.gelu())


class SwiGLU(nn.Module):
    r"""SwiGLU: content gated by **Swish** — ``y = content * swish(gate_pre)``.

    This closes the loop with Exercise 01. The engine has no ``swish`` method, so
    you gate with the one **you** implemented there (``ExTensor.swish``), whose
    forward *and* ``_backward`` you wrote by hand. ``nn.Linear`` returns a plain
    ``Tensor``, so call your method on it as a function — a method is just a
    function of a tensor (the same trick used by ``exercises/q01_activations.py``)::

        ExTensor.swish(self.gate(x))

    If the SwiGLU gradient check below passes, it cross-validates *both* this
    forward and your Exercise 01 ``swish`` backward at once.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.content = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # TODO: 
        raise NotImplementedError("implement SwiGLU.forward")


# ============================================================================ #
# GIVEN — you do not edit below. Once your forwards work, this grades them.
# ============================================================================ #
def gate_statistics(gate: cpu.Tensor) -> dict:
    """Summarise a sigmoid gate: how open/closed are its values?"""
    g = gate.data
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "min": float(g.min()),
        "max": float(g.max()),
        "frac<0.1 (closed)": float(np.mean(g < 0.1)),
        "frac>0.9 (open)": float(np.mean(g > 0.9)),
    }


def grade() -> None:
    """Gradient-check every unit (smooth gates -> finite differences agree)."""
    cpu.set_seed(0)
    n_in, n_out, batch = 3, 2, 4
    x = cpu.Tensor(np.random.randn(n_in, batch))
    target = cpu.Tensor(np.random.randn(n_out, batch))

    cases = [
        ("GatedLinearUnit (GLU)", GatedLinearUnit(n_in, n_out)),
        ("BilinearUnit (= StarLinear)", BilinearUnit(n_in, n_out)),
        ("GatedTanhUnit (GTU)", GatedTanhUnit(n_in, n_out)),
        ("GEGLU", build_geglu(n_in, n_out)),
        ("SwiGLU (uses your Ex.01 swish)", SwiGLU(n_in, n_out)),
    ]
    print("Gradient check (analytic backward vs finite differences):\n")
    all_ok = True
    for name, layer in cases:
        print(f"  {name}:")
        try:
            worst = gradient_check(layer, x, target)
        except NotImplementedError as exc:
            # e.g. SwiGLU needs ExTensor.swish, which is implemented in Ex.01.
            all_ok = False
            print(f"    -> SKIPPED ({exc}). Finish that piece first.\n")
            continue
        ok = worst < 1e-5
        all_ok = all_ok and ok
        print(f"    -> {'PASS' if ok else 'FAIL'} (worst {worst:.2e})\n")
    print("All units correct — and you wrote no backward pass!"
          if all_ok else "Some units are off (or skipped); revisit their forward().")


def demonstrate_gate() -> None:
    """Show the sigmoid gate of a GLU: a learned, bounded on/off switch in (0,1)."""
    cpu.set_seed(1)
    glu = GatedLinearUnit(6, 6)
    x = cpu.Tensor(np.random.randn(6, 8))
    glu(x)                                   # populate the graph
    gate = sigmoid(glu.gate(x))              # the gate tensor itself
    print("\nGLU gate statistics on a random batch (sigmoid -> always in (0, 1)):")
    for key, value in gate_statistics(gate).items():
        print(f"    {key:18s} {value:+.3f}")


def main() -> None:
    print("=" * 70)
    print("Gated Linear Units — the GLU family on the bert_cpu engine")
    print("=" * 70 + "\n")
    grade()
    demonstrate_gate()


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"Implement the forwards first ({exc}).")
