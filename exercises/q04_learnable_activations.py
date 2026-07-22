"""Exercise 04 — Learnable Activations (a learned mix of activations).

YOUR TASK
=========
Instead of picking one fixed activation (ReLU *or* GELU *or* SiLU), let the layer
**learn how much each contributes**. With trainable scalars ``α_k`` and a fixed
dictionary ``{ReLU, GELU, SiLU}``::

    φ_LA(z) = α_relu · ReLU(z) + α_gelu · GELU(z) + α_silu · SiLU(z)

The coefficients are input-independent (the same ``α_k`` for every sample and
feature) and are trained by gradient descent like any other parameter.

As in the other exercises you implement **only ``forward``**. The combination is
built from ops the engine already differentiates (``*``, ``+``, ``relu``,
``gelu``, ``exp`` ...), so the engine produces the backward pass for you. The
gradient check (Part 3, reused from Exercise 02) confirms it.

Relation to the other exercises
-------------------------------
- ``q01_activations.py`` (Ex. 01): there you built ``swish`` (= SiLU) as a *primitive*
  op with a hand-written ``_backward``. Here SiLU is instead **composed** from
  engine ops (given as ``silu`` below), so this file stands alone — same maths,
  no ``_backward`` to write.
- ``q02_rewrite_the_stars.py`` / ``q03_gated_linear_units.py`` (Ex. 02/03): those *gate*
  or *multiply* branches; this one **adds** activations with learned weights. All
  three share the same lesson — compose Tensor ops and autograd does the rest —
  and the same ``gradient_check`` harness.

The maths of the backward (Part 2 — for understanding; you do NOT code it)
-------------------------------------------------------------------------
For ``y = Σ_k α_k φ_k(x)`` the engine derives, automatically:

    dL/dx   = dL/dy ⊙ Σ_k α_k φ_k'(x)          (chain rule through each φ_k)
    dL/dα_k = Σ (dL/dy ⊙ φ_k(x))               (summed over all batch/feature dims)

The ``α_k`` gradient is a *sum* because one scalar multiplies the whole tensor —
the engine's ``*`` backward calls ``_unbroadcast``, which sums a ``(1, 1)``
parameter's gradient over every position. You get both for free.

Required reading
----------------
[1] Wang, M., Wang, J., Xia, Y., Shen, K., & Zhong, S. "More Expressive
    Feedforward Layers: Part I. Token-Adaptive Mixing of Activations." 2026.

Conventions
-----------
Column-oriented like the rest of the library: tensors are ``(features, batch)``
and the coefficients are ``(1, 1)`` so they broadcast over both axes.

HOW TO WORK
===========
1. Fill in the two ``forward`` methods (remove each ``raise NotImplementedError``).
2. Run the file to grade yourself — it gradient-checks both layers and shows how
   much each activation contributes to the mix::

       python -m exercises.q04_learnable_activations

TIP: a ``(1,1)`` Parameter multiplied by a tensor broadcasts and is
differentiated by the engine, so each weighted term is a one-liner and the layer
is their sum. See ``q02_rewrite_the_stars.StarLinear`` for the shape.

"""

from __future__ import annotations

import numpy as np

from bert_cpu import engine as cpu
from bert_cpu import nn

# Share Exercise 02's finite-difference gradient checker (Part 3).
from exercises.q02_rewrite_the_stars import gradient_check


# SiLU / Swish, composed from engine ops (the swish you hand-wrote in Ex. 01).
def _sigmoid(t: cpu.Tensor) -> cpu.Tensor:
    return 1.0 / (1.0 + (-t).exp())


def silu(t: cpu.Tensor) -> cpu.Tensor:
    """SiLU(t) = t · σ(t)."""
    return t * _sigmoid(t)


# ============================================================================ #
# Part 1 — the unconstrained Learnable Activation
# ============================================================================ #
class LearnableActivation(nn.Module):
    """A learned linear combination ``α_relu·ReLU + α_gelu·GELU + α_silu·SiLU``.

    The three coefficients are ``(1, 1)`` ``Parameter``s, initialised to 1 (so the
    layer starts as the plain *sum* of the activations, per the paper). They are
    collected automatically by ``Module.parameters()`` and trained like weights.
    """

    def __init__(self) -> None:
        self.alpha_relu = nn.Parameter(cpu.ones(1, 1))
        self.alpha_gelu = nn.Parameter(cpu.ones(1, 1))
        self.alpha_silu = nn.Parameter(cpu.ones(1, 1))

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # ReLU, GELU, SiLU calculations
        valueRelu = x.relu()
        valueGelu = x.gelu()
        valueSilu = silu(x)

        # Linear combination
        return (self.alpha_relu * valueRelu + 
                self.alpha_gelu * valueGelu + 
                self.alpha_silu * valueSilu )

    def coefficients(self) -> dict:
        """The current (unconstrained) coefficients, for reporting."""
        return {
            "alpha_relu": self.alpha_relu.data.item(),
            "alpha_gelu": self.alpha_gelu.data.item(),
            "alpha_silu": self.alpha_silu.data.item(),
        }


# ============================================================================ #
# Part 6 — the softmax-normalized variant
# ============================================================================ #
class NormalizedLearnableActivation(nn.Module):
    """Same idea, but the weights are a softmax over trainable logits ``β_k``::

        π_k = exp(β_k) / Σ_j exp(β_j)            (so π_k > 0 and Σ_k π_k = 1)
        φ(x) = Σ_k π_k φ_k(x)

    You train the *logits* ``β_k`` (initialised to 0 → equal π = 1/3), not the
    π directly. This keeps the mix a convex combination: bounded, interpretable,
    and stable, at the cost of forbidding negative weights.
    """

    def __init__(self) -> None:
        self.beta_relu = nn.Parameter(cpu.zeros(1, 1))
        self.beta_gelu = nn.Parameter(cpu.zeros(1, 1))
        self.beta_silu = nn.Parameter(cpu.zeros(1, 1))

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        # Softmax manual calculation
        expRelu = self.beta_relu.exp()
        expGelu = self.beta_gelu.exp()
        expSilu = self.beta_silu.exp()

        soma = expRelu + expGelu + expSilu

        piRelu = expRelu / soma
        piGelu = expGelu / soma
        piSilu = expSilu / soma

        # ReLU, GELU, SiLU calculations
        valueRelu = x.relu()
        valueGelu = x.gelu()
        valueSilu = silu(x)

        # Linear combination
        return (piRelu * valueRelu +
                piGelu * valueGelu + 
                piSilu * valueSilu )

    def coefficients(self) -> dict:
        """The normalized weights π_k, for reporting."""
        betas = np.array([self.beta_relu.data, self.beta_gelu.data, self.beta_silu.data]).ravel()
        e = np.exp(betas - betas.max())
        pi = e / e.sum()
        return {"pi_relu": float(pi[0]), "pi_gelu": float(pi[1]), "pi_silu": float(pi[2])}


# ============================================================================ #
# GIVEN — you do not edit below. Once your forwards work, this grades and trains.
# ============================================================================ #
def grade() -> None:
    """Part 3: gradient-check both layers (analytic backward vs finite diff)."""
    cpu.set_seed(0)
    x = cpu.Tensor(np.random.randn(4, 3))
    target = cpu.Tensor(np.random.randn(4, 3))

    print("Gradient check (analytic backward vs finite differences):\n")
    all_ok = True
    for name, layer in [
        ("LearnableActivation", LearnableActivation()),
        ("NormalizedLearnableActivation", NormalizedLearnableActivation()),
    ]:
        print(f"  {name}:")
        try:
            worst = gradient_check(layer, x, target)
        except NotImplementedError as exc:
            all_ok = False
            print(f"    -> SKIPPED ({exc}).\n")
            continue
        ok = worst < 1e-5
        all_ok = all_ok and ok
        print(f"    -> {'PASS' if ok else 'FAIL'} (worst {worst:.2e})\n")
    print("Gradients correct — and you wrote no backward pass!"
          if all_ok else "Some layers are off (or skipped); revisit their forward().")


def demonstrate_mix() -> None:
    """Forward-only: with all coefficients = 1, each activation still contributes
    a *different* amount — so a coefficient's size alone does not say how important
    an activation is; you must weigh it against the activation's output scale.
    """
    cpu.set_seed(0)
    la = LearnableActivation()                       # coefficients all 1.0
    # A negative-shifted input: ReLU zeros most of it, GELU/SiLU do not — so the
    # same coefficient buys very different contributions.
    x = cpu.Tensor(np.random.randn(1, 256) - 1.5)
    terms = {
        "alpha_relu * ReLU(x)": (la.alpha_relu * x.relu()).data,
        "alpha_gelu * GELU(x)": (la.alpha_gelu * x.gelu()).data,
        "alpha_silu * SiLU(x)": (la.alpha_silu * silu(x)).data,
    }
    print("\nWith every coefficient = 1, mean |contribution| of each term:")
    for name, value in terms.items():
        print(f"    {name:22s} {float(np.abs(value).mean()):.3f}")
    print("Equal coefficients do NOT mean equal contribution — the activations"
          "\ndiffer in output scale, so read each alpha_k together with phi_k(x).")


def main() -> None:
    print("=" * 70)
    print("Learnable Activations — a learnable mix of ReLU/GELU/SiLU on bert_cpu")
    print("=" * 70 + "\n")
    grade()
    demonstrate_mix()


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"Implement the forwards first ({exc}).")


