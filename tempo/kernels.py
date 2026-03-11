import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Kernel functions  κ(u)  and their derivatives  κ'(u)
#  All operate element-wise on dot-product values u = <y, y'> ∈ [-1, 1]
#  (inputs are unit-normalised patches on the sphere).
#
#  Each entry in `kernels` is a dict with:
#    'fn'   : κ(u, *args)
#    'deriv': κ'(u, *args)   ← needed for Proposition 1 backprop
# ─────────────────────────────────────────────────────────────────────────────

def exp_fn(u, alpha):
    """κ(u) = exp(α (u − 1))  — RBF kernel on the sphere (paper Eq. 2)."""
    return np.exp(alpha * (u - 1.0))

def exp_deriv(u, alpha):
    """κ'(u) = α exp(α (u − 1))"""
    return alpha * np.exp(alpha * (u - 1.0))

def poly_fn(u, alpha=2):
    """κ(u) = u^α  — polynomial kernel."""
    return np.power(u, alpha)

def poly_deriv(u, alpha=2):
    """κ'(u) = α u^(α-1)"""
    return alpha * np.power(u, alpha - 1)


kernels = {
    "exp":  {"fn": exp_fn,  "deriv": exp_deriv},
    "poly": {"fn": poly_fn, "deriv": poly_deriv},
}
