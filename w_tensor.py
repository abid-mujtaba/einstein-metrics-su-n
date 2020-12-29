"""Implement the 𝜔-tensor and its differential."""

from sympy import Array, S, expand
from sympy.tensor import permutedims as pd, tensorcontraction as tc, tensorproduct as tp


def create_w_dd(c_ddd: Array, K_u: Array) -> Array:
    """Create the 𝜔_ab tensor from the c_ddd tensor and the K 1-forms."""
    c3 = c_ddd + pd(c_ddd, (0,2,1)) + pd(c_ddd, (2, 1, 0))
    c_ddd = S.Half * tc(tp(c3, K_u), (2,3))

    # Expand all elements
    c_ddd = c_ddd.applyfunc(expand)

    return c_ddd
