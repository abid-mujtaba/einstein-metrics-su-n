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


def create_w_ud(w_dd: Array, g_uu: Array) -> Array:
    """Create the w^a_b tensor by using the inverse metric."""
    w_ud = tc(tp(g_uu, w_dd), (1,2)).applyfunc(expand)

    return w_ud
