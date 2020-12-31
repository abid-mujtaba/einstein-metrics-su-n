"""Implement the 𝜔-tensor, its wedge and its differential."""

from functools import partial
from sympy import Array, S, expand, factor, Expr
from sympy.tensor import permutedims as pd, tensorcontraction as tc, tensorproduct as tp
from typing import List

from K_1_forms import K
from wedge import Wedge, antisymm, expand_K, extract_factor_K


def create_w_dd(c_ddd: Array, K_u: Array) -> Array:
    """Create the 𝜔_ab tensor from the c_ddd tensor and the K 1-forms."""
    c3 = c_ddd + pd(c_ddd, (0, 2, 1)) + pd(c_ddd, (2, 1, 0))
    c_ddd = S.Half * tc(tp(c3, K_u), (2, 3))

    # Expand all elements
    c_ddd = c_ddd.applyfunc(expand)

    return c_ddd


def create_w_ud(w_dd: Array, g_uu: Array) -> Array:
    """Create the w^a_b tensor by using the inverse metric."""
    w_ud = tc(tp(g_uu, w_dd), (1, 2)).applyfunc(expand)

    return w_ud


def create_w_wedge(n: int, w_ud: Array) -> Array:
    """Create the w^a_c ^ w_c^b wedge with the c being summed over."""
    dim = n ** 2 - 1

    # Create the w wedge tensor by summing over the c index
    # Simplify the expressions using the simplification utility functions from
    # the wedge module as well as factorizing the end-result
    wedge = (
        Array(
            [
                [
                    sum(Wedge(w_ud[a, c], w_ud[c, b]) for c in range(dim))
                    for b in range(dim)
                ]
                for a in range(dim)
            ]
        )
        .applyfunc(expand_K)
        .applyfunc(extract_factor_K)
        .applyfunc(antisymm)
        .applyfunc(factor)
    )

    return wedge


def _differential_of_K_expr(dK: List[Expr], expr: Expr) -> Expr:
    """
    Calculate the differential of a linear combination of K 1-forms.

    We use a pre-calculated list of differentials of K: dK.
    """
    if isinstance(expr, K):  # Replace
        return dK[expr.index]

    if args := expr.args:  # Descend and recurse
        return expr.func(*(_differential_of_K_expr(dK, arg) for arg in args))

    return expr  # Return atoms as is


def create_dw_ud(w_ud: Array, dK: List[Expr]) -> Array:
    """Create the differential of the w_ud tensor."""
    differentiate = partial(_differential_of_K_expr, dK)

    return w_ud.applyfunc(differentiate)
