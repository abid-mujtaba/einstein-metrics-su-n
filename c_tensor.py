"""
Create the c tensor.

Based on the definitions:

1. dK^a = -1/2 c_ab^c K^b ∧ K^c
2. c_abc = g_cd c_ab^d
"""

from sympy import Expr, Add, Mul, Array
from sympy.tensor import tensorcontraction as tc, tensorproduct as tp
from typing import List

from wedge import extract_wedge_coeff
from K_1_forms import K


def _coeff(dK: List[Expr], a: int, b: int, c: int) -> Expr:
    """
    Calculate the c_ab^c coefficients from dK.

    Based on the definition dK^a = 1/2 c_ab^c K^b ∧ K^c.
    The dK have had the anti-symmetric terms combined so we will extract for b < c and
    split back in to the two anti-symmetric parts (one positive, the other negative).
    """
    if b == c:  # Simple consequence of the anti-symmetry in b and c
        return 0

    if b < c:
        return extract_wedge_coeff(dK[a], b, c)

    else:
        return -1 * extract_wedge_coeff(dK[a], c, b)


def create_c_ddu(dK: List[Expr], n: int) -> Array:
    """
    Create the c_ddu tensor using the mapping from dK to K 2-forms (Wedges).

    Based on the definition: dK^a = -1/2 * c_bc^a K^b ^ K^c
    """
    dim = n ** 2 - 1

    return Array(
        [
            [[-1 * _coeff(dK, a, b, c) for a in range(dim)] for c in range(dim)]
            for b in range(dim)
        ]
    )


def create_c_ddd(c_ddu: Array, g_dd: Array) -> Array:
    """
    Create the c_ddd tensor using c_ddu and the metric tensor (to lower).

    Definition: c_abc = g_cd * c_ab^d = c_ab^d * g_cd
    """
    return tc(tp(c_ddu, g_dd), (2, 4))
