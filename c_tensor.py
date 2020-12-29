"""
Create the c tensor.

Based on the definitions:

1. dK^a = -1/2 c_ab^c K^b ∧ K^c
2. c_abc = g_cd c_ab^d
"""

from sympy import Expr, Add, Mul
from typing import List

from wedge import Wedge
from K_1_forms import K


def _extract_coeff(expr: Expr, b: int, c: int) -> Expr:
    """Extract coefficient of K^b ∧ K^c from wedge expression."""
    # To that end we will substitute K(b) ^ K(c) = 1 and replace all other wedges
    # with 0 and simplify the expression
    def sub(e: Expr) -> Expr:
        """Recurse into expression substituting the Wedges with 1 or 0 based on indices."""
        if isinstance(e, Wedge):  # Substitute
            i1, i2 = (_.index for _ in e.args)

            return 1 if (i1, i2) == (b, c) else 0

        if e.args:  # Descend and recurse
            args = (sub(arg) for arg in e.args)
            return e.func(*args)

        return e  # Return (immutable) leafs as is

    return sub(expr)


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
        return _extract_coeff(dK[a], b, c)

    else:
        return -1 * _extract_coeff(dK[a], c, b)
