"""Calculate the differential (d<.>) of the L and K 1-forms."""

from itertools import product
from sympy import I, Expr
from typing import List

from K_L_mappings import create_K2L
from L_K_mappings import create_L2K
from L_1_forms import L
from wedge import Wedge


def dL(a: int, b: int, n: int) -> Expr:
    """
    Calculate the differential of L_a^b in SU(n).

    This is derived from the Lie Algebra of SU(n) to be dL_a^b = I * (L_a^c L_c^b)
    where c is begin summed over (Einstein notation).
    """
    return I * sum(Wedge(L(a, i), L(i, b)) for i in range(n))


def _differentiate_sum_of_L_1_forms(n: int, expr) -> Expr:
    """Calculate the differential of an expression which is the sum of L 1-forms."""
    # We will directly replace each L 1-form with its corresponding dL value to
    # simulate the differential
    # We are assuming that the expression is a linear combination of the L 1-forms
    # with constant coefficients (whose differential is zero)
    def sub_L(l: L) -> Expr:
        a = l.index_1
        b = l.index_2

        return dL(a, b, n)

    def sub(e: Expr) -> Expr:
        """Recursive function for replacing L with dL."""
        if isinstance(e, L):
            return sub_L(e)

        e._args = tuple(sub(arg) for arg in e.args)
        return e

    return sub(expr)


def _convert_L_2_form(n: int, expr: Expr) -> Expr:
    """Convert an expression containing wedged L 1-forms to K 1-forms."""
    L2K = create_L2K(n)

    def sub_L(l: L) -> Expr:
        a = l.index_1
        b = l.index_2

        return L2K[a][b]

    def sub(e: Expr) -> Expr:
        """Recursive function for replace L with K."""
        if isinstance(e, L):
            return sub_L(e)

        e._args = tuple(sub(arg) for arg in e.args)
        return e

    return sub(expr)


def create_dK(n: int) -> List[Expr]:
    """
    Calculate the differential for all the K 1-forms in SU(n).

    This is an involved process where we map to the L 1-forms,
    calculate the differential on them,
    and then map back to the K 1-forms.
    """
    K2L = create_K2L(n)
    dK2L = [_differentiate_sum_of_L_1_forms(n, e) for e in K2L]

    # TODO: Place-holder for now
    return dK2L
