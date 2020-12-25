"""Calculate the differential (d<.>) of the L and K 1-forms."""

from sympy import I, Expr
from typing import List

from K_L_mappings import create_K2L
from L_1_forms import L
from wedge import Wedge


def dL(a: int, b: int, n: int) -> Expr:
    """
    Calculate the differential of L_a^b in SU(n).

    This is derived from the Lie Algebra of SU(n) to be dL_a^b = I * (L_a^c L_c^b)
    where c is begin summed over (Einstein notation).
    """
    return I * sum(Wedge(L(a, i), L(i, b)) for i in range(n))


def create_dK(n: int) -> List[Expr]:
    """
    Calculate the differential for all the K 1-forms in SU(n).

    This is an involved process where we map to the L 1-forms,
    calculate the differential on them,
    and then map back to the K 1-forms.
    """
    K2L = create_K2L(n)

    # TODO: Place-holder for now
    return K2L
