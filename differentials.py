"""Calculate the differential (d<.>) of the L and K 1-forms."""

from sympy import I

from L_1_forms import L
from wedge import Wedge


def dL(a: int, b: int, n: int):
    """
    Calculate the differential of L_a^b in SU(n).

    This is derived from the Lie Algebra of SU(n) to be dL_a^b = I * (L_a^c L_c^b)
    where c is begin summed over (Einstein notation).
    """
    return I * sum(Wedge(L(a, i), L(i, b)) for i in range(n))
