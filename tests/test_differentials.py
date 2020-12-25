"""Test the differentials module."""

from sympy import I

import differentials as sut

from L_1_forms import L
from wedge import Wedge


def test_dL():
    """Test the calculation of the differential of L 1-forms."""
    # GIVEN
    n = 3
    a = 1
    b = 2

    # WHEN
    result = sut.dL(a, b, n)

    # THEN
    assert result == I * (Wedge(L(1,0), L(0,2)) + Wedge(L(1,1), L(1,2)) + Wedge(L(1,2), L(2,2)))
