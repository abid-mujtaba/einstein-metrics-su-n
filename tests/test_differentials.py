"""Test the differentials module."""

from sympy import I, expand

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


def test_create_dK():
    """Test the creation of the dK expressions."""
    # GIVEN
    n = 2

    # WHEN
    dK = sut.create_dK(n)

    # THEN
    assert len(dK) == n**2


def test_differentiate_sum_of_L_1_forms_single():
    """Test the differentiation function using a single L 1-form."""
    # GIVEN
    n = 3
    a = 1
    b = 2
    expr = L(a, b)

    # WHEN
    result = sut._differentiate_sum_of_L_1_forms(n, expr)

    # THEN
    assert result == sut.dL(a, b, n)


def test_differential_sum_of_L_1_forms_double():
    """Test the differentiation function using a linear combination of 2 L 1-forms."""
    # GIVEN
    n = 4
    l_1 = L(1, 2)
    l_2 = L(3, 0)

    expr = 2 * l_1 - 3 * l_2

    # WHEN
    result = sut._differentiate_sum_of_L_1_forms(n, expr)

    # THEN
    assert expand(result) == expand(2 * sut.dL(1, 2, n) - 3 * sut.dL(3, 0, n))
