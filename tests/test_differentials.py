"""Test the differentials module."""

from L_K_mappings import create_L2K
from sympy import I, expand, sqrt

import differentials as sut

from K_1_forms import K
from L_1_forms import L
from wedge import Wedge


def test_dL() -> None:
    """Test the calculation of the differential of L 1-forms."""
    # GIVEN
    n = 3
    a = 1
    b = 2

    # WHEN
    result = sut.dL(a, b, n)

    # THEN
    assert result == I * (
        Wedge(L(1, 0), L(0, 2)) + Wedge(L(1, 1), L(1, 2)) + Wedge(L(1, 2), L(2, 2))
    )


def test_differentiate_sum_of_L_1_forms_single() -> None:
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


def test_differential_sum_of_L_1_forms_double() -> None:
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


def test_convert_L_2_forms() -> None:
    """Test the conversion of L 2-forms (via wedge) to wedged K 1-forms."""
    # GIVEN
    n = 2
    wedge = Wedge(L(0, 1), L(1, 0))

    # L2K = create_L2K(n)

    # WHEN
    result = sut._convert_L_2_form(n, wedge)

    # THEN
    assert result


def test_create_dK() -> None:
    """Test the creation of the dK expressions."""
    # GIVEN
    n = 2

    # WHEN
    dK = sut.create_dK(n)

    # THEN
    assert len(dK) == n ** 2


def test_create_dK_n_equals_2() -> None:
    """Test the creation of dK against hand calculations for n = 2."""
    # GIVEN
    n = 2

    # WHEN
    dK = sut.create_dK(n)

    # THEN
    assert len(dK) == 4

    assert dK[0] == -1 * sqrt(2) * Wedge(K(1), K(2))
    assert dK[1] == sqrt(2) * Wedge(K(0), K(2))
    assert dK[2] == -1 * Wedge(K(0), K(1)) / sqrt(2)
    assert dK[3] == 0
