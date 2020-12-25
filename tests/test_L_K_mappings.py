"""Test the L2K mappings module."""

from sympy import I

import L_K_mappings as sut

from K_1_forms import K
from utilities import is_in_expr


def test_L2K_a_less_b():
    """Test the calculation for L^a_b when a < b."""
    # GIVEN
    n = 4
    a = 1
    b = 2

    # WHEN
    result = sut._L2K_a_less_b(a, b, n)

    # THEN
    assert is_in_expr(K(3), result)
    assert is_in_expr(K(9), result)

    assert result == K(3) / 2 - I * K(9) / 2


def test_L2K_a_more_b():
    """Test the calculation for L^a_b when a < b."""
    # GIVEN
    n = 4
    a = 2
    b = 1

    # WHEN
    result = sut._L2K_a_more_b(a, b, n)

    # THEN
    assert is_in_expr(K(3), result)
    assert is_in_expr(K(9), result)

    assert result == K(3) / 2 + I * K(9) / 2


def test_L_diag_mappings():
    """Test the creation of the diagonal L 1-form mappings using inverse of P and Q."""
    # GIVEN
    n = 4

    # WHEN
    L_diag = sut._L_diag_mappings(n)

    # THEN
    assert len(L_diag) == n

    for i, j in zip(range(n), range(n)):
        assert is_in_expr(K(n**2 - i), L_diag[j])


def test_create_L2K():
    """Test the full method for creating L2K mappings."""
    # GIVEN
    n = 4

    # WHEN
    L2K = sut.create_L2K(n)

    # THEN
    assert len(L2K) == n
    for i in range(n): assert len(L2K[i]) == n
