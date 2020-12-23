"""Test the K to L mappings defined in K_L_mappings.py."""

from sympy import I

import K_L_mappings as sut

from L_1_forms import L


def test_category_1_K2L():
    """Test creation of the category 1 mapping from K to L 1-forms."""
    # GIVEN
    n = 3
    m = n * (n - 1) / 2

    # WHEN
    K2L = tuple(sut._category_1_K2L(n))

    # THEN
    assert len(K2L) == m

    assert K2L[0] == L(0,1) + L(1,0)
    assert K2L[-1] == L(2,1) + L(1,2)


def test_category_2_K2L():
    """Test creation of the category 2 mapping from K to L 1-forms."""
    # GIVEN
    n = 3
    m = n * (n - 1) / 2

    # WHEN
    K2L = tuple(sut._category_2_K2L(n))

    # THEN
    assert len(K2L) == m

    assert K2L[0] == I * (L(0,1) - L(1,0))
    assert K2L[-1] == I * (L(1,2) - L(2,1))
