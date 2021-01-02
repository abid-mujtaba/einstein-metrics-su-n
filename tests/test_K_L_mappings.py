"""Test the K to L mappings defined in K_L_mappings.py."""

import pytest

from itertools import product
from sympy import I, Rational, expand
from sympy.functions import sqrt
from sympy.matrices import Matrix

import K_L_mappings as sut

from L_1_forms import L
from utilities import is_in_expr


def test_category_1_K2L() -> None:
    """Test creation of the category 1 mapping from K to L 1-forms."""
    # GIVEN
    n = 3
    m = n * (n - 1) / 2

    # WHEN
    K2L = tuple(sut._category_1_K2L(n))

    # THEN
    assert len(K2L) == m

    assert K2L[0] == L(0, 1) + L(1, 0)
    assert K2L[-1] == L(2, 1) + L(1, 2)


def test_category_2_K2L() -> None:
    """Test creation of the category 2 mapping from K to L 1-forms."""
    # GIVEN
    n = 3
    m = n * (n - 1) / 2

    # WHEN
    K2L = tuple(sut._category_2_K2L(n))

    # THEN
    assert len(K2L) == m

    assert K2L[0] == I * (L(0, 1) - L(1, 0))
    assert K2L[-1] == I * (L(1, 2) - L(2, 1))


@pytest.mark.parametrize("n", (3, 4))
def test_category_3_K2L(n: int) -> None:
    """Test the creation of the category 3 K2L mappings (uses the P qnd Q matrices)."""
    # WHEN
    K2L = tuple(sut._category_3_K2L(n))

    # THEN
    assert len(K2L) == n

    # Each entry (K 1-form) must have contributions from all the diagonal L 1-forms
    for i, j in product(range(n), repeat=2):
        assert is_in_expr(L(i, i), K2L[j])


def test_create_K2L() -> None:
    """Test creation of complete K2L mapping."""
    # GIVEN
    n = 4

    # WHEN
    K2L = sut.create_K2L(n)

    # THEN
    assert len(K2L) == n ** 2


def test_create_K2L_n_equals_2() -> None:
    """Test mapping for n = 2 with results calculated by hand."""
    # GIVEN
    n = 2

    # WHEN
    K2L = sut.create_K2L(n)

    # THEN
    assert len(K2L) == 4

    assert K2L[0] == L(0, 1) + L(1, 0)
    assert K2L[1] == I * (L(0, 1) - L(1, 0))
    assert K2L[2] == expand((L(0, 0) - L(1, 1)) / sqrt(2))
    assert K2L[3] == expand((L(0, 0) + L(1, 1)) / sqrt(2))
