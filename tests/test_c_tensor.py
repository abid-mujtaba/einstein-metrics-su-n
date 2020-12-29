"""Test the c_tensor module."""

from sympy.abc import x, y

from K_1_forms import K
from wedge import Wedge

import c_tensor as sut


def test_extract_coeff_just_matching_wedge() -> None:
    """Extract coeff from an expression consisting only of a matching wedge."""
    # GIVEN
    b = 1
    c = 2

    expr = Wedge(K(b), K(c))

    # WHEN
    result = sut._extract_coeff(expr, b, c)

    # THEN
    assert result == 1


def test_extract_coeff_just_not_matching_wedge() -> None:
    """Extract coeff from an expression consisting only of a non-matching wedge."""
    # GIVEN
    b = 1
    c = 2

    expr = Wedge(K(b), K(c))

    # WHEN
    result = sut._extract_coeff(expr, 0, 1)

    # THEN
    assert result == 0


def test_extract_coeff_matching_wedge_with_multiplicative_factor() -> None:
    """Extract coeff from an expression consisting of a matching wedge with a factor."""
    # GIVEN
    b = 1
    c = 2

    expr = x * Wedge(K(b), K(c))

    # WHEN
    result = sut._extract_coeff(expr, b, c)

    # THEN
    assert result == x


def test_extract_coeff_non_matching_wedge_with_multiplicative_factor() -> None:
    """Extract coeff from an expression consisting of a non-matching wedge with a factor."""
    # GIVEN
    b = 1
    c = 2

    expr = x * Wedge(K(b), K(c))

    # WHEN
    result = sut._extract_coeff(expr, 0, 1)

    # THEN
    assert result == 0


def test_extract_coeff_matching_wedge_in_linear_combination() -> None:
    """Extract coeff from a linear combination containing a matching wedge."""
    # GIVEN
    b = 1
    c = 2

    expr = x * Wedge(K(0), K(1)) + y * Wedge(K(b), K(c))

    # WHEN
    result = sut._extract_coeff(expr, b, c)

    # THEN
    assert result == y


def test_extract_coeff_non_matching_wedge_in_linear_combination() -> None:
    """Extract coeff from a linear combination containing a non-matching wedge."""
    # GIVEN
    expr = x * Wedge(K(0), K(1)) + y * Wedge(K(1), K(2))

    # WHEN
    result = sut._extract_coeff(expr, 0, 2)

    # THEN
    assert result == 0
