"""Test the utilities module."""

import pytest

from sympy.abc import a, b, x, y, z

import utilities as sut

from K_1_forms import K
from wedge import Wedge


def test_is_in_expr_error_if_object_is_not_a_leaf() -> None:
    """Attempting to find a non-leaf object should raise a ValueError."""
    # GIVEN
    obj = x + y
    expr = x * y + z

    # WHEN
    with pytest.raises(ValueError):
        sut.is_in_expr(obj, expr)


def test_is_in_expr_true() -> None:
    """Test when the object is in the expression."""
    # GIVEN
    obj = x
    expr = x + y

    # WHEN
    result = sut.is_in_expr(obj, expr)

    # THEN
    assert result is True


def test_is_in_expr_false() -> None:
    """Test when the object is NOT in the expression."""
    # GIVEN
    obj = x
    expr = y + z

    # WHEN
    result = sut.is_in_expr(obj, expr)

    # THEN
    assert result is False


def test_is_K_in_expr_true() -> None:
    """Test the utility on a linear combination of K 1-forms."""
    # GIVEN
    n = 2
    expr = a * K(0) + b * K(0)

    # WHEN
    result = sut.is_K_in_expr(n, expr)

    # THEN
    assert result is True


def test_is_K_in_expr_false() -> None:
    """Test the utility on a linear combination without K 1-forms."""
    # GIVEN
    n = 2
    expr = a * x + b * y

    # WHEN
    result = sut.is_K_in_expr(n, expr)

    # THEN
    assert result is False


def test_is_Wedge_in_expr_True() -> None:
    """Test the utility on a linear combination of wedges."""
    # GIVEN
    n = 2
    expr = a * Wedge(K(0), K(1)) + b * Wedge(K(1), K(2))

    # WHEN
    result = sut.is_Wedge_of_K_in_expr(n, expr)

    # THEN
    assert result is True


def test_is_Wedge_of_K_in_expr_no_wedge_false() -> None:
    """Test the utility on a linear combination without wedges."""
    # GIVEN
    n = 2
    expr = a * x + b * y

    # WHEN
    result = sut.is_Wedge_of_K_in_expr(n, expr)

    # THEN
    assert result is False


def test_is_Wedge_of_K_in_expr_no_K_wedge_false() -> None:
    """Test the utility on a Wedge which is NOT over K 1-forms."""
    # GIVEN
    n = 2
    expr = a * Wedge(x, y) + b * Wedge(y, z)

    # WHEN
    result = sut.is_Wedge_of_K_in_expr(n, expr)

    # THEN
    assert result is False
