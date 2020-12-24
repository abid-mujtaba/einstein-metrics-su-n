"""Test the utilities module."""

import pytest

from sympy.abc import x, y, z

import utilities as sut


def test_is_in_expr_error_if_object_is_not_a_leaf():
    """Attempting to find a non-leaf object should raise a ValueError."""
    # GIVEN
    obj = x + y
    expr = x * y + z

    # WHEN
    with pytest.raises(ValueError):
        sut.is_in_expr(obj, expr)


# TODO: Parametrize this
def test_is_in_expr_true():
    """Test when the object is in the expression."""
    # GIVEN
    obj = x
    expr = x + y

    # WHEN
    result = sut.is_in_expr(obj, expr)

    # THEN
    assert result is True


# TODO: Parametrize this
def test_is_in_expr_false():
    """Test when the object is NOT in the expression."""
    # GIVEN
    obj = x
    expr = y + z

    # WHEN
    result = sut.is_in_expr(obj, expr)

    # THEN
    assert result is False
