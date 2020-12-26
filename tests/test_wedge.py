"""Test the wrapper Wedge class in the wedge module."""

from sympy import expand

import wedge as sut

from K_1_forms import K


def test_representation() -> None:
    """Test string representation of a WedgeProduct."""
    # GIVEN
    k_1 = K(1)
    k_2 = K(2)

    # WHEN
    wedge = sut.Wedge(k_1, k_2)

    # THEN
    assert str(wedge) == "K₁ ∧ K₂"


def test_zero_wedge_of_equal_1_forms() -> None:
    """Taking the wedge of a 1-form with itself results in 0."""
    # GIVEN
    k_1 = K(1)

    # WHEN
    wedge = sut.Wedge(k_1, k_1)

    # THEN
    assert wedge == 0


def test_expand_K() -> None:
    """Test the expansion of K 1-form wedges over addition."""
    # GIVEN
    k_1 = K(1)
    k_2 = K(2)
    k_3 = K(3)
    k_4 = K(4)

    expr = sut.Wedge(k_1 + k_2, k_3 + k_4)

    expected_result = expand(
        sut.Wedge(k_1, k_3)
        + sut.Wedge(k_1, k_4)
        + sut.Wedge(k_2, k_3)
        + sut.Wedge(k_2, k_4)
    )

    # WHEN
    result = sut.expand_K(expr)

    # THEN
    assert result == expected_result
