"""Test the wrapper Wedge class in the wedge module."""

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


def test_antisymmetry_of_wedge() -> None:
    """Test that A ∧ B = - B ∧ A."""
    # GIVEN
    k_1 = K(1)
    k_2 = K(2)

    # WHEN
    wedge_1_2 = sut.Wedge(k_1, k_2)
    wedge_2_1 = sut.Wedge(k_2, k_1)

    # THEN
    assert k_1 < k_2
    assert wedge_2_1 == -1 * wedge_1_2
