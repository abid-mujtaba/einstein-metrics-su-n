"""Test the su_n_one_forms module."""

import sympy.diffgeom as dg
import su_n_one_forms as sut


def test_creation_of_L_1_forms():
    """
    Test the creation of the L 1-forms.

    In particular them being sub-class of Differential.
    """
    # WHEN
    l_12 = sut.L(1,2)

    # THEN
    assert isinstance(l_12, dg.Differential)

    # Applying a Differential twice should result in 0
    assert dg.Differential(l_12) == 0


def test_L_1_form_representation():
    """Should be represented with the indices correctly lowered and raised."""
    # WHEN
    l_12 = sut.L(1,2)

    # THEN
    assert repr(l_12) == "L₁²"
    assert str(l_12) == "L₁²"


def test_L_1_form_representation_higher_numbers():
    """Representation moves away from unicode for higher numbers."""
    # WHEN
    l_13_17 = sut.L(13, 17)

    # THEN
    assert repr(l_13_17) == "L_13^17"
    assert str(l_13_17) == "L_13^17"


def test_L_1_form_equality():
    """Equal only when they have the same indices."""
    # Given
    a = sut.L(1,2)
    b = sut.L(1,2)

    # THEN
    assert a == b


def test_L_1_form_inequality():
    """Equal only when they have the same indices."""
    # Given
    a = sut.L(1,2)
    b = sut.L(2,3)

    # THEN
    assert a != b


def test_create_su_n_L_1_forms():
    """Creates a 2-array of the n² L 1-forms for SU(n)."""
    # GIVEN
    n = 3

    # WHEN
    L = sut.create_su_n_L_1_forms(n)

    # THEN
    assert len(L) == 3

    for i in range(3):
        assert len(L[i]) == 3

    for i in range(3):
        for j in range(3):
            assert isinstance(L[i][j], dg.Differential)
