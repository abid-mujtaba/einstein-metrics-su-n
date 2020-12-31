"""Test the su_n_one_forms module."""

import sympy as sp
import L_1_forms as sut


def test_creation_of_L_1_forms() -> None:
    """
    Test the creation of the L 1-forms.

    In particular them being sub-class of Differential.
    """
    # WHEN
    l_12 = sut.L(1, 2)

    # THEN
    assert isinstance(l_12, sp.Expr)
    assert l_12.index_1 == 1
    assert l_12.index_2 == 2


def test_L_1_form_representation() -> None:
    """Should be represented with the indices correctly lowered and raised."""
    # WHEN
    l_12 = sut.L(1, 2)

    # THEN
    assert repr(l_12) == "L₁²"
    assert str(l_12) == "L₁²"


def test_L_1_form_representation_higher_numbers() -> None:
    """Representation moves away from unicode for higher numbers."""
    # WHEN
    l_13_17 = sut.L(13, 17)

    # THEN
    assert repr(l_13_17) == "L_13^17"
    assert str(l_13_17) == "L_13^17"


def test_L_1_form_equality() -> None:
    """Equal only when they have the same indices."""
    # GIVEN
    a = sut.L(1, 2)
    b = sut.L(1, 2)

    # THEN
    assert a == b


def test_L_1_form_inequality() -> None:
    """Equal only when they have the same indices."""
    # GIVEN
    a = sut.L(1, 2)
    b = sut.L(2, 3)

    # THEN
    assert a != b


def test_L_1_form_hash_equality() -> None:
    """Equal L 1-forms have the same hash."""
    # GIVEN
    a = sut.L(1, 2)
    b = sut.L(1, 2)

    # THEN
    assert hash(a) == hash(b)


def test_L_1_form_hash_inequality() -> None:
    """Equal L 1-forms have the same hash."""
    # GIVEN
    a = sut.L(1, 2)
    b = sut.L(2, 3)

    # THEN
    assert hash(a) != hash(b)


def test_L_1_form_ordering() -> None:
    """L 1-forms are ordered based on the 2 indices, the first one first."""
    # GIVEN
    l_12 = sut.L(1, 2)
    l_13 = sut.L(1, 3)
    l_21 = sut.L(2, 1)

    # THEN
    assert l_12 < l_13
    assert l_13 > l_12

    assert l_12 < l_21
    assert l_21 > l_12


def test_addition() -> None:
    """Good test of sympy functionality, ability to add to create a new expression."""
    # GIVEN
    a = sut.L(1, 2)
    b = sut.L(2, 3)

    # WHEN
    expr = a + b

    # THEN
    assert str(expr) == "L₁² + L₂³"


def test_create_L_1_forms() -> None:
    """Creates a 2-array of the n² L 1-forms for SU(n)."""
    # GIVEN
    n = 3

    # WHEN
    L = sut.create_L_1_forms(n)

    # THEN
    assert len(L) == 3

    for i in range(3):
        assert len(L[i]) == 3

    for i in range(3):
        for j in range(3):
            assert isinstance(L[i][j], sp.Expr)
