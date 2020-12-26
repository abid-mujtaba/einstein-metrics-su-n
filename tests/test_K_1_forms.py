"""Test the su(n) K 1-forms module."""

import sympy as sp

import K_1_forms as sut

from L_1_forms import L as createL


def test_creation_of_K_1_forms():
    """
    Test the creation of the K 1-forms.

    In particular them being sub-class of Differential.
    """
    # WHEN
    k_1 = sut.K(1)

    # THEN
    assert isinstance(k_1, sp.Expr)
    assert k_1.index == 1


def test_K_1_form_representation():
    """Should be represented with the index lowered."""
    # WHEN
    k_3 = sut.K(3)
    k_12 = sut.K(12)

    # THEN
    assert repr(k_3) == "K₃"
    assert str(k_3) == "K₃"

    assert repr(k_12) == "K₁₂"
    assert str(k_12) == "K₁₂"


def test_K_1_form_equality():
    """Equal only when the index matches."""
    # GIVEN
    a = sut.K(7)
    b = sut.K(7)

    # THEN
    assert a == b
    assert hash(a) == hash(b)


def test_K_1_form_inequality():
    """Equal only when the index matches."""
    # GIVEN
    a = sut.K(3)
    b = sut.K(7)

    # THEN
    assert a != b
    assert hash(a) != hash(b)


def test_K_1_form_ordering():
    """Ordering is based on indices."""
    # GIVEN
    k_3 = sut.K(3)
    k_7 = sut.K(7)

    # THEN
    assert k_3 < k_7
    assert k_7 > k_3


def test_addition():
    """Good test of sympy functionality, ability to add to create a new expression."""
    # GIVEN
    k_3 = sut.K(3)
    k_7 = sut.K(7)

    # WHEN
    expr = k_3 + k_7

    # THEN
    assert str(expr) == "K₃ + K₇"
