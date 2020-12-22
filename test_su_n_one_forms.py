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
