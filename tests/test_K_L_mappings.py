"""Test the K to L mappings defined in K_L_mappings.py."""

import pytest

from sympy import I, Rational
from sympy.functions import sqrt
from sympy.matrices import Matrix

import K_L_mappings as sut

from L_1_forms import L
from utilities import is_in_expr


def test_category_1_K2L():
    """Test creation of the category 1 mapping from K to L 1-forms."""
    # GIVEN
    n = 3
    m = n * (n - 1) / 2

    # WHEN
    K2L = tuple(sut._category_1_K2L(n))

    # THEN
    assert len(K2L) == m

    assert K2L[0] == L(0,1) + L(1,0)
    assert K2L[-1] == L(2,1) + L(1,2)


def test_category_2_K2L():
    """Test creation of the category 2 mapping from K to L 1-forms."""
    # GIVEN
    n = 3
    m = n * (n - 1) / 2

    # WHEN
    K2L = tuple(sut._category_2_K2L(n))

    # THEN
    assert len(K2L) == m

    assert K2L[0] == I * (L(0,1) - L(1,0))
    assert K2L[-1] == I * (L(1,2) - L(2,1))


def test_P_matrix():
    """Test the creation of the P matrix used to create the category 3 mappings."""
    # GIVEN
    n = 4

    # WHEN
    P = sut._create_P_matrix(n)

    # THEN
    assert isinstance(P, Matrix)
    assert P.shape == (n, n)

    # Diagnoal entries must equal 2/(n - 1) - 1 (except for the last)
    assert P[0,0] == Rational(2, n - 1) - 1
    assert float(P[0, 0]) == pytest.approx(-0.33333333)

    # Off-diagnoal entries must equal 2/(n - 1) (except for the last row and col)
    assert P[0, 1] == Rational(2, n - 1)
    assert float(P[0, 1]) == pytest.approx(0.666666667)

    # Last row and column must have zeros (except for the last diagonal entry)
    assert P[0,3] == 0
    assert P[3,0] == 0

    # Last diagonal entry must equal 1
    assert P[3,3] == 1

    # Test orthonormality of the rows of the P matrix
    row_1 = P[0,:]
    row_2 = P[1,:]

    assert row_1.dot(row_1) == 1
    assert row_1.dot(row_2) == 0


def test_Q_matrix():
    """Test the Q matrix used for mapping the category 3 K 1-forms to diag L 1-forms."""
    # GIVEN
    n = 4

    # WHEN
    Q = sut._create_Q_matrix(n)

    # THEN
    assert isinstance(Q, Matrix)
    assert Q.shape == (n, n)

    assert Q[0,0] == 1 / sqrt(2)
    assert Q[0,1] == - 1 / sqrt(2)
    assert Q[0,2] == 0

    assert Q[1,0] == 1 / sqrt(6)
    assert Q[1,2] == - 2 / sqrt(6)
    assert Q[1,3] == 0

    assert Q[n - 1, 0] == 1 / sqrt(n)
    assert Q[n - 1, n - 1] == 1 / sqrt(n)

    # Verify orthonormality of the rows of Q
    row_1 = Q[0, :]
    row_2 = Q[1, :]
    row_n = Q[n - 1, :]

    assert row_1.dot(row_1) == 1
    assert row_1.dot(row_2) == 0
    assert row_1.dot(row_n) == 0

    assert row_n.dot(row_n) == 1


def test_category_3_K2L():
    """Test the creation of the category 3 K2L mappings (uses the P qnd Q matrices)."""
    # GIVEN
    n = 4

    # WHEN
    K2L = tuple(sut._category_3_K2L(n))

    # THEN
    assert len(K2L) == n

    # Each entry (K 1-form) must have contributions from all the diagonal L 1-forms
    for i, j in zip(range(n), range(n)):
        assert is_in_expr(L(i,i), K2L[j])
