"""Test the mappings module."""

import pytest

from sympy import Rational, expand
from sympy.functions import sqrt
from sympy.matrices import Matrix

import mappings as sut


def test_C_matrix() -> None:
    """Verify that the C mixing matrix is 2x2 and orthonormal."""
    # WHEN
    C = sut.create_C_matrix()

    row_0 = C[0, :]
    row_1 = C[1, :]

    # THEN
    assert C.shape == (2, 2)

    # Verify that the rows of C are Hermitian othornormal (complex conjugate dot product)
    assert expand(row_0.H.dot(row_0)) == 1
    assert expand(row_1.H.dot(row_1)) == 1

    assert expand(row_0.H.dot(row_1)) == 0
    assert expand(row_1.H.dot(row_0)) == 0


def test_P_matrix() -> None:
    """Test the creation of the P matrix used to create the category 3 mappings."""
    # GIVEN
    n = 4

    # WHEN
    P = sut.create_P_matrix(n)

    # THEN
    assert isinstance(P, Matrix)
    assert P.shape == (n, n)

    # Diagnoal entries must equal 2/(n - 1) - 1 (except for the last)
    assert P[0, 0] == Rational(2, n - 1) - 1
    assert float(P[0, 0]) == pytest.approx(-0.33333333)

    # Off-diagnoal entries must equal 2/(n - 1) (except for the last row and col)
    assert P[0, 1] == Rational(2, n - 1)
    assert float(P[0, 1]) == pytest.approx(0.666666667)

    # Last row and column must have zeros (except for the last diagonal entry)
    assert P[0, 3] == 0
    assert P[3, 0] == 0

    # Last diagonal entry must equal 1
    assert P[3, 3] == 1

    # Test orthonormality of the rows of the P matrix
    row_1 = P[0, :]
    row_2 = P[1, :]

    assert row_1.dot(row_1) == 1
    assert row_1.dot(row_2) == 0


def test_Q_matrix() -> None:
    """Test the Q matrix used for mapping the category 3 K 1-forms to diag L 1-forms."""
    # GIVEN
    n = 4

    # WHEN
    Q = sut.create_Q_matrix(n)

    # THEN
    assert isinstance(Q, Matrix)
    assert Q.shape == (n, n)

    assert Q[0, 0] == 1 / sqrt(2)
    assert Q[0, 1] == -1 / sqrt(2)
    assert Q[0, 2] == 0

    assert Q[1, 0] == 1 / sqrt(6)
    assert Q[1, 2] == -2 / sqrt(6)
    assert Q[1, 3] == 0

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
