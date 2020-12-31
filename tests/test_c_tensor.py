"""Test the c_tensor module."""

from itertools import product
from sympy import sqrt
from sympy.tensor import permutedims as pd

from differentials import create_dK
from K_1_forms import K
from metric import create_metric, x1, x2, x3

import c_tensor as sut


def test_coeff_n_equals_2() -> None:
    """Test _coeff against hand-calculations for n=2."""
    # GIVEN
    n = 2
    dK = create_dK(n)

    # THEN
    assert sut._coeff(dK, 0, 1, 2) == -1 * sqrt(2)
    assert sut._coeff(dK, 1, 0, 2) == sqrt(2)
    assert sut._coeff(dK, 2, 0, 1) == -1 / sqrt(2)

    # ._coeff is designed to be anti-symmetric in b and c
    assert sut._coeff(dK, 0, 2, 1) == sqrt(2)
    assert sut._coeff(dK, 1, 2, 0) == -1 * sqrt(2)
    assert sut._coeff(dK, 2, 1, 0) == 1 / sqrt(2)

    # Everything else is zero
    assert sut._coeff(dK, 0, 0, 0) == 0
    assert sut._coeff(dK, 0, 0, 1) == 0
    assert sut._coeff(dK, 0, 0, 2) == 0
    assert sut._coeff(dK, 0, 1, 1) == 0
    assert sut._coeff(dK, 0, 2, 2) == 0


def test_create_c_ddu() -> None:
    """Test the creation of the c_ddu tensor."""
    # GIVEN
    n = 3
    dim = n ** 2 - 1
    dK = create_dK(n)

    # WHEN
    c_ddu = sut.create_c_ddu(dK, n)

    # THEN
    # Verify that the only non-zero entries of the tensor occur for unique indices
    for i, j, k in product(range(dim), repeat=3):
        if c_ddu[i, j, k]:
            assert len(set((i, j, k))) == 3  # Unique indices

    # Verify that c_ddu is antisymmetric in the first two indices
    antisymm_sum = c_ddu + pd(c_ddu, (1, 0, 2))

    for i, j, k in product(range(dim), repeat=3):
        assert antisymm_sum[i, j, k] == 0


def test_create_c_ddu_n_equals_2() -> None:
    """Test the creation of the c_ddu tensor against had calculations for n = 2."""
    # GIVEN
    n = 2
    dK = create_dK(n)

    # WHEN
    c_ddu = sut.create_c_ddu(dK, n)

    # THEN
    assert c_ddu[1, 2, 0] == sqrt(2)
    assert c_ddu[0, 2, 1] == -1 * sqrt(2)
    assert c_ddu[0, 1, 2] == 1 / sqrt(2)

    assert c_ddu[2, 1, 0] == -1 * sqrt(2)
    assert c_ddu[2, 0, 1] == 1 * sqrt(2)
    assert c_ddu[1, 0, 2] == -1 / sqrt(2)

    assert c_ddu[0, 0, 0] == 0
    assert c_ddu[0, 0, 1] == 0
    assert c_ddu[0, 1, 0] == 0
    assert c_ddu[0, 1, 1] == 0


def test_create_c_ddd_n_equals_2() -> None:
    """Test the creation of c_ddd against hand calculated results for n = 2."""
    # GIVEN
    n = 2
    g_dd, _ = create_metric(n)
    dK = create_dK(n)
    c_ddu = sut.create_c_ddu(dK, n)

    # WHEN
    c_ddd = sut.create_c_ddd(c_ddu, g_dd)

    assert c_ddd[0, 1, 2] == x3 / sqrt(2)
    assert c_ddd[2, 0, 1] == sqrt(2) * x2
    assert c_ddd[1, 2, 0] == sqrt(2) * x1

    assert c_ddd[1, 0, 2] == -1 * x3 / sqrt(2)
    assert c_ddd[0, 2, 1] == -sqrt(2) * x2
    assert c_ddd[2, 1, 0] == -sqrt(2) * x1

    assert c_ddd[0, 0, 0] == 0
    assert c_ddd[0, 0, 1] == 0
    assert c_ddd[0, 1, 0] == 0
    assert c_ddd[0, 1, 1] == 0


def test_create_c_ddd() -> None:
    """Test the creation of the c_ddd tensor."""
    # GIVEN
    n = 3
    dim = n ** 2 - 1
    g_dd, _ = create_metric(n)
    dK = create_dK(n)

    # WHEN
    c_ddu = sut.create_c_ddu(dK, n)
    c_ddd = sut.create_c_ddd(c_ddu, g_dd)

    # THEN
    # Verify that the only non-zero entries of the tensor occur for unique indices
    for i, j, k in product(range(dim), repeat=3):
        if c_ddu[i, j, k]:
            assert len(set((i, j, k))) == 3  # Unique indices

    # Verify that c_ddu is antisymmetric in the first two indices
    antisymm_sum = c_ddd + pd(c_ddd, (1, 0, 2))

    for i, j, k in product(range(dim), repeat=3):
        assert antisymm_sum[i, j, k] == 0
