"""Test the ricci module."""

import pytest

from itertools import combinations, product
from sympy import Array, expand

import ricci as sut

from metric import x1, x2, x3


@pytest.mark.parametrize("n", (3,))
def test_create_R_uddd(n: int, theta_ud: Array) -> None:
    """Test the creation of the R_uddd tensor."""
    # GIVEN
    dim = n ** 2 - 1

    # WHEN
    R_uddd = sut.create_R_uddd(n, theta_ud)

    # THEN
    for i in range(dim):
        for j, k in product(range(dim), repeat=2):
            assert R_uddd[i, i, j, k] == 0

    for i, j in product(range(dim), repeat=2):
        for k in range(dim):
            assert R_uddd[i, j, k, k] == 0

        # Verify antisymmetry of the last two indices
        for k, l in combinations(range(dim), 2):
            assert expand(R_uddd[i, j, k, l] + R_uddd[i, j, l, k]) == 0


@pytest.mark.parametrize("n", (2,))
def test_create_R_uddd_n_equals_2(n: int, theta_ud: Array) -> None:
    """Test the creation of R_uddd against hand calculations for n=2."""
    # GIVEN
    dim = n ** 2 - 1

    def _assert_zero(tensor: Array, c: int, d: int) -> None:
        """
        Assert that all entries of the 2-tensor other than c, d and d, c are zero.

        Also confirm that the tensor is antisymmetric in the c and d indices.
        """
        for i, j in product(range(dim), repeat=2):
            if (i, j) != (c, d) and (i, j) != (d, c):
                assert tensor[i, j] == 0

        # Verify antisymmetry in the c and d indices
        assert expand(tensor[c, d] + tensor[d, c]) == 0

    # WHEN
    R_uddd = sut.create_R_uddd(n, theta_ud)

    # THEN
    for i in range(dim):
        for j, k in product(range(dim), repeat=2):
            assert R_uddd[i, i, j, k] == 0

    for i, j in product(range(dim), repeat=2):
        for k in range(dim):
            assert R_uddd[i, j, k, k] == 0

    _assert_zero(R_uddd[0, 1, :, :], 0, 1)
    _assert_zero(R_uddd[1, 2, :, :], 1, 2)
    _assert_zero(R_uddd[2, 0, :, :], 2, 0)

    _assert_zero(R_uddd[1, 0, :, :], 1, 0)
    _assert_zero(R_uddd[2, 1, :, :], 2, 1)
    _assert_zero(R_uddd[0, 2, :, :], 0, 2)

    # Check one entry (sanity)
    assert expand(R_uddd[0, 1, 0, 1]) == expand(
        (
            (2 * x1 + 2 * x2 - x3)
            + (2 * x1 - 2 * x2 + x3) * (2 * x1 - 2 * x2 - x3) / (2 * x3)
        )
        / (8 * x1)
    )
