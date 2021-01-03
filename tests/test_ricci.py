"""Test the ricci module."""

import pytest

from itertools import combinations, product
from sympy import Array, expand
from sympy.abc import a, b

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


@pytest.mark.parametrize("n", (3,))
def test_create_R_dd(n: int, R_uddd: Array) -> Array:
    """Test the creation of R_dd tensor."""
    # GIVEN
    dim = n ** 2 - 1

    # WHEN
    R_dd = sut.create_R_dd(R_uddd)

    # THEN
    for i in range(dim):
        assert R_dd[i, i]

    for i, j in combinations(range(dim), 2):
        assert R_dd[i, j] == 0
        assert R_dd[j, i] == 0

    # Verify that there are only 3 unique entries in the (n**2 - 1) diagonal elements
    assert len(set(R_dd[i, i] for i in range(dim))) == 3


@pytest.mark.parametrize("n", (2,))
def test_create_R_dd_n_equals_2(R_uddd: Array) -> Array:
    """Test the creation of R_dd against hand calculations for n=2."""
    # WHEN
    R_dd = sut.create_R_dd(R_uddd)

    R_00 = R_dd[0, 0]
    R_11 = R_dd[1, 1]
    R_22 = R_dd[2, 2]

    # THEN
    assert R_dd[0, 1] == 0
    assert R_dd[0, 2] == 0
    assert R_dd[1, 0] == 0
    assert R_dd[1, 2] == 0
    assert R_dd[2, 0] == 0
    assert R_dd[2, 1] == 0

    assert expand(R_00) == expand(
        (
            (2 * x1 + 2 * x2 - x3)
            + (2 * x1 - 2 * x2 + x3) * (2 * x1 - 2 * x2 - x3) / (2 * x3)
        )
        / (8 * x2)
        + (
            (2 * x1 - 2 * x2 + x3)
            + (2 * x1 - 2 * x2 - x3) * (2 * x1 + 2 * x2 - x3) / (4 * x2)
        )
        / (4 * x3)
    )
    assert expand(R_11) == expand(
        (
            (2 * x1 + 2 * x2 - x3)
            + (2 * x1 - 2 * x2 + x3) * (2 * x1 - 2 * x2 - x3) / (2 * x3)
        )
        / (8 * x1)
        - (
            (2 * x1 - 2 * x2 - x3)
            + (2 * x1 - 2 * x2 + x3) * (2 * x1 + 2 * x2 - x3) / (4 * x1)
        )
        / (4 * x3)
    )
    assert expand(R_22) == expand(
        (
            (2 * x1 - 2 * x2 + x3)
            + (2 * x1 + 2 * x2 - x3) * (2 * x1 - 2 * x2 - x3) / (4 * x2)
        )
        / (4 * x1)
        - (
            (2 * x1 - 2 * x2 - x3)
            + (2 * x1 + 2 * x2 - x3) * (2 * x1 - 2 * x2 + x3) / (4 * x1)
        )
        / (4 * x2)
    )

    # Document the factorized results (partially hand calculated)
    assert expand(R_00) == expand(
        (2 * x1 + 2 * x2 - x3) * (2 * x1 - 2 * x2 + x3) / (8 * x2 * x3)
    )
    assert expand(R_11) == expand(
        -1 * (2 * x1 + 2 * x2 - x3) * (2 * x1 - 2 * x2 - x3) / (8 * x1 * x3)
    )
    assert expand(R_22) == expand(
        -1 * (2 * x1 - 2 * x2 - x3) * (2 * x1 - 2 * x2 + x3) / (8 * x1 * x2)
    )

    # Verify the x1, x2 - exchange symmetry between R_00 and R_11, and
    # from R_22 to itself
    assert expand(R_00.subs({x1: a, x2: b}).subs({a: x2, b: x1})) == expand(R_11)
    assert expand(R_11.subs({x1: a, x2: b}).subs({a: x2, b: x1})) == expand(R_00)

    assert expand(R_22.subs({x1: a, x2: b}).subs({a: x2, b: x1})) == expand(R_22)


@pytest.mark.parametrize("n", (2,))
def test_calculate_Riem_2_simple(n: int, g_dd: Array, g_uu: Array) -> None:
    """Use a simple R_uddd to test calculate_Riem_2."""
    # GIVEN
    dim = n ** 2 - 1
    entries = [[[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]

    # We create an R_uddd with only one non-zero entry: first index first entry
    entries[0][0][0][0] = a
    R_uddd = Array(entries)

    # WHEN
    Riem_2 = sut.calculate_Riem_2(R_uddd, g_dd, g_uu)

    # THEN
    # Because the calculatioin involves one g_dd and three g_uu we have the following
    # relation:
    assert Riem_2 == a**2 * x1 * x1 ** (-3)


@pytest.mark.parametrize("n", (2,))
def test_calculate_Riem_2(n: int, R_uddd: Array, g_dd: Array, g_uu: Array) -> None:
    """Calculate Riem_2 using the full calculation."""
    # GIVEN
    dim = n ** 2 - 1

    # WHEN
    Riem_2 = sut.calculate_Riem_2(R_uddd, g_dd, g_uu)

    # THEN
    assert Riem_2
