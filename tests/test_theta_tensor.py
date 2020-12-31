"""Test the theta_tensor module."""

import pytest

from itertools import product
from sympy import expand, Array

import theta_tensor as sut

from K_1_forms import K
from metric import x1, x2, x3
from utilities import is_Wedge_of_K_in_expr
from wedge import Wedge


@pytest.mark.parametrize("n", (3,))
def test_create_theta_ud(n: int, dw_ud: Array, w_wedge_ud: Array) -> None:
    """Test the creation of theta_ud."""
    # WHEN
    theta_ud = sut.create_theta_ud(dw_ud, w_wedge_ud)

    # THEN
    for i, j in product(range(n**2 - 1), repeat=2):
        expr = theta_ud[i,j]
        if i == j:
            assert expr == 0
        else:
            assert expr == theta_ud[i,j] == 0 or is_Wedge_of_K_in_expr(n, expr)


@pytest.mark.parametrize("n", (2,))
def test_create_theta_ud_n_equals_2(n: int, dw_ud: Array, w_wedge_ud: Array) -> None:
    """Test the creation of theta_ud against hand calculations for n=2."""
    # WHEN
    theta_ud = sut.create_theta_ud(dw_ud, w_wedge_ud)

    # THEN
    for i in range(n ** 2 - 1):
        assert theta_ud[i, i] == 0

    assert expand(theta_ud[0, 1]) == expand(
        (
            (2 * x1 + 2 * x2 - x3) / (4 * x1)
            + (2 * x1 - 2 * x2 + x3) * (2 * x1 - 2 * x2 - x3) / (8 * x1 * x3)
        )
        * Wedge(K(0), K(1))
    )
    assert expand(theta_ud[1, 2]) == expand(
        (
            - (2 * x1 - 2 * x2 - x3) / (2 * x2)
            - (2 * x1 + 2 * x2 - x3) * (2 * x1 - 2 * x2 + x3) / (8 * x1 * x2)
        )
        * Wedge(K(1), K(2))
    )
    assert expand(theta_ud[2, 0]) == expand(
        (
            - (2 * x1 - 2 * x2 + x3) / (2 * x3)
            - (2 * x1 - 2 * x2 - x3) * (2 * x1 + 2 * x2 - x3) / (8 * x2 * x3)
        )
        * Wedge(K(0), K(2))
    )

    assert expand(theta_ud[1, 0]) == expand(
        (
            - (2 * x1 + 2 * x2 - x3) / (4 * x2)
            - (2 * x1 - 2 * x2 + x3) * (2 * x1 - 2 * x2 - x3) / (8 * x2 * x3)
        )
        * Wedge(K(0), K(1))
    )
    assert expand(theta_ud[2, 1]) == expand(
        (
            (2 * x1 - 2 * x2 - x3) / (2 * x3)
            + (2 * x1 + 2 * x2 - x3) * (2 * x1 - 2 * x2 + x3) / (8 * x1 * x3)
        )
        * Wedge(K(1), K(2))
    )
    assert expand(theta_ud[0, 2]) == expand(
        (
            (2 * x1 - 2 * x2 + x3) / (2 * x1)
            + (2 * x1 - 2 * x2 - x3) * (2 * x1 + 2 * x2 - x3) / (8 * x1 * x2)
        )
        * Wedge(K(0), K(2))
    )
