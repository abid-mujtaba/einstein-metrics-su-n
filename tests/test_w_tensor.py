"""Test the 𝜔_tensor.py module."""

import pytest

from itertools import product
from sympy import sqrt, Array, Expr, expand
from sympy.abc import a, b
from sympy.tensor import permutedims as pd
from typing import List

from c_tensor import create_c_ddu, create_c_ddd
from differentials import create_dK
from K_1_forms import create_K_u, K
from metric import create_metric, x1, x2, x3
from wedge import Wedge

import w_tensor as sut


@pytest.fixture(name="dK")
def dK_fixture(n: int) -> List[Expr]:
    """Create the list of dK (differential of the K 1-forms)."""
    return create_dK(n)


@pytest.fixture(name="c_ddd")
def c_ddd_fixture(n: int, dK: List[Expr]) -> Array:
    """Create the c_ddd tensor for SU(n)."""
    g_dd, _ = create_metric(n)
    c_ddu = create_c_ddu(dK, n)

    return create_c_ddd(c_ddu, g_dd)


@pytest.fixture(name="K_u")
def dK_u_fixture(n: int) -> Array:
    """Create the dK_u tensor of the differentials of the K 1-forms."""
    return create_K_u(n)


def _K_in_expr(n: int, expr: Expr) -> bool:
    """Verify that there is a K 1-form in the expression."""
    if isinstance(expr, K):
        return 0 <= expr.index < n ** 2

    if expr.args:
        return any(_K_in_expr(n, arg) for arg in expr.args)

    return False


def _Wedge_of_K_in_expr(n: int, expr: Expr) -> bool:
    """Verify that there is a Wedge of K 1-forms in the expression."""
    if isinstance(expr, Wedge):
        op1, op2 = expr.args

        return (isinstance(op1, K) and 0 <= op1.index < n ** 2) and (
            isinstance(op2, K) and 0 <= op2.index < n ** 2
        )

    if expr.args:
        return any(_Wedge_of_K_in_expr(n, arg) for arg in expr.args)

    return False


@pytest.mark.parametrize("n", (3,))
def test_create_w_dd(n: int, c_ddd: Array, K_u: Array) -> None:
    """General test of the creation of the w_dd tensor."""
    # WHEN
    w_dd = sut.create_w_dd(c_ddd, K_u)
    antisymm_sum = w_dd + pd(w_dd, (1, 0))

    # THEN: Verify antisymmetry. Verify that the w_dd contain K 1-forms
    for i, j in product(range(n ** 2 - 1), repeat=2):
        assert antisymm_sum[i, j] == 0
        assert w_dd[i, j] == 0 or _K_in_expr(n, w_dd[i, j])


@pytest.mark.parametrize("n", (2,))
def test_create_w_dd_tensor_n_equals_2(c_ddd: Array, K_u: Array) -> None:
    """Test the values of 𝜔_dd against hand calculations for n=2."""
    # GIVEN (expected values)
    factor = 1 / (2 * sqrt(2))

    e_01 = -1 * factor * (2 * x1 + 2 * x2 - x3) * K(2)
    e_12 = factor * (2 * x1 - 2 * x2 - x3) * K(0)
    e_20 = -1 * factor * (2 * x1 - 2 * x2 + x3) * K(1)

    # WHEN
    w_dd = sut.create_w_dd(c_ddd, K_u)

    # THEN
    assert w_dd[0, 1] == expand(e_01)
    assert w_dd[1, 2] == expand(e_12)
    assert w_dd[2, 0] == expand(e_20)

    assert w_dd[1, 0] == expand(-1 * w_dd[0, 1])
    assert w_dd[2, 1] == expand(-1 * w_dd[1, 2])
    assert w_dd[0, 2] == expand(-1 * w_dd[2, 0])

    assert w_dd[0, 0] == 0
    assert w_dd[1, 1] == 0
    assert w_dd[2, 2] == 0


@pytest.fixture(name="w_dd")
def w_dd_fixture(c_ddd: Array, K_u: Array) -> Array:
    """Create w_dd for SU(n)."""
    return sut.create_w_dd(c_ddd, K_u)


@pytest.fixture(name="g_uu")
def g_uu_fixture(n: int) -> Array:
    """Create the inverse metric tensor."""
    _, g_uu = create_metric(n)

    return g_uu


@pytest.mark.parametrize("n", (3,))
def test_create_w_ud(n: int, w_dd: Array, g_uu: Array) -> None:
    """General test of the creation of the w_dd tensor."""
    # WHEN
    w_ud = sut.create_w_ud(w_dd, g_uu)

    # THEN: Verify that the w_dd contain K 1-forms. Verify that when i=j the value is 0
    for i, j in product(range(n ** 2 - 1), repeat=2):
        if i == j:
            assert w_ud[i, j] == 0
        else:
            assert w_ud[i, j] == 0 or _K_in_expr(n, w_ud[i, j])


@pytest.mark.parametrize("n", (2,))
def test_create_w_ud_n_equals_2(w_dd: Array, g_uu: Array) -> None:
    """Test the creation of w_du using hand calculations for n=2."""
    # WHEN
    w_ud = sut.create_w_ud(w_dd, g_uu)

    # THEN
    assert w_ud[0, 1] == expand(w_dd[0, 1] / x1)
    assert w_ud[1, 2] == expand(w_dd[1, 2] / x2)
    assert w_ud[2, 0] == expand(w_dd[2, 0] / x3)

    assert w_ud[1, 0] == expand(-1 * w_dd[0, 1] / x2)
    assert w_ud[2, 1] == expand(-1 * w_dd[1, 2] / x3)
    assert w_ud[0, 2] == expand(-1 * w_dd[2, 0] / x1)

    assert w_ud[0, 0] == 0
    assert w_ud[1, 1] == 0
    assert w_ud[2, 2] == 0


@pytest.fixture(name="w_ud")
def w_ud_fixture(w_dd: Array, g_uu: Array) -> Array:
    """Create w_ud for testing."""
    return sut.create_w_ud(w_dd, g_uu)


@pytest.mark.parametrize("n", (3,))
def test_create_w_wedge(n: int, w_ud: Array) -> None:
    """Test the creation of the w_wedge tensor."""
    # WHEN
    w_wedge = sut.create_w_wedge(n, w_ud)

    # THEN
    for i, j in product(range(n ** 2 - 1), repeat=2):
        expr = w_wedge[i, j]
        if i == j:
            assert expr == 0
        else:
            assert expr == 0 or _Wedge_of_K_in_expr(n, expr)


@pytest.mark.parametrize("n", (2,))
def test_create_w_wedge_n_equals_w(n: int, w_ud: Array) -> None:
    """Test the creation of the w_wedge against hand calculations for n=2."""
    # WHEN
    w_wedge = sut.create_w_wedge(n, w_ud)

    # THEN
    for i in range(n ** 2 - 1):
        assert w_wedge[i, i] == 0  # Should be zero when both indices are equal

    e_01 = (2 * x1 - 2 * x2 + x3) * (2 * x1 - 2 * x2 - x3)
    e_12 = (2 * x1 + 2 * x2 - x3) * (2 * x1 - 2 * x2 + x3)
    e_20 = (2 * x1 - 2 * x2 - x3) * (2 * x1 + 2 * x2 - x3)

    assert expand(w_wedge[0, 1]) == expand(e_01 / (8 * x1 * x3) * Wedge(K(0), K(1)))
    assert expand(w_wedge[1, 2]) == expand(-e_12 / (8 * x1 * x2) * Wedge(K(1), K(2)))
    assert expand(w_wedge[2, 0]) == expand(-e_20 / (8 * x2 * x3) * Wedge(K(0), K(2)))

    assert expand(w_wedge[1, 0]) == expand(-e_01 / (8 * x2 * x3) * Wedge(K(0), K(1)))
    assert expand(w_wedge[2, 1]) == expand(e_12 / (8 * x1 * x3) * Wedge(K(1), K(2)))
    assert expand(w_wedge[0, 2]) == expand(e_20 / (8 * x1 * x2) * Wedge(K(0), K(2)))


@pytest.mark.parametrize("n", (2,))
def test_differential_of_K_in_expr_basic(dK: List[Expr]) -> None:
    """Verify the utility function using a simple K 1-form."""
    # GIVEN
    k_0 = K(0)

    # WHEN
    result = sut._differential_of_K_expr(dK, k_0)

    # THEN
    assert result == dK[0]


@pytest.mark.parametrize("n", (2,))
def test_differential_of_K_in_linear_sum(dK: List[Expr]) -> None:
    """Verify the utility function using a simple K 1-form."""
    # GIVEN
    expr = a * K(0) + b * K(1)

    # WHEN
    result = sut._differential_of_K_expr(dK, expr)

    # THEN
    assert result == a * dK[0] + b * dK[1]


@pytest.mark.parametrize("n", (3,))
def test_create_dw_ud(n: int, w_ud: Array, dK: List[Expr]) -> None:
    """Test the creation of the dw_ud tensor."""
    # WHEN
    dw_ud = sut.create_dw_ud(w_ud, dK)

    # THEN
    for i, j in product(range(n ** 2 - 1), repeat=2):
        expr = dw_ud[i, j]
        if i == j:
            assert expr == 0
        else:
            assert expr == 0 or _Wedge_of_K_in_expr(n, expr)


@pytest.mark.parametrize("n", (2,))
def test_create_dw_ud_n_equals_2(n: int, w_ud: Array, dK: List[Expr]) -> None:
    """Test the creation of dw_ud against hand calculations for n=2."""
    # WHEN
    dw_ud = sut.create_dw_ud(w_ud, dK)

    # THEN
    for i in range(n ** 2 - 1):  # diagonal entries should be zero
        assert dw_ud[i, i] == 0

    assert expand(dw_ud[0, 1]) == expand(
        1 / (4 * x1) * (2 * x1 + 2 * x2 - x3) * Wedge(K(0), K(1))
    )
    assert expand(dw_ud[1, 2]) == expand(
        -1 / (2 * x2) * (2 * x1 - 2 * x2 - x3) * Wedge(K(1), K(2))
    )
    assert expand(dw_ud[2, 0]) == expand(
        -1 / (2 * x3) * (2 * x1 - 2 * x2 + x3) * Wedge(K(0), K(2))
    )

    assert expand(dw_ud[1, 0]) == expand(
        -1 / (4 * x2) * (2 * x1 + 2 * x2 - x3) * Wedge(K(0), K(1))
    )
    assert expand(dw_ud[2, 1]) == expand(
        1 / (2 * x3) * (2 * x1 - 2 * x2 - x3) * Wedge(K(1), K(2))
    )
    assert expand(dw_ud[0, 2]) == expand(
        1 / (2 * x1) * (2 * x1 - 2 * x2 + x3) * Wedge(K(0), K(2))
    )
