"""Test the 𝜔_tensor.py module."""

import pytest

from sympy import sqrt, Array, Expr, expand
from typing import List

from c_tensor import create_c_ddu, create_c_ddd
from differentials import create_dK
from K_1_forms import create_K_u, K
from metric import create_metric, x1, x2, x3

import w_tensor as sut


@pytest.fixture(name="dK")
def dK_fixture(n:int) -> List[Expr]:
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


@pytest.mark.parametrize("n", (2,))
def test_create_w_dd_tensor_n_equals_2(c_ddd: Array, K_u: Array) -> None:
    """Test the values of 𝜔_dd against hand calculations for n=2."""
    # GIVEN (expected values)
    e_01 = -1 / sqrt(2) * (x1 + x2 - x3/2) * K(2)
    e_12 = 1 / sqrt(2) * (x1 - x2 - x3/2) * K(0)
    e_20 = -1 / sqrt(2) * (x1 - x2 + x3/2) * K(1)

    # WHEN
    w_dd = sut.create_w_dd(c_ddd, K_u)

    # THEN
    assert w_dd[0,1] == expand(e_01)
    assert w_dd[1,2] == expand(e_12)
    assert w_dd[2,0] == expand(e_20)

    assert w_dd[1,0] == expand(-1 * w_dd[0,1])
    assert w_dd[2,1] == expand(-1 * w_dd[1,2])
    assert w_dd[0,2] == expand(-1 * w_dd[2,0])

    assert w_dd[0,0] == 0
    assert w_dd[1,1] == 0
    assert w_dd[2,2] == 0
