"""Shared fixtures for tests."""

import pytest
from sympy import Array, Expr
from typing import List

from c_tensor import create_c_ddd, create_c_ddu
from differentials import create_dK
from K_1_forms import create_K_u
from metric import create_metric
from w_tensor import create_w_dd, create_w_ud


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


@pytest.fixture(name="w_dd")
def w_dd_fixture(c_ddd: Array, K_u: Array) -> Array:
    """Create w_dd for SU(n)."""
    return create_w_dd(c_ddd, K_u)


@pytest.fixture(name="g_uu")
def g_uu_fixture(n: int) -> Array:
    """Create the inverse metric tensor."""
    _, g_uu = create_metric(n)

    return g_uu


@pytest.fixture(name="w_ud")
def w_ud_fixture(w_dd: Array, g_uu: Array) -> Array:
    """Create w_ud for testing."""
    return create_w_ud(w_dd, g_uu)
