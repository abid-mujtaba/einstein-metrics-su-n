"""Shared fixtures for tests."""

import pytest
from sympy import Array, Expr
from typing import List

from c_tensor import create_c_ddd, create_c_ddu
from differentials import create_dK
from K_1_forms import create_K_u
from metric import create_metric
from ricci import create_R_uddd
from theta_tensor import create_theta_ud
from w_tensor import create_w_dd, create_w_ud, create_w_wedge_ud, create_dw_ud


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


@pytest.fixture(name="g_dd")
def g_dd_fixture(n: int) -> Array:
    """Create the metric tensor."""
    g_dd, _ = create_metric(n)

    return g_dd


@pytest.fixture(name="g_uu")
def g_uu_fixture(n: int) -> Array:
    """Create the inverse metric tensor."""
    _, g_uu = create_metric(n)

    return g_uu


@pytest.fixture(name="w_ud")
def w_ud_fixture(w_dd: Array, g_uu: Array) -> Array:
    """Create w_ud for testing."""
    return create_w_ud(w_dd, g_uu)


@pytest.fixture(name="dw_ud")
def dw_ud_fixture(n: int, w_ud: Array, dK: List[Expr]) -> Array:
    """Create dw_ud for testing."""
    return create_dw_ud(w_ud, dK)


@pytest.fixture(name="w_wedge_ud")
def w_wedge_ud_fixture(n: int, w_ud: Array) -> Array:
    """Create w_wedge_ud for testing."""
    return create_w_wedge_ud(n, w_ud)


@pytest.fixture(name="theta_ud")
def theta_ud_fixture(n: int, dw_ud: Array, w_wedge_ud: Array) -> Array:
    """Create theta_ud for testing."""
    return create_theta_ud(dw_ud, w_wedge_ud)


@pytest.fixture(name="R_uddd")
def R_uddd_fixture(n: int, theta_ud: Array) -> Array:
    """Create R_uddd for testing."""
    return create_R_uddd(n, theta_ud)
