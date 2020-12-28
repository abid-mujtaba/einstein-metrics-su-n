"""Test the metric module."""

from sympy.tensor import tensorcontraction as tc, tensorproduct as tp
import metric as sut


def test_create_metric() -> None:
    """Test the creation of the metric, and its inverse."""
    # GIVEN
    n = 3

    # WHEN
    g_dd, g_uu = sut.create_metric(n)

    # Test inversion using tensor contraction
    kron_ud = tc(tp(g_uu, g_dd), (1, 2))

    # Contraction of the kronecker delta
    scalar = tc(kron_ud, (0,1))

    # THEN
    assert g_dd.shape == (8, 8)
    assert g_uu.shape == (8, 8)

    assert g_dd[0,0] == sut.x1
    assert g_dd[3,3] == sut.x2
    assert g_dd[7,7] == sut.x3

    assert g_dd[0,1] == 0
    assert g_dd[1,0] == 0

    assert g_uu[0,0] == 1 / sut.x1
    assert g_uu[3,3] == 1 / sut.x2
    assert g_uu[7,7] == 1 / sut.x3

    assert g_uu[0,1] == 0
    assert g_uu[1,0] == 0

    assert kron_ud.shape == (8, 8)

    for i in range(8):
        assert kron_ud[i,i] == 1

    assert kron_ud[0,1] == 0
    assert kron_ud[1,0] == 0

    assert scalar == 8
