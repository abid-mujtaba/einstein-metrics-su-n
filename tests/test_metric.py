"""Test the metric module."""

import metric as sut


def test_create_metric() -> None:
    """Test the creation of the metric."""
    # GIVEN
    n = 3

    # WHEN
    g_dd = sut.create_metric(n)

    # THEN
    assert g_dd.shape == (8, 8)

    assert g_dd[0,0] == sut.x1
    assert g_dd[3,3] == sut.x2
    assert g_dd[7,7] == sut.x3

    assert g_dd[0,1] == 0
    assert g_dd[1,0] == 0
