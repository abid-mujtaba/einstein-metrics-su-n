"""Test the L2K mappings module."""

from itertools import product
from sympy import I, expand, collect, simplify

import L_K_mappings as sut

from L_1_forms import L
from K_1_forms import K
from K_L_mappings import create_K2L
from utilities import is_in_expr


def test_L2K_a_less_b():
    """Test the calculation for L^a_b when a < b."""
    # GIVEN
    n = 4
    a = 1
    b = 2

    # WHEN
    result = sut._L2K_a_less_b(a, b, n)

    # THEN
    assert is_in_expr(K(3), result)
    assert is_in_expr(K(9), result)

    assert result == K(3) / 2 - I * K(9) / 2


def test_L2K_a_more_b():
    """Test the calculation for L^a_b when a < b."""
    # GIVEN
    n = 4
    a = 2
    b = 1

    # WHEN
    result = sut._L2K_a_more_b(a, b, n)

    # THEN
    assert is_in_expr(K(3), result)
    assert is_in_expr(K(9), result)

    assert result == K(3) / 2 + I * K(9) / 2


def test_L_diag_mappings():
    """Test the creation of the diagonal L 1-form mappings using inverse of P and Q."""
    # GIVEN
    n = 4

    # WHEN
    L_diag = sut._L_diag_mappings(n)

    # THEN
    assert len(L_diag) == n

    for i, j in zip(range(n, 0, -1), range(n)):
        assert is_in_expr(K(n ** 2 - i), L_diag[j])

    for l in L_diag:
        assert not is_in_expr(K(16), l)


def test_create_L2K():
    """Test the full method for creating L2K mappings."""
    # GIVEN
    n = 4

    # WHEN
    L2K = sut.create_L2K(n)

    # THEN
    assert len(L2K) == n
    for i in range(n):
        assert len(L2K[i]) == n


def test_mapping_inversion_off_diagonal():
    """Map a K 1-form to the off-diagonal L 1-forms and back to confirm mappings."""
    # GIVEN
    n = 4
    K2L = create_K2L(n)
    L2K = sut.create_L2K(n)

    k_0 = K2L[0]  # the first K 1-form mapped to L 1-forms

    l_0_1 = L(0, 1)
    l_1_0 = L(1, 0)

    # WHEN
    k_0_inverted = k_0.subs({l_0_1: L2K[0][1], l_1_0: L2K[1][0]})

    # THEN
    assert is_in_expr(l_0_1, k_0)
    assert is_in_expr(l_1_0, k_0)

    assert k_0_inverted == K(0)


def test_mapping_inverstion_diagonal():
    """Map a K 1-form to the diagonal L 1-forms and back to confirm mappings."""
    # GIVEN
    n = 4
    K2L = create_K2L(n)
    L2K = sut.create_L2K(n)

    k_15 = K2L[15]

    l_0_0 = L(0, 0)
    l_1_1 = L(1, 1)
    l_2_2 = L(2, 2)
    l_3_3 = L(3, 3)

    # WHEN
    k_15_inverted = k_15.subs(
        {l_0_0: L2K[0][0], l_1_1: L2K[1][1], l_2_2: L2K[2][2], l_3_3: L2K[3][3]}
    )

    # THEN
    assert is_in_expr(l_0_0, k_15)
    assert is_in_expr(l_1_1, k_15)
    assert is_in_expr(l_2_2, k_15)
    assert is_in_expr(l_3_3, k_15)

    assert expand(k_15_inverted) == K(15)


def test_mapping_inversion_complete():
    """Map all K 1-forms to L 1-forms and back to verify mappings completely."""
    # GIVEN
    n = 4
    K2L = create_K2L(n)
    L2K = sut.create_L2K(n)

    k = [K2L[i] for i in range(n**2)]
    maps = {L(i,j): L2K[i][j] for i, j in product(range(n), repeat=2)}

    # WHEN
    k_inv = [e.subs(maps) for e in k]

    # THEN
    for i in range(n**2):
        assert expand(k_inv[i]) == K(i)
