"""The mappings between the K and L 1-forms, in both directions."""

import itertools
from sympy import I

from L_1_forms import L
from K_1_forms import K


def _category_1_K2L(n: int):
    """First n(n - 1)/2 mappings of the form L_A^B + L^B_A for A != B."""
    return (L(a, b) + L(b, a) for a, b in itertools.combinations(range(n), 2))


def _category_2_K2L(n: int):
    """Second n(n -1)/2 mappings of the form i*(L_A^B - L_B^A) for A != B."""
    return (I*(L(a,b) - L(b,a)) for a, b in itertools.combinations(range(n), 2))


def create_K2L(n: int):
    """Create mappings from the K to the L 1-forms for SU(n)."""

