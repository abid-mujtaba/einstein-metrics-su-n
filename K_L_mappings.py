"""The mappings between the K and L 1-forms, in both directions."""

import itertools
from sympy import I, Rational
from sympy.matrices import ones, diag, eye, Matrix

from L_1_forms import L
from K_1_forms import K


def _category_1_K2L(n: int):
    """First n(n - 1)/2 mappings of the form L_A^B + L^B_A for A != B."""
    return (L(a, b) + L(b, a) for a, b in itertools.combinations(range(n), 2))


def _category_2_K2L(n: int):
    """Second n(n -1)/2 mappings of the form i*(L_A^B - L_B^A) for A != B."""
    return (I*(L(a,b) - L(b,a)) for a, b in itertools.combinations(range(n), 2))


def _create_P_matrix(n: int) -> Matrix:
    """
    Create P matrix used to mix L 1-forms while creating the category 3 K 1-forms.

    For N defined equal to (n - 1),
    the P matrix is block diagonal with a large (n - 1) * (n - 1) first entry
    and a single 1 in the last diagonal element

    The initial sub-matrix can be defined in Einstein notation as
    2 / N - delta_i,j
    where delta_i,j is the Kroneker delta.

    Both the structure of the larger matrix (block diagonal) and
    the sub-matrix can be defined in a straight-forward fashion using sympy's
    built in matrix utility functions.
    """
    N = n - 1
    sub_matrix = (Rational(2, N) * ones(N)) - eye(N)

    return diag(sub_matrix, 1)


def create_K2L(n: int):
    """Create mappings from the K to the L 1-forms for SU(n)."""

