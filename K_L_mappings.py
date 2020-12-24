"""The mappings between the K and L 1-forms, in both directions."""

import itertools
from sympy import Expr, I, Rational
from sympy.functions import sqrt
from sympy.matrices import ones, diag, eye, Matrix
from typing import List

from L_1_forms import L
from K_1_forms import K


def _category_1_K2L(n: int):
    """First n(n - 1)/2 mappings of the form L_A^B + L^B_A for A != B."""
    return (L(a, b) + L(b, a) for a, b in itertools.combinations(range(n), 2))


def _category_2_K2L(n: int):
    """Second n(n -1)/2 mappings of the form i*(L_A^B - L_B^A) for A != B."""
    return (I * (L(a, b) - L(b, a)) for a, b in itertools.combinations(range(n), 2))


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


def _create_Q_matrix(n: int) -> Matrix:
    """
    Create Q matrix used to mix L 1-forms while creating the category 3 K 1-forms.

    For the first (n - 1) rows the a-th row (1-indexed) has 1s in the first a entries,
    -a in the next entry and zeros after.
    The row is then divided by the sum of the squares of the entries making the row
    normalized (dot-product with self = 1).

    The last row is just a series of 1/sqrt(n).
    """

    def create_row(n: int, i: int) -> List[Expr]:
        """Create the i-th (0-indexed) row of the Q matrix."""
        a = i + 1  # 1-indexed row number
        norm = 1 / sqrt(a + a ** 2)  # Normalization factor

        return [*(norm for _ in range(a)), -a * norm, *(0 for _ in range(n - a - 1))]

    return Matrix(
        [
            *(create_row(n, i) for i in range(n - 1)),
            [1 / sqrt(n) for _ in range(n)],
        ]
    )


def _category_3_K2L(n: int):
    """Create the category 3 K2L mappings."""
    # Create a columns vector from the L 1-forms which is used in the matrix based
    # creation of the mapping to K 1-forms
    l = Matrix([[L(i, i) for i in range(n)]]).T

    P = _create_P_matrix(n)
    Q = _create_Q_matrix(n)

    k = P * Q * l

    return (k[i] for i in range(n))


def create_K2L(n: int):
    """Create mappings from the K to the L 1-forms for SU(n)."""
