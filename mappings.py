"""Construct common matrices for mapping K to L 1-forms and back."""

from sympy import Expr, Integer, Rational
from sympy.functions import sqrt
from sympy.matrices import Matrix, ones, eye, diag
from typing import List


def create_P_matrix(n: int) -> Matrix:
    """
    Create P matrix used to mix L 1-forms while creating the category 3 K 1-forms.

    For N defined equal to (n - 1),
    Only works for N > 2 (since N = 2 does not mix (zero diagonals))
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
    # assert N > 2  # N = 2 does not create the necessary mixing (zero diagonals)

    sub_matrix = (Rational(2, N) * ones(N)) - eye(N)

    return diag(sub_matrix, 1)


def create_Q_matrix(n: int) -> Matrix:
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

        return [
            *(norm for _ in range(a)),
            -a * norm,
            *(Integer(0) for _ in range(n - a - 1)),
        ]

    return Matrix(
        [
            *(create_row(n, i) for i in range(n - 1)),
            [1 / sqrt(n) for _ in range(n)],
        ]
    )
