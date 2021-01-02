"""The mappings between the K and L 1-forms, in both directions."""

import itertools
from sympy import Expr, I, Integer, Rational
from sympy.functions import sqrt
from sympy.matrices import ones, diag, eye, Matrix
from typing import Iterator, List

from L_1_forms import L
from mappings import create_C_matrix, create_P_matrix, create_Q_matrix


# Create the mixing C matrix
C = create_C_matrix()


def _category_1_K2L(n: int) -> Iterator[Expr]:
    """First n(n - 1)/2 mappings using the first row of the C matrix."""
    for a, b in itertools.combinations(range(n), 2):
        l = Matrix([L(a, b), L(b, a)])
        k = C * l

        yield k[0]


def _category_2_K2L(n: int) -> Iterator[Expr]:
    """Second n(n -1)/2 mappings using the second row of the C matrix."""
    for a, b in itertools.combinations(range(n), 2):
        l = Matrix([L(a, b), L(b, a)])
        k = C * l

        yield k[1]


def _category_3_K2L(n: int) -> Iterator[Expr]:
    """Create the category 3 K2L mappings."""
    # Create a columns vector from the L 1-forms which is used in the matrix based
    # creation of the mapping to K 1-forms
    l = Matrix([[L(i, i) for i in range(n)]]).T

    # create_P_matrix only works for N = (n - 1) > 2 so we provide two special cases
    if n == 2:
        P = eye(2)
    if n == 3:
        P = Matrix([[1, 1, 0], [1, -1, 0], [0, 0, 1]])
    else:
        P = create_P_matrix(n)

    Q = create_Q_matrix(n)

    k = P * Q * l

    return (k[i] for i in range(n))


def create_K2L(n: int) -> List[Expr]:
    """Create mappings from the K to the L 1-forms for SU(n)."""
    return [*_category_1_K2L(n), *_category_2_K2L(n), *_category_3_K2L(n)]
