"""Calculate the Riemann Curvature and Ricci tensors."""

from sympy import Array, Expr

from wedge import extract_wedge_coeff


def _coeff(expr: Expr, c: int, d: int) -> Expr:
    """
    Extract the coeff of K^c ^ K^d in the specified expression.

    We divide by 1/2 to separate the coefficient into two antisymmetric parts.
    """
    if c == d:
        return 0

    if c < d:
        return extract_wedge_coeff(expr, c, d) / 2

    else:
        return -1 * extract_wedge_coeff(expr, d, c) / 2


def create_R_uddd(n: int, theta_ud: Array) -> Array:
    """Create the Riemann Curvature Tensor."""
    dim = n ** 2 - 1

    return Array(
        [
            [
                [[_coeff(theta_ud[a, b], c, d) for d in range(dim)] for c in range(dim)]
                for b in range(dim)
            ]
            for a in range(dim)
        ]
    )
