"""Create the metric tensor and its inverse in the K 1-form space."""

from sympy import Array, Symbol
from sympy.matrices import eye


# Create the metric constants (one for each category for the K 1-forms)
x1 = Symbol("x₁")
x2 = Symbol("x₂")
x3 = Symbol("x₃")


def create_metric(n: int) -> Array:
    """Create the metric tensor in the K 1-form space for SU(n)."""
    dim = n**2 - 1  # Dimension of the space
    m = int(n * (n - 1) / 2)  # Size of categories 1 and 2

    # Start with a diagonal matrix
    e = eye(dim)

    for i in range(m):
        e[i, i] = x1
        e[m + i, m + i] = x2

    for i in range(n - 1):  # Category 3 K 1-forms are (n - 1) in number after you
                            # ignore the K_{n**2} 1-form which is NOT a part of SU(n)
        e[2 * m + i, 2 * m + i] = x3

    return Array(e)
