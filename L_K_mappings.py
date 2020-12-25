"""
Create mappings from L 1-forms to K 1-forms.

These are the inverse of the mappings created in K_L_mappings.py.
"""

from sympy import I, S, Expr
from sympy.matrices import Matrix

from K_1_forms import K
from K_L_mappings import _create_P_matrix, _create_Q_matrix


def _L2K_a_less_b(a: int, b: int, n: int) -> Expr:
    """Create mapping from L_a^b to K 1-forms when a < b."""
    # Calculate the indices of the K_i which are created from L_a^b
    # One each in category 1 and 2
    # The exact form is given by inverting the transformation (a 2x2 matrix) that
    # is used to create the K 1-forms from the L in the first place
    m = int(n * (n - 1) / 2)
    i = a * n + b
    j = m + i

    return S.Half * K(i) - I * S.Half * K(j)


def _L2K_a_more_b(a: int, b: int, n: int) -> Expr:
    """Create mapping from L_a^b to K 1-forms when a < b."""
    # Calculate the indices of the K_i which are created from L_a^b
    # One each in category 1 and 2
    m = n * (n - 1) / 2
    i = a * n + b
    j = m + i

    return S.Half * K(i) + I * S.Half * K(j)


def _L_diag_mappings(n: int):
    """Generate the mappings for the diagonal L_a^a using the P and Q matrices."""
    P = _create_P_matrix(n)
    Q = _create_Q_matrix(n)

    invP = P.inv()
    invQ = Q.inv()

    # Create a column vector of the K 1-forms that correspond to the category 3
    # mappings to the diagonal L 1-forms.
    # These are the LAST n entries
    indices = [n**2 - i for i in range(n)]
    indices.reverse()

    k = Matrix([[K(i) for i in indices]]).T

    return invQ * invP * k


def _L2K_mapping(a: int, b: int, n: int, Ldiag: Matrix):
    """Create the mapping for L_a^b."""
    if a < b:
        return _L2K_a_less_b(a, b, n)

    if a > b:
        return _L2K_a_more_b(a, b, n)

    return Ldiag[a]
