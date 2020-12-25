"""
Create and manipulate the left-invariant 1-forms of SU(n).

These include the Hermitian L_A^B with the simple differential and
the K_i which are created from their liner combination to be both hermitian and
traceless.
"""

import sympy as sp

from typing import Tuple


class L(sp.Expr):
    """
    L_A^B 1-forms for SU(n).

    These are neither Hermitian nor traceless. They in fact correspond to
    each element in an SU(n) matrix in that A, B ∊ 𝕫^n and
    correspond to the element in the A-th row and B-th column of an SU(n) matrix.
    """

    def __new__(cls, index_1: int, index_2: int):
        """
        Create a new L_A^B object which is a sub-class of dg.Differential.

        :param index_1: The first index of the 1-form (between 0 and (n-1))
        :param index_2: The second index of the 1-form (between 0 and (n-1))
        """
        obj = sp.Expr.__new__(cls)

        obj.index_1 = index_1
        obj.index_2 = index_2

        obj._args = tuple()  # By definition the L 1-forms do not have expressions
                             # inside. .args is a property so have to override the
                             # underlying _args attribute

        return obj

    @property
    def is_number(self):
        """Used by sympy to process expressions."""
        return False

    def _hashable_content(self):
        """The hashable (identifying) content of the L 1-form, mainly its indices."""
        return (self.index_1, self.index_2)

    def __eq__(self, other: object) -> bool:
        """Two L 1-forms are considered equal ONLY if they have same indices."""
        if not isinstance(other, L):
            return NotImplemented

        return self.index_1 == other.index_1 and self.index_2 == other.index_2


    _unicode_subscripts = ("₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉")
    _unicode_superscripts = ("⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹")

    @classmethod
    def _get_subscript(cls, index):
        try:
            return cls._unicode_subscripts[index]

        except IndexError:  # If index is out of range use it without unicode
            return f"_{index}"

    @classmethod
    def _get_superscript(cls, index):
        try:
            return cls._unicode_superscripts[index]

        except IndexError:
            return f"^{index}"

    def __str__(self):
        """Print the L 1-form in its classic subscript+superscript form."""
        return (
            f"L{self._get_subscript(self.index_1)}{self._get_superscript(self.index_2)}"
        )

    def __repr__(self):
        """Falls back on __str__."""
        return self.__str__()

    def _sympystr(self, printer, *args):
        """Sympy method used when string-type printing an expression."""
        return self.__str__()

    def __hash__(self):
        """
        Unique hash based on indices.

        We use _hashable_contents() to calculate the has so that they remain in sync.
        """
        return hash(self._hashable_content())


typeL = Tuple[Tuple[L, ...], ...]

def create_L_1_forms(n: int) -> typeL:
    """Create the 2-array of the n² L 1-forms for SU(n)."""
    return tuple(tuple(L(i, j) for j in range(n)) for i in range(n))
