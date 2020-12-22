"""
Create and manipulate the left-invariant 1-forms of SU(n).

These include the Hermitian L_A^B with the simple differential and
the K_i which are created from their liner combination to be both hermitian and
traceless.
"""

import sympy as sp
import sympy.diffgeom as dg

from typing import Literal, Tuple


class L(dg.Differential):
    """
    L_A^B 1-forms for SU(n).

    These are neither Hermitian nor traceless. They in fact correspond to
    each element in an SU(n) matrix in that A, B ∊ 𝕫^n and
    correspond to the element in the A-th row and B-th column of an SU(n) matrix.

    They are based on a `None` form_field which means that
    most methods of the Differential form will **not** work.
    """

    def __new__(cls, index_1: int, index_2: int):
        """
        Create a new L_A^B object which is a sub-class of dg.Differential.

        :param index_1:
        """
        obj = dg.Differential.__new__(cls, None)

        obj.index_1 = index_1
        obj.index_2 = index_2

        return obj

    def _hashable_content(self):
        """The hashable (identifying) content of the L 1-form, mainly its indices."""
        return (self.index_1, self.index_2)

    def __eq__(self, other: object) -> bool:
        """Two L 1-forms are considered equal ONLY if they have same indices."""
        if not isinstance(other, L):
            return NotImplemented

        return self.index_1 == other.index_1 and self.index_2 == other.index_2

    def __lt__(self, other: object) -> bool:
        """An L 1-form is < another based on first and then second index."""
        if not isinstance(other, L):
            return NotImplemented

        return self.index_1 < other.index_1 or (
            self.index_1 == other.index_1 and self.index_2 < other.index_2
        )

    def compare(self, other: object) -> Literal[-1, 0, 1]:
        """sympy makes heavy use of the compare method when organizing expressions."""
        if not isinstance(other, L):
            return NotImplemented

        if self < other:
            return -1

        if self == other:
            return 0

        return 1


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

    def __repr__(self):
        """Represent the L 1-form in its classic subscript+superscript form."""
        return (
            f"L{self._get_subscript(self.index_1)}{self._get_superscript(self.index_2)}"
        )

    def __str__(self):
        """Falls back on __repr__."""
        return self.__repr__()

    def __hash__(self):
        """Use the representation of the L 1-form since it is complete."""
        return hash(repr(self))


typeL = Tuple[Tuple[L, ...], ...]

def create_su_n_L_1_forms(n: int) -> typeL:
    """Create the 2-array of the n² L 1-forms for SU(n)."""
    return tuple(tuple(L(i, j) for j in range(n)) for i in range(n))
