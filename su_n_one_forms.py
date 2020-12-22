"""
Create and manipulate the left-invariant 1-forms of SU(n).

These include the Hermitian L_A^B with the simple differential and
the K_i which are created from their liner combination to be both hermitian and
traceless.
"""

import sympy as sp
import sympy.diffgeom as dg


class L(dg.Differential):
    """
    L_A^B 1-forms for SU(n).

    These are Hermitian but not traceless. They in fact correspond to each element in
    an SU(n) matrix in that A, B ∊ 𝕫^n and correspond to the element in the A-th row
    and B-th column of an SU(n) matrix.

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


    _unicode_subscripts =   ("₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉")
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
        return f"L{self._get_subscript(self.index_1)}{self._get_superscript(self.index_2)}"

    def __str__(self):
        """Falls back on __repr__."""
        return self.__repr__()
