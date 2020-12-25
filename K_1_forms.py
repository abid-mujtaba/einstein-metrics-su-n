"""Implement the Hermitian and traceless type K 1-forms."""

import sympy as sp

from typing import Tuple


class K(sp.Expr):
    """
    K_i 1-forms for SU(n).

    These are Hermitian and traceless by construction (from the L 1-forms).
    The are (n²-1) in number.
    """

    # __slots__ = ("index", *sp.Basic.__slots__)

    def __new__(cls, index: int):
        """
        Create a new K_i 1-form which is a sub-class of dg.Differential.

        :param index: The index of the 1-form between 0 and (n² - 2)
        """
        obj = sp.Expr.__new__(cls)
        obj.index = index
        obj._args = tuple()
        return obj

    @property
    def is_number(self):
        """Used by sympy to process expressions."""
        return False

    def _hashable_content(self):
        """The hashable (identifying) content of the K 1-form, mainly its index."""
        return (self.index,)

    def __eq__(self, other: object) -> bool:
        """Two K 1-forms are considered equal ONLY if they have the same index."""
        if not isinstance(other, K):
            return NotImplemented

        return self.index == other.index


    _unicode_subscripts = ("₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉")

    @classmethod
    def _index_to_unicode(cls, index: int) -> str:
        """Convert an integer to a subscript unicode string."""
        return "".join(cls._unicode_subscripts[int(_)] for _ in str(index))

    def __str__(self):
        """Print the K 1-form in its classic K_{index} form."""
        return f"K{self._index_to_unicode(self.index)}"

    def __repr__(self):
        """Same as str."""
        return self.__str__()

    def _sympystr(self, printer, *args):
        """Sympy method used when string-type printing an expression."""
        return self.__str__()

    def __hash__(self):
        """Unique hash based on _hashable_contents(), mainly index."""
        return hash(self._hashable_content())



# The K 1-forms are returned as a tuple of expressions (sums of L 1-forms)
typeK = Tuple[sp.Expr, ...]


# def create_K_1_forms(L: typeL) -> typeK:
#     """Create the 1-array of n² K 1-forms for SU(n)."""
#     n = len(L)

#     class_1 = (L[i][j] + L[j][i] for i in range(n) for j in range(n) if i != j)

#     return tuple(class_1)
