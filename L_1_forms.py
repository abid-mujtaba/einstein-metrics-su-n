"""
Create and manipulate the left-invariant 1-forms of SU(n).

These include the Hermitian L_A^B with the simple differential and
the K_i which are created from their liner combination to be both hermitian and
traceless.
"""

from sympy import AtomicExpr, Expr
from sympy.printing import StrPrinter
from typing import Any, Tuple


# Sub-class from AtomicExpr because the 1-forms are NOT expressions they are atomic
# operands
class L(AtomicExpr):  # type: ignore
    """
    L_A^B 1-forms for SU(n).

    These are neither Hermitian nor traceless. They in fact correspond to
    each element in an SU(n) matrix in that A, B ∊ 𝕫^n and
    correspond to the element in the A-th row and B-th column of an SU(n) matrix.
    """

    def __new__(cls, index_1: int, index_2: int) -> Expr:
        """
        Create a new L_A^B object which is a sub-class of sp.Expr.

        :param index_1: The first index of the 1-form (between 0 and (n-1))
        :param index_2: The second index of the 1-form (between 0 and (n-1))
        """
        return Expr.__new__(cls)

    __slots__ = ("index_1", "index_2")

    def __init__(self, index_1: int, index_2: int):
        """
        Create a new L_{index_1}^{index_2} object.Expr.

        :param index_1: The first index of the 1-form (between 0 and (n-1))
        :param index_2: The second index of the 1-form (between 0 and (n-1))
        """
        self.index_1 = index_1
        self.index_2 = index_2

    @property
    def is_number(self) -> bool:
        """Used by sympy to process expressions."""
        return False

    def _hashable_content(self) -> Tuple[int, int]:
        """The hashable (identifying) content of the L 1-form, mainly its indices."""
        return (self.index_1, self.index_2)

    def __eq__(self, other: object) -> bool:
        """Two L 1-forms are considered equal ONLY if they have same indices."""
        if not isinstance(other, L):
            return NotImplemented

        return self.index_1 == other.index_1 and self.index_2 == other.index_2

    def __lt__(self, other: object) -> bool:
        """One L 1-form is considered < another based on the indices."""
        if not isinstance(other, L):
            return NotImplemented

        if self.index_1 < other.index_1:
            return True

        if self.index_1 > other.index_1:
            return False

        return self.index_2 < other.index_2

    def __gt__(self, other: object) -> bool:
        """One L 1-form is considered > another based on the indices."""
        if not isinstance(other, L):
            return NotImplemented

        if self == other:
            return False

        return not (self < other)

    _unicode_subscripts = ("₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉")
    _unicode_superscripts = ("⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹")

    @classmethod
    def _get_subscript(cls, index: int) -> str:
        try:
            return cls._unicode_subscripts[index]

        except IndexError:  # If index is out of range use it without unicode
            return f"_{index}"

    @classmethod
    def _get_superscript(cls, index: int) -> str:
        try:
            return cls._unicode_superscripts[index]

        except IndexError:
            return f"^{index}"

    def __str__(self) -> str:
        """Print the L 1-form in its classic subscript+superscript form."""
        return (
            f"L{self._get_subscript(self.index_1)}{self._get_superscript(self.index_2)}"
        )

    def __repr__(self) -> str:
        """Falls back on __str__."""
        return self.__str__()

    def _sympystr(self, printer: StrPrinter, *args: Any) -> str:
        """Sympy method used when string-type printing an expression."""
        return self.__str__()

    def __hash__(self) -> int:
        """
        Unique hash based on indices.

        We use _hashable_contents() to calculate the has so that they remain in sync.
        """
        return hash(self._hashable_content())


typeL = Tuple[Tuple[L, ...], ...]


def create_L_1_forms(n: int) -> typeL:
    """Create the 2-array of the n² L 1-forms for SU(n)."""
    return tuple(tuple(L(i, j) for j in range(n)) for i in range(n))
