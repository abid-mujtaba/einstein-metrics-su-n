"""Implement the Hermitian and traceless type K 1-forms."""

from sympy import Array, Expr, AtomicExpr
from sympy.printing import StrPrinter
from typing import Any, Tuple


# Sub-class from AtomicExpr because the 1-forms are NOT expressions they are atomic
# operands
class K(AtomicExpr):  # type: ignore
    """
    K_i 1-forms for SU(n).

    These are Hermitian and traceless by construction (from the L 1-forms).
    The are (n²-1) in number.
    """

    def __new__(cls, index: int) -> Expr:
        """Create a new K_i 1-form which is a sub-class of sp.Expr."""
        return Expr.__new__(cls)

    __slots__ = ("index",)

    def __init__(self, index: int):
        """
        Create a new K_{index} 1-form.

        :param index: The index of the 1-form between 0 and (n² - 1)
        """
        self.index = index

    @property
    def is_number(self) -> bool:
        """Used by sympy to process expressions."""
        return False

    def _hashable_content(self) -> Tuple[int]:
        """The hashable (identifying) content of the K 1-form, mainly its index."""
        return (self.index,)

    def __eq__(self, other: object) -> bool:
        """Two K 1-forms are considered equal ONLY if they have the same index."""
        if not isinstance(other, K):
            return NotImplemented

        return self.index == other.index

    def __lt__(self, other: object) -> bool:
        """A K 1-form is considered < another ONLY if the first has a lower index."""
        if not isinstance(other, K):
            return NotImplemented

        return self.index < other.index

    def __gt__(self, other: object) -> bool:
        """A K 1-form is considered > another ONLY if the first has a larger index."""
        if not isinstance(other, K):
            return NotImplemented

        return self.index > other.index

    _unicode_subscripts = ("₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉")

    @classmethod
    def _index_to_unicode(cls, index: int) -> str:
        """Convert an integer to a subscript unicode string."""
        return "".join(cls._unicode_subscripts[int(_)] for _ in str(index))

    def __str__(self) -> str:
        """Print the K 1-form in its classic K_{index} form."""
        return f"K{self._index_to_unicode(self.index)}"

    def __repr__(self) -> str:
        """Same as str."""
        return self.__str__()

    def _sympystr(self, printer: StrPrinter, *args: Any) -> str:
        """Sympy method used when string-type printing an expression."""
        return self.__str__()

    def __hash__(self) -> int:
        """Unique hash based on _hashable_contents(), mainly index."""
        return hash(self._hashable_content())


def create_K_u(n: int) -> Array:
    """Create K_u tensor of K 1-forms."""
    return Array([K(i) for i in range(n ** 2 - 1)])
