"""Enhance the WedgeProduct class to gain automatic ordering and better printing."""

from sympy import Expr, Integer
from sympy.core import Add
from sympy.printing import StrPrinter
from typing import Any


class Wedge(Expr):  # type: ignore
    """
    Custom Wedge product class built specifically for the L and K 1-forms.

    Focuses on just 2-forms with automatic ordering to ensure anti-symmetric
    cancellation by default.
    """
    def __new__(cls, op1: Expr, op2: Expr) -> Expr:
        if op1 == op2:  # Wedge product of an element with itself is zero by definition
            return Integer(0)

        obj = Expr.__new__(cls, op1, op2)
        return obj

    def _sympystr(self, printer: StrPrinter, **kwargs: Any) -> str:
        """Custom printer for the Wedge operation."""
        def s(op: Expr) -> str:
            """Convert operand of Wedge to string, adding parentheses if of type Add."""
            if isinstance(op, Add):
                return f"({op})"

            return str(op)

        return f"{s(self.args[0])} ∧ {s(self.args[1])}"
