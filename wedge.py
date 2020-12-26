"""Enhance the WedgeProduct class to gain automatic ordering and better printing."""

from sympy import Expr
from sympy.core import Add


class Wedge(Expr):
    """
    Custom Wedge product class built specifically for the L and K 1-forms.

    Focuses on just 2-forms with automatic ordering to ensure anti-symmetric
    cancellation by default.
    """
    def __new__(cls, *args):
        assert len(args) == 2
        op1, op2 = args

        if op1 < op2:
            obj = Expr.__new__(cls, *args)

        elif op1 == op2:  # Wedge product of an element with itself is zero by definition
            return 0

        else:  # If op2 > op1 we reverse the order and multiply by -1 to implement
               # antisymmetry at construction
            obj = -1 * Wedge(op2, op1)

        return obj

    def _sympystr(self, printer, **kwargs):
        """Custom printer for the Wedge operation."""
        def s(op):
            """Convert operand of Wedge to string, adding parentheses if of type Add."""
            if isinstance(op, Add):
                return f"({op})"

            return str(op)

        return f"{s(self.args[0])} ∧ {s(self.args[1])}"
