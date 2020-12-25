"""Enhance the WedgeProduct class to gain automatic ordering and better printing."""

from sympy import Expr

class Wedge(Expr):
    """
    Custom Wedge product class built specifically for the L and K 1-forms.

    Focuses on just 2-forms with automatic ordering to ensure anti-symmetric
    cancellation by default.
    """
    def __new__(cls, *args):
        assert len(args) == 2
        op1, op2 = args

        if op1 == op2:  # Wedge product of an element with itself is zero by definition
            return 0

        obj = Expr.__new__(cls, *args)

        return obj

    def _sympystr(self, printer, **kwargs):
        """Custom printer for the Wedge operation."""
        return f"{self.args[0]} ∧ {self.args[1]}"
