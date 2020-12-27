"""Enhance the WedgeProduct class to gain automatic ordering and better printing."""

from sympy import Expr, Integer
from sympy.core import Add
from sympy.printing import StrPrinter
from typing import Any, List


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

        return f"({s(self.args[0])} ∧ {s(self.args[1])})"


def expand_K(expr: Expr) -> Expr:
    """
    Expand operands consisting of addition of K 1-forms in a Wedge.

    e.g. Wedge(K_1, K_2 + K_3) => Wedge(K_1, K_2) + Wedge(K_1, K_3)
    """
    def _collect_add(e: Add) -> Add:
        """
        Collect Add over Add into a single Add.

        e.g. (a + b) + (c + d) => (a + b + c + d)
        """
        args: List[Expr] = []

        for arg in e.args:
            if isinstance(arg, Add):
                args = [*args, *arg.args]
            else:
                args.append(arg)

        return sum(args)

    def _expand(wedge: Wedge) -> Expr:
        """Expand the operands of a Wedge."""
        op1 = wedge.args[0]
        op2 = wedge.args[1]

        if isinstance(op1, Add):
            # After op1 expansion recursively call expand_K to deal with op2
            return expand_K(sum(Wedge(arg, op2) for arg in op1.args))

        if isinstance(op2, Add):
            return sum(Wedge(op1, arg) for arg in op2.args)

        return wedge

    if isinstance(expr, Wedge):  # Replace
        # If both op1 and op2 were Add we will end up with an Add over Add which
        # should be collected in to a single Add
        return _collect_add(_expand(expr))

    if expr.args:  # Descend and recurse
        return expr.func(*(expand_K(arg) for arg in expr.args))

    return expr  # Return immutable leafs as is
