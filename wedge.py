"""Enhance the WedgeProduct class to gain automatic ordering and better printing."""

from more_itertools import partition
from sympy import Expr, Integer
from sympy.core import Add, Mul
from sympy.printing import StrPrinter
from typing import Any, List

from K_1_forms import K


class Wedge(Expr):  # type: ignore
    """
    Custom Wedge product class built specifically for the L and K 1-forms.

    Focuses on just 2-forms with automatic ordering to ensure anti-symmetric
    cancellation by default.
    """

    def __new__(cls, op1: Expr, op2: Expr) -> Expr:
        if op1 == op2:  # Wedge product of an element with itself is zero by definition
            return Integer(0)

        if op1 == 0 or op2 == 0:
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
        if isinstance(e, Add):
            args: List[Expr] = []

            for arg in e.args:
                if isinstance(arg, Add):
                    args = [*args, *arg.args]
                else:
                    args.append(arg)

            return sum(args)

        return e

    def _expand(wedge: Wedge) -> Expr:
        """Expand the operands of a Wedge."""
        op1, op2 = wedge.args

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


def extract_factor_K(expr: Expr) -> Expr:
    """
    For wedges of K 1-forms extract multiplicative factors outside the wedge.

    e.g. K(1) ^ (2 * K(2)) => 2 * (K(1) ^ K(2))
    """

    def _extract(wedge: Wedge) -> Expr:
        """Extract multiplicative factors of K 1-forms from Wedge."""
        op1, op2 = wedge.args

        if isinstance(op1, Mul) and any(isinstance(arg, K) for arg in op1.args):
            args_, match = partition(lambda e: isinstance(e, K), op1.args)
            op1_ = next(match)

            # Recursive call so that the right-operand is extracted as well
            return extract_factor_K(Mul(*args_, Wedge(op1_, op2)))

        if isinstance(op2, Mul) and any(isinstance(arg, K) for arg in op2.args):
            args_, match = partition(lambda e: isinstance(e, K), op2.args)
            op2_ = next(match)

            return Mul(*args_, Wedge(op1, op2_))

        return wedge

    if isinstance(expr, Wedge):  # Replace
        return _extract(expr)

    if expr.args:  # Descend and recurse
        return expr.func(*(extract_factor_K(arg) for arg in expr.args))

    return expr  # Return immutable leafs as is


def antisymm(expr: Expr) -> Expr:
    """
    Use antisymmetry of Wedge to consolidate.

    Each Wedge will be transformed in to the explicit order where the left operand is
    "less" than the right operand.

    e.g. K(2) ^ K(1) => - (K(1) ^ K(2))
    """

    def _antisymm(wedge: Wedge) -> Expr:
        """Carry out antisymmetric simplification of a Wedge."""
        op1, op2 = wedge.args

        if op2 < op1:  # Flip order and sign
            return -1 * Wedge(op2, op1)

        return wedge

    if isinstance(expr, Wedge):  # Replace
        return _antisymm(expr)

    if expr.args:  # Descend and recurse
        return expr.func(*(antisymm(arg) for arg in expr.args))

    return expr  # Return immutable leafs as is
