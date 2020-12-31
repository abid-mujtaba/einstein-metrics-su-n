"""Common utilities for manipulating or analyzing symbolic expressions."""

from sympy import Expr
from sympy.core.basic import preorder_traversal

from K_1_forms import K
from wedge import Wedge


def is_in_expr(obj: Expr, expr: Expr) -> bool:
    """
    Determine whether the obj is inside the expr.

    If the obj is NOT a leaf (i.e. it has args, it combines other expressions) then
    raise a ValueError.
    """
    if obj.args:
        raise ValueError(f"obj {obj} is not a leaf, it has .args")

    for e in preorder_traversal(expr):
        if not e.args and e == obj:
            return True

    return False


def is_K_in_expr(n: int, expr: Expr) -> bool:
    """Verify that there is a K 1-form in the expression."""
    if isinstance(expr, K):
        return 0 <= expr.index < n ** 2

    if expr.args:
        return any(is_K_in_expr(n, arg) for arg in expr.args)

    return False


def is_Wedge_of_K_in_expr(n: int, expr: Expr) -> bool:
    """Verify that there is a Wedge of K 1-forms in the expression."""
    if isinstance(expr, Wedge):
        op1, op2 = expr.args

        return (isinstance(op1, K) and 0 <= op1.index < n ** 2) and (
            isinstance(op2, K) and 0 <= op2.index < n ** 2
        )

    if expr.args:
        return any(is_Wedge_of_K_in_expr(n, arg) for arg in expr.args)

    return False
