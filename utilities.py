"""Common utilities for manipulating or analyzing symbolic expressions."""

from sympy import Expr
from sympy.core.basic import preorder_traversal


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
