"""Solve the 3 non-linear equations to calculate the metric constants and lambda."""

import sys

from sympy import Eq, Expr, Mul, N, Pow, Rational, Symbol
from sympy.solvers import solve_poly_system

from cache import ricci
from metric import x1, x2, x3


if not len(sys.argv) > 1:
    print("Need to specify a value of n.")
    sys.exit(1)


n = int(sys.argv[1])
e_0, e_1, e_2 = ricci[n]

lmbda = Symbol("𝜆", real=True)


def create_equation(expr: Expr, var: Expr) -> Eq:
    """
    Create the equation equating expr = lambda * var.

    Substitute x2 = x1 to implement exchange symmetry.
    Cross-multiply the denominator to put all the vars in the numerator.
    """
    factor = Mul(
        *(arg for arg in expr.args if isinstance(arg, Rational) or isinstance(arg, Pow))
    )

    lhs = (expr / factor).subs({x2: x1, x3: 1})
    rhs = (lmbda * var / factor).subs({x2: x1, x3: 1})

    return Eq(lhs, rhs)


# Construct the equations from the defining relation of Einstein metrics:
# R_ab = lambda * g_ab
equations = [create_equation(e, v) for e, v in zip(ricci[n], (x1, x2, x3))]


for eqn in equations:
    print(eqn)

results = solve_poly_system(equations, lmbda, x1)
# from sympy.solvers import solve
# results = solve(equations, lmbda, x1, x2)

# Simplify the complicated (sqrt(2) and I containing) results using N if needed
# Rational results are presented as is
approx = [
    tuple(value if isinstance(value, Rational) else N(value) for value in res)
    for res in results
]

print()
for solution in approx:
    print(solution)
