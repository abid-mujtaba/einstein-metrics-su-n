"""Solve the 3 non-linear equations to calculate the metric constants and lambda."""

import sys

from sympy import Eq, N, S, Rational, Symbol, expand
from sympy.solvers import solve, nonlinsolve, solve_poly_system

from cache import ricci
from metric import x1, x2, x3


if not len(sys.argv) > 1:
    print("Need to specify a value of n.")
    sys.exit(1)


n = int(sys.argv[1])
e_0, e_1, e_2 = ricci[n]

lmbda = Symbol("𝜆", real=True)

# Construct the equations from the defining relation of Einstein metrics:
# R_ab = lambda * g_ab
# We normalize the results by setting x3=1
# We multiply both sides of the equation by the denominator to remove inverse variables
equations = [
    Eq(expand(e_0.subs({x3: 1}) * 8 * x1 * x2), lmbda * x1 * 8 * x1 * x2),
    Eq(expand(e_1.subs({x3: 1}) * 16 * x1 ** 2), lmbda * x2 * 16 * x1 ** 2),
    Eq(expand(e_2.subs({x3: 1}) * 16 * x1 * x2), lmbda * 16 * x1 * x2),
]

for eqn in equations:
    print(eqn)

results = solve_poly_system(equations, lmbda, x1, x2)

# Simplify the complicated (sqrt(2) and I containing) results using N if needed
# Rational results are presented as is
approx = [
    tuple(value if isinstance(value, Rational) else N(value) for value in res)
    for res in results
]

print()
for solution in approx:
    print(solution)
