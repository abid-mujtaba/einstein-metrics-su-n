"""Solve the 3 non-linear equations to calculate the metric constants and lambda."""

import dill
import sys

from sympy import Eq, Expr, Mul, N, Pow, Rational, Symbol, expand
from sympy.solvers import solve_poly_system

from cache import ricci
from metric import create_metric, x1, x2, x3
from ricci import calculate_Riem_2


if not len(sys.argv) > 1:
    print("Need to specify a value of n.")
    sys.exit(1)


def main() -> None:
    """Entrypoint."""
    n = int(sys.argv[1])
    lmbda = Symbol("𝜆", real=True)

    # We normalize the results by setting x3=2
    x3_val = 1


    def create_equation(expr: Expr, var: Expr) -> Eq:
        """
        Create the equation equating expr = lambda * var.

        Substitute for x3 to normalize the metric constants.
        Cross-multiply the denominator to put all the vars in the numerator.
        """
        factor = Mul(
            *(arg for arg in expr.args if isinstance(arg, Rational) or isinstance(arg, Pow))
        )

        lhs = (expr / factor).subs({x3: x3_val})
        rhs = (lmbda * var / factor).subs({x3: x3_val})

        return Eq(lhs, rhs)


    # Construct the equations from the defining relation of Einstein metrics:
    # R_ab = lambda * g_ab
    equations = [create_equation(e, v) for e, v in zip(ricci[n], (x1, x2, x3))]


    for eqn in equations:
        print(eqn)

    results = solve_poly_system(equations, lmbda, x1, x2)

    # Simplify the complicated (sqrt(2) and I containing) results using N if needed
    # Rational results are presented as is
    approx = [
        tuple(value if isinstance(value, Rational) else N(value) for value in res)
        for res in results
    ]

    print(f"\nNumber of solutions: {len(approx)}\n")

    for solution in approx:
        if all(isinstance(value, Rational) for value in solution):
            lmbda_s, x1_s, x2_s = solution
            x3_s = x3_val

            invariant = calculate_invariant(n, x1_s, x2_s, x3_s, lmbda_s)

            print(f"x₁: {x1_s}, x₂: {x2_s}, x₃: {x3_s}, 𝜆: {lmbda_s}, inv: {invariant}")


def calculate_invariant(n, x1_s, x2_s, x3_s, lmbda_s):
    """Calculate the invariant using Riem_2."""
    # We start by loading the serialized R_uddd expression
    with open(f"R_uddd_n_{n}.dat", "rb") as fin:
        R_uddd = dill.load(fin)

    g_dd, g_uu = create_metric(n)

    # Create a substitution dict
    subs = {x1: x1_s, x2: x2_s, x3: x3_s}

    # Substitute values in to R_uddd before we calculate Riem_2 (otherwise it takes
    # too long)
    R_uddd = R_uddd.applyfunc(lambda e : expand(e.subs(subs)))
    g_dd = g_dd.applyfunc(lambda e: e.subs(subs))
    g_uu = g_uu.applyfunc(lambda e: e.subs(subs))

    Riem_2 = calculate_Riem_2(R_uddd, g_dd, g_uu)

    return Riem_2 / (lmbda_s**2)


if __name__ == "__main__":
    main()
