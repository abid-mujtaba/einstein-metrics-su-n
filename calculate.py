"""Carry out the full sequence of calculations to find Einstein Metrics."""

import dill
import sys

from c_tensor import create_c_ddd, create_c_ddu
from differentials import create_dK
from K_1_forms import create_K_u
from metric import create_metric, x1, x2, x3
from ricci import create_R_uddd, create_R_dd, calculate_Riem_2
from theta_tensor import create_theta_ud
from w_tensor import create_w_dd, create_w_ud, create_w_wedge_ud, create_dw_ud

# Choose the dimenions of the Group SU(n)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2

dim = n ** 2 - 1
m = int(n * (n - 1) / 2)

dK = create_dK(n)

g_dd, g_uu = create_metric(n)

c_ddu = create_c_ddu(dK, n)
c_ddd = create_c_ddd(c_ddu, g_dd)

K_u = create_K_u(n)

w_dd = create_w_dd(c_ddd, K_u)
w_ud = create_w_ud(w_dd, g_uu)

dw_ud = create_dw_ud(w_ud, dK)
w_wedge_ud = create_w_wedge_ud(n, w_ud)
theta_ud = create_theta_ud(dw_ud, w_wedge_ud)

R_uddd = create_R_uddd(n, theta_ud)
R_dd = create_R_dd(R_uddd)

assert len(set(R_dd[i, i] for i in range(dim))) == 3

e_00 = R_dd[0, 0]
e_11 = R_dd[m, m]
e_22 = R_dd[2 * m, 2 * m]

print(f"Calculation for n = {n}\n")
print("Unique elements of Ricci tensor:\n")
print(e_00)
print(e_11)
print(e_22)

# Serialize and save the R_uddd tensor
dill.settings["recurse"] = True
with open(f"R_uddd_n_{n}.dat", "wb") as fout:
    dill.dump(R_uddd, fout)
