"""
Create the theta_tensor.

Use the definition: theta^a_b = dw^a_b +  w^a_c ^ w^c_b
"""

from sympy import Array


def create_theta_ud(dw_ud: Array, w_wedge_ud: Array) -> Array:
    """Create the theta_ud tensor."""
    return dw_ud + w_wedge_ud
