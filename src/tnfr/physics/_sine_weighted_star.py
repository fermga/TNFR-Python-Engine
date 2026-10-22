"""Fixed support shared by weighted-star state and transverse analyses."""

from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real

NODES = (0, 1, 2, 3, 4)
BLOCKS = ((0, 1, 2, 3), (4,))
INTERNAL_EDGES = ((0, 1), (0, 2), (0, 3))
EDGES = INTERNAL_EDGES + ((0, 4), (1, 4), (2, 4), (3, 4))
CONDUCTANCES = (3, 3, 3, 2, 2, 2, 2)
BARE_DEGREES = (4, 2, 2, 2, 4)
CAPACITY_FAMILY = "(4*s,2*s,2*s,2*s,8), held s"
LAW = "conservative_weighted_form_bare_sine_zero_loss_unit_exchange_beta"


def admit_scale(value):
    """Admit the fixed law's strictly positive held capacity scale."""
    scale = exact_or_represented_real(value, "capacity_scale")
    if scale <= 0:
        raise ValueError("capacity_scale must be strictly positive")
    return scale


def capacities(scale):
    """Materialize mathematical capacities after shared scalar admission."""
    scale = admit_scale(scale)
    return (4 * scale, 2 * scale, 2 * scale, 2 * scale, Q(8))


def internal_form_action(values):
    """Apply (3 L_star + 2 I) to an already admitted four-vector."""
    result = [2 * value for value in values]
    for i, j in INTERNAL_EDGES:
        current = 3 * (values[i] - values[j])
        result[i] += current
        result[j] -= current
    return tuple(result)
