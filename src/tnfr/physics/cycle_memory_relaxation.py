"""Connect fixed-cycle nodal diffusion to the existing REMESH policy theorem.

This exact reference requires uniform positive held capacity and vanishing
phase/capacity/topology pressures. It derives conservative full-spectrum
gains, then reuses the shared sealed delayed-history envelope. It does not
admit an operator word or identify a runtime graph, history or solver.
"""

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

from .._exact_time import exact_or_represented_real as _rational
from ..constants import DEFAULTS
from ..utils import normalize_weights
from ._cycle_algebra import Vector, ordered_vector
from .remesh_history_stability import (
    UniformRemeshHistoryStabilityCertificate,
    certify_uniform_remesh_history_stability,
)
from .remesh_schedule_policy_stability import (
    UniformRemeshSchedulePolicyStabilityCertificate,
    certify_uniform_remesh_schedule_policy_stability,
)

__all__ = ["CycleMemoryRelaxationReference", "certify_cycle_memory_relaxation"]


@dataclass(frozen=True)
class CycleMemoryRelaxationReference:
    """Declared exact cycle geometry and two retained history-policy proofs.

    Both policies use the ordinary centered Euclidean energy, equivalent
    to the reversible metric up to a scalar on this uniform-capacity cycle.
    Histories hold pre-REMESH, post-schedule fields, with newest row first
    in the companion. Full state convergence needs additional mean control.
    """

    node_count: int
    capacity: Fraction
    epi_weight: Fraction
    step_sizes: Vector
    cycle_duration: Fraction
    spectral_gap_lower_bound: Fraction
    euler_step_coefficients: Vector
    euler_energy_gain_upper_bound: Fraction
    continuous_energy_gain_upper_bound: Fraction
    remesh_certificate: UniformRemeshHistoryStabilityCertificate
    euler_policy: UniformRemeshSchedulePolicyStabilityCertificate
    continuous_policy: UniformRemeshSchedulePolicyStabilityCertificate

    @property
    def scope(self) -> str:
        return (
            "Exact fixed unit-cycle diffusion with uniform positive capacity, "
            "zero other channel pressures and the declared refreshed Euler "
            "partition or exact continuous flow. Reuses the uniform unclipped "
            "REMESH envelope for pre-REMESH histories. No live graph, grammar "
            "admission, history provenance, clipping/rounding, solver accuracy "
            "or future binary64 execution is certified."
        )


def _integer(value, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def certify_cycle_memory_relaxation(
    node_count,
    *,
    capacity=1,
    epi_weight=None,
    step_sizes=(Fraction(1, 4), Fraction(1, 4)),
    alpha=None,
    tau_local=1,
    tau_global=1,
) -> CycleMemoryRelaxationReference:
    """Derive exact diffusion gains and reuse the shared REMESH envelope.

    On C_n, lambda_2=2*sin(pi/n)^2 >= 8/n^2. For every represented exact
    Euler coefficient kappa=h*e*nu in (0,1/2], all modal factors are
    nonnegative. Multiplying (1-kappa*8/n^2)^2 bounds the complete schedule's
    disagreement-energy gain. Exact continuous flow instead has the rational
    upper bound 1/(1+2*e*nu*T*8/n^2), from exp(-u)<=1/(1+u).

    All step sizes must be positive; they repeat each cycle in this reference.
    Explicit effective coefficients are used as supplied. Canonical runtime
    identification requires the actual normalized full-channel EPI weight,
    uniform capacity and zero remaining channel pressures. The omitted weight
    uses existing full-channel normalization; omitted alpha uses the configured
    repository default as a fixed reference, not an inferred runtime choice.
    Delay one is an explicitly selected history domain, not a physical time.
    The algebraic alpha=0 boundary is supported by the source theorem but is
    not an admitted runtime REMESH coefficient. Omit the delayed operation
    when executing a memory-free control.
    """
    count = _integer(node_count, "node_count", 3)
    nu = _rational(capacity, "capacity")
    if epi_weight is None:
        epi_weight = normalize_weights(
            DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo")
        )["epi"]
    e = _rational(epi_weight, "epi_weight")
    if nu <= 0 or e <= 0:
        raise ValueError("capacity and epi_weight must be strictly positive")
    steps = ordered_vector(step_sizes, "step_sizes")
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("step_sizes must contain positive physical durations")
    coefficients = tuple(step * e * nu for step in steps)
    if any(value > Fraction(1, 2) for value in coefficients):
        raise ValueError("each h*epi_weight*capacity must be at most 1/2")
    local = _integer(tau_local, "tau_local", 1)
    global_ = _integer(tau_global, "tau_global", 1)
    alpha_q = _rational(
        DEFAULTS["REMESH_ALPHA"] if alpha is None else alpha, "alpha"
    )
    history = certify_uniform_remesh_history_stability(
        alpha=alpha_q, tau_local=local, tau_global=global_
    )
    gap = Fraction(8, count**2)
    horizon = sum(steps, Fraction(0))
    euler_gain = Fraction(1)
    for coefficient in coefficients:
        euler_gain *= (1 - coefficient * gap) ** 2
    continuous_gain = 1 / (1 + 2 * e * nu * horizon * gap)
    euler = certify_uniform_remesh_schedule_policy_stability(history, euler_gain)
    continuous = certify_uniform_remesh_schedule_policy_stability(
        history, continuous_gain
    )
    return CycleMemoryRelaxationReference(
        count, nu, e, steps, horizon, gap, coefficients, euler_gain,
        continuous_gain, history, euler, continuous,
    )
