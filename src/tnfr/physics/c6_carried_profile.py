"""Canonical C6 relative profiles and exact carried-step decompositions.

The existing pressure producer, forced-support Poisson solver and carried
nodal kernel supply every model term. These detached observations certify
their supplied numerical inputs, not a graph history or future phase law.
Centered shape and arithmetic mean remain separate dynamical coordinates.
"""

from dataclasses import dataclass
from fractions import Fraction

from ..dynamics._euler_kernel import (
    NodalRemainderStep, _validate_nodal_remainder_state, advance_nodal_remainder,
)
from ._cycle_algebra import Vector, laplacian_action
from .c6_pressure_lattice import (
    C6PressureLatticeReference, C6PressureLatticeObservation,
    _rebuild_lattice, _observe_rebuilt_c6_pressure_lattice,
)
from .forced_support import (
    ForcedSupportBalance, derive_forced_support_balance,
    observe_forced_support_pattern,
)
from .nodal_remainder_pressure import (
    NodalRemainderPressureReadout, observe_nodal_remainder_pressure_readout,
)
from .support_transport import _from_data

__all__ = [
    "C6CarriedProfile", "derive_c6_carried_profile",
    "C6CarriedProfileStep", "observe_c6_carried_profile_step",
]


def _mean(values: Vector) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


def _center(values: Vector) -> Vector:
    mean = _mean(values)
    return tuple(value - mean for value in values)


@dataclass(frozen=True, slots=True)
class C6CarriedProfile:
    """Fixed CPU phase source and its canonical centered Poisson profile.

    Unit C6 has H=2I. The shared forced-support solver therefore supplies
    the arithmetic-centered z satisfying w*L*z=A-mean(A)*1. A nonzero
    mean(A) produces a drifting relative profile, not an equilibrium.
    The reference makes no claim that its phase tuple is reachable or held
    by a live operator word. Public caches are rebuilt before observation.
    """

    lattice: C6PressureLatticeReference
    forced_balance: ForcedSupportBalance

    @property
    def graph_provenance_certified(self) -> bool:
        return False


def derive_c6_carried_profile(lattice: C6PressureLatticeReference) -> C6CarriedProfile:
    """Reuse the exact fixed-support solver with the actual CPU phase source.

    Rebuilding the lattice re-evaluates its canonical zero-EPI phase probe.
    The transport snapshot below retains that detached probe; its zero EPI
    is not a positive-band preparation or a graph write. Unit capacities,
    conductances and the ordered six-cycle are exactly the lattice scope.
    """
    lattice = _rebuild_lattice(lattice)
    neighbors = tuple(((i - 1) % 6, (i + 1) % 6) for i in range(6))
    edges = tuple((i, j, Fraction(1)) for i, row in enumerate(neighbors) for j in row)
    forcing = tuple(Fraction(value) for value in lattice.sources)
    snapshot = _from_data(
        tuple(range(6)), edges, neighbors, (Fraction(0),) * 6,
        (Fraction(1),) * 6, forcing,
    )
    balance = derive_forced_support_balance(
        snapshot, epi_weight=lattice.source.epi_weight, forcing=forcing,
    )
    if balance.metric_weights != (Fraction(2),) * 6 or balance.mean_drift != _mean(forcing):
        raise RuntimeError("unit C6 lost its arithmetic-mean forced-profile identity")
    return C6CarriedProfile(lattice, balance)


@dataclass(frozen=True, slots=True)
class C6CarriedProfileStep:
    """One replayed, freshly bound carried step around the fixed profile.

    Every mean contribution includes the supplied timestep. The forcing
    defect is in pressure units: carry_feedback=w*L*r and rounding_defect
    is the sum of the observed EPI-product and assembly errors. Neither
    term includes an inferred error of the nonlinear phase evolution.
    Replay and a fresh shared pressure reading do not authenticate a live
    graph invocation, intervening events or repeated phase constancy.
    """

    profile: C6CarriedProfile
    step: NodalRemainderStep
    pressure_observation: C6PressureLatticeObservation
    readout: NodalRemainderPressureReadout
    centered_before: Vector
    centered_after: Vector
    error_before: Vector
    error_after: Vector
    carry_feedback: Vector
    rounding_defect: Vector
    forcing_defect: Vector
    centered_forcing_defect: Vector
    modeled_error_after: Vector
    recurrence_residual: Vector
    mean_change: Fraction
    mean_source_contribution: Fraction
    mean_rounding_contribution: Fraction
    mean_carry_contribution: Fraction
    mean_identity_residual: Fraction

    @property
    def graph_provenance_certified(self) -> bool:
        return False


def observe_c6_carried_profile_step(
    profile: C6CarriedProfile, *, step: NodalRemainderStep,
) -> C6CarriedProfileStep:
    """Bind exact carried dynamics to refreshed pressure and centered shape.

    Let X=x+r, A be the held represented phase contribution, and
    eta=p-A+w*L*x the actual product/assembly rounding defect. Replaying
    the supplied unit-capacity step verifies X'=X+h*p. Consequently
    X'=(I-h*w*L)*X+h*A+h*(w*L*r+eta). With P arithmetic centering,
    y=P*X-z obeys y'=(I-h*w*L)*y+h*P*(w*L*r+eta).
    The independent mean identity is mean(X'-X)=h*(mean(A)+mean(eta));
    the carry-feedback mean is exactly zero. No contraction or mean-prefix
    bound is inferred from this one-step identity.

    Require h>0, six unit capacities, a canonical supplied carried step,
    and its complete declared positive band inside the reference slab.
    Stored pressure must equal a fresh shared CPU reading at the displayed
    before-state. Forged output caches are rejected; profile caches rebuild.
    """
    if type(profile) is not C6CarriedProfile:
        raise TypeError("profile must be a C6CarriedProfile")
    profile = derive_c6_carried_profile(profile.lattice)
    if type(step) is not NodalRemainderStep:
        raise TypeError("step must be a NodalRemainderStep")
    for name in ("exact_increment", "visible_increment", "carry_transfer", "nodal_balance_residual"):
        values = getattr(step, name)
        if (type(values) is not tuple or len(values) != 6
                or any(type(value) is not Fraction for value in values)):
            raise TypeError(f"step.{name} must contain six exact Fraction values")
    before = _validate_nodal_remainder_state(step.before)
    after = _validate_nodal_remainder_state(step.after)
    replay = advance_nodal_remainder(
        step.before, timestep=step.timestep, capacity=step.capacity, pressure=step.pressure,
    )
    if replay != step:
        raise ValueError("the supplied step must match its complete shared-kernel replay")
    if step.timestep <= 0 or step.capacity != (1.,) * 6:
        raise ValueError("the C6 profile requires a positive timestep and six unit capacities")
    source = profile.lattice.source
    if not source.epi_lower <= step.before.epi_lower <= step.before.epi_upper <= source.epi_upper:
        raise ValueError("the complete carried-state band must lie inside the reference slab")
    observation = _observe_rebuilt_c6_pressure_lattice(profile.lattice, step.before.epi)
    if step.pressure != observation.pressure:
        raise ValueError("the supplied pressure must equal the freshly evaluated C6 CPU pressure")
    balance = profile.forced_balance
    neighbors = balance.source.support_neighbors
    conductance = tuple(tuple(Fraction(j in row) for j in range(6)) for row in neighbors)
    readout = observe_nodal_remainder_pressure_readout(
        state=step.before, conductance=conductance, capacity=step.capacity,
        stored_pressure=step.pressure, epi_weight=balance.epi_weight,
    )
    first = observe_forced_support_pattern(balance, nodes=balance.source.nodes, epi=before)
    last = observe_forced_support_pattern(balance, nodes=balance.source.nodes, epi=after)
    centered_before, centered_after = _center(before), _center(after)
    error_before, error_after = first.relative_error, last.relative_error
    carry = readout.readout_shift
    rounding = tuple(a + b for a, b in zip(
        observation.epi_reduction_error, observation.assembly_error, strict=True,
    ))
    forcing = tuple(a + b for a, b in zip(carry, rounding, strict=True))
    centered_forcing = _center(forcing)
    h, weight = Fraction(step.timestep), balance.epi_weight
    modeled = tuple(value - h * weight * gradient + h * defect for value, gradient, defect in zip(
        error_before, laplacian_action(error_before), centered_forcing, strict=True,
    ))
    residual = tuple(actual - expected for actual, expected in zip(error_after, modeled, strict=True))
    change = last.mean - first.mean
    source_mean = h * balance.mean_drift
    rounding_mean, carry_mean = h * _mean(rounding), h * _mean(carry)
    mean_residual = change - source_mean - rounding_mean - carry_mean
    readout_expected = tuple(a + b for a, b in zip(balance.forcing, forcing, strict=True))
    if (any(residual) or mean_residual or carry_mean
            or readout.stored_minus_reconstructed_reference != readout_expected):
        raise RuntimeError("the carried C6 profile lost its exact recurrence or mean decomposition")
    return C6CarriedProfileStep(
        profile, step, observation, readout, centered_before, centered_after,
        error_before, error_after, carry, rounding, forcing, centered_forcing,
        modeled, residual, change, source_mean, rounding_mean, carry_mean, mean_residual,
    )
