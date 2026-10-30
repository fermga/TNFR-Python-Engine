"""Exact reset and Euler budgets for canonical support on a fixed cycle.

The supplied coordinate ``phase_offset_over_pi`` is the periodic phase
perturbation divided by pi. No phase chart or winding is inferred from it.
The pressure identity and canonical UM/SHA interpretation require the
strict target-only cycle regime stated in ``CYCLE_SUPPORT_DYNAMICS.md``.
"""

from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import exact_or_represented_real as _rational
from ..constants import DEFAULTS
from ..constants.canonical import COUPLING_GENTLE, SHA_VF_FACTOR, UM_THETA_PUSH
from ..utils import normalize_weights
from ._cycle_algebra import (
    Vector,
    dirichlet_energy,
    dot,
    laplacian_action,
    ordered_vector,
)

__all__ = [
    "CycleSupportBalance",
    "CycleSupportReset",
    "CycleSupportEuler",
    "observe_cycle_support_balance",
    "observe_cycle_support_reset",
    "observe_cycle_support_euler",
]


@dataclass(frozen=True)
class CycleSupportBalance:
    """Detached exact fixed-support state, pressure and Dirichlet balance."""

    epi: Vector
    capacity: Vector
    phase_offset_over_pi: Vector
    epi_weight: Fraction
    vf_weight: Fraction
    phase_weight: Fraction
    shifted_epi: Vector
    pressure: Vector
    epi_rate: Vector
    dirichlet_energy: Fraction
    dirichlet_derivative: Fraction
    pressure_identity_residual: Vector
    dirichlet_balance_residual: Fraction


def observe_cycle_support_balance(
    epi,
    capacity,
    phase_offset_over_pi,
    *,
    epi_weight=None,
    vf_weight=None,
    phase_weight=None,
) -> CycleSupportBalance:
    """Observe the exact three-channel cycle reference without graph writes.

    Missing weights use the normalized full four-channel defaults; explicit
    coefficients are used as supplied. General positive coefficient models
    need separate normalized-runtime identification. Capacity may vanish:
    the fixed Dirichlet balance needs no division by capacity. Neither a
    compatible phase chart nor positive-capacity convergence is inferred.
    """
    x = ordered_vector(epi, "epi")
    nu = ordered_vector(capacity, "capacity")
    a = ordered_vector(phase_offset_over_pi, "phase_offset_over_pi")
    if len(x) < 3 or len(nu) != len(x) or len(a) != len(x):
        raise ValueError("cycle support requires matching vectors of length >= 3")
    if any(value < 0 for value in nu):
        raise ValueError("capacity must be nonnegative")
    supplied = (epi_weight, vf_weight, phase_weight)
    if any(value is None for value in supplied):
        defaults = normalize_weights(
            DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo")
        )
        supplied = tuple(
            defaults[name] if value is None else value
            for name, value in zip(("epi", "vf", "phase"), supplied)
        )
    e, f, w = tuple(
        _rational(value, name)
        for value, name in zip(supplied, ("epi_weight", "vf_weight", "phase_weight"))
    )
    if e <= 0 or f < 0 or w < 0:
        raise ValueError("EPI weight must be positive; other weights nonnegative")
    shifted = tuple(xi + (f * vi + w * ai) / e for xi, vi, ai in zip(x, nu, a))
    lx, ln, la, ly = (laplacian_action(values) for values in (x, nu, a, shifted))
    pressure = tuple(-e * xi - f * vi - w * ai for xi, vi, ai in zip(lx, ln, la))
    rate = tuple(vi * pi for vi, pi in zip(nu, pressure))
    derivative = 2 * dot(ly, rate)
    pressure_residual = tuple(pi + e * yi for pi, yi in zip(pressure, ly))
    energy_residual = derivative + 2 * e * dot(nu, tuple(yi**2 for yi in ly))
    if any(pressure_residual) or energy_residual or derivative > 0:
        raise RuntimeError("exact support flow balance lost its identities")
    return CycleSupportBalance(
        x,
        nu,
        a,
        e,
        f,
        w,
        shifted,
        pressure,
        rate,
        dirichlet_energy(shifted),
        derivative,
        pressure_residual,
        energy_residual,
    )


def _rebalance(value: CycleSupportBalance) -> CycleSupportBalance:
    if type(value) is not CycleSupportBalance:
        raise TypeError("before must be a CycleSupportBalance")
    # Public dataclass fields are not provenance: rebuild all cached quantities.
    return observe_cycle_support_balance(
        value.epi,
        value.capacity,
        value.phase_offset_over_pi,
        epi_weight=value.epi_weight,
        vf_weight=value.vf_weight,
        phase_weight=value.phase_weight,
    )


def _with_state(before, epi, capacity, phase):
    return observe_cycle_support_balance(
        epi,
        capacity,
        phase,
        epi_weight=before.epi_weight,
        vf_weight=before.vf_weight,
        phase_weight=before.phase_weight,
    )


def _unit(value, label: str) -> Fraction:
    result = _rational(value, label)
    if not 0 <= result <= 1:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class CycleSupportReset:
    """Exact target-only all-node UM followed by uniform SHA reference."""

    before: CycleSupportBalance
    after: CycleSupportBalance
    eta: Fraction
    vf_sync: Fraction
    silence_factor: Fraction
    support_jump: Vector
    cross_term: Fraction
    quadratic_term: Fraction
    energy_change: Fraction
    identity_residual: Fraction


def observe_cycle_support_reset(
    before,
    *,
    eta=UM_THETA_PUSH,
    vf_sync=COUPLING_GENTLE,
    silence_factor=SHA_VF_FACTOR,
) -> CycleSupportReset:
    """Observe the exact support reset and its signed energy change.

    Identity factors are formal controls, not a claim of operator admission:
    in particular q=1 is not an admitted SHA attenuation. q=0 is an exact
    freezing boundary. EPI stays fixed, while the new support is assumed
    available to a subsequent canonical pressure refresh. Runtime rounding,
    clipping and intermediate pressure writes are outside this reference.
    """
    before = _rebalance(before)
    eta_q = _unit(eta, "eta")
    sync = _unit(vf_sync, "vf_sync")
    retention = _unit(silence_factor, "silence_factor")
    ln = laplacian_action(before.capacity)
    la = laplacian_action(before.phase_offset_over_pi)
    nu = tuple(retention * (v - sync * lv) for v, lv in zip(before.capacity, ln))
    phase = tuple(v - eta_q * lv for v, lv in zip(before.phase_offset_over_pi, la))
    after = _with_state(before, before.epi, nu, phase)
    jump = tuple(b - a for a, b in zip(before.shifted_epi, after.shifted_epi))
    cross = 2 * dot(laplacian_action(before.shifted_epi), jump)
    quadratic = dirichlet_energy(jump)
    change = after.dirichlet_energy - before.dirichlet_energy
    residual = change - cross - quadratic
    if residual:
        raise RuntimeError("exact support reset lost its energy identity")
    return CycleSupportReset(
        before, after, eta_q, sync, retention, jump, cross, quadratic, change, residual
    )


@dataclass(frozen=True)
class CycleSupportEuler:
    """One detached refreshed-Euler reference with exact energy remainder."""

    before: CycleSupportBalance
    after: CycleSupportBalance
    dt: Fraction
    linear_energy_change: Fraction
    quadratic_energy_change: Fraction
    energy_change: Fraction
    identity_residual: Fraction
    convex_step_condition: bool


def observe_cycle_support_euler(before, *, dt) -> CycleSupportEuler:
    """Observe one exact affine nodal Euler step with held structural support.

    The nonnegative quadratic energy term is the exact finite-step remainder,
    not a continuous dissipation term. dt*e*max(nu)<=1 is a sufficient convex
    cycle step and Dirichlet nonincrease condition. Larger steps remain
    observable but are not silently promoted to stable solver execution.
    """
    before = _rebalance(before)
    step = _rational(dt, "dt")
    if step <= 0:
        raise ValueError("dt must be positive")
    epi = tuple(x + step * rate for x, rate in zip(before.epi, before.epi_rate))
    after = _with_state(before, epi, before.capacity, before.phase_offset_over_pi)
    linear = step * before.dirichlet_derivative
    quadratic = step**2 * dirichlet_energy(before.epi_rate)
    change = after.dirichlet_energy - before.dirichlet_energy
    residual = change - linear - quadratic
    convex = step * before.epi_weight * max(before.capacity) <= 1
    if residual or (convex and change > 0):
        raise RuntimeError("exact support Euler step lost its energy identity")
    return CycleSupportEuler(
        before, after, step, linear, quadratic, change, residual, convex
    )
