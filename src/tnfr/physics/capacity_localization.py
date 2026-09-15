"""Exact capacity/EPI balance on a fixed unit-conductance cycle.

For zero phase and topology pressures, the canonical two-channel pressure
is ``-epi_weight*L*x-vf_weight*L*nu``. Fixed positive capacity makes
``y=x+(vf_weight/epi_weight)*nu`` a heterogeneous diffusion coordinate.
Its equilibrium can retain a prepared capacity contrast; no spontaneous
confinement, live graph admission or changing-capacity balance is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import exact_or_represented_real as _rational
from ..constants import DEFAULTS
from ..utils import normalize_weights
from ._cycle_algebra import (
    Matrix, Vector, dirichlet_energy as _dirichlet, dot as _dot,
    laplacian_action as _laplacian_action,
    laplacian_matrix as _laplacian_matrix, ordered_vector as _vector,
)

__all__ = ["CycleCapacityBalance", "observe_cycle_capacity_balance"]


@dataclass(frozen=True)
class CycleCapacityBalance:
    """Detached exact balance of supplied cycle EPI and held capacity.

    ``metric_weights`` are ``2/nu_i``. The Lyapunov value measures deviation
    from the predicted held-capacity profile, whereas shifted Dirichlet
    energy measures gradients of ``x+(vf_weight/epi_weight)*nu``. Ordinary
    EPI Dirichlet energy need not decrease under the capacity pressure.
    """

    epi: Vector
    capacity: Vector
    epi_weight: Fraction
    vf_weight: Fraction
    shift_ratio: Fraction
    laplacian: Matrix
    metric_weights: Vector
    shifted_epi: Vector
    conserved_epi_total: Fraction
    shifted_consensus: Fraction
    harmonic_capacity: Fraction
    equilibrium_epi: Vector
    equilibrium_pressure: Vector
    pressure: Vector
    epi_rate: Vector
    deviation: Vector
    lyapunov_value: Fraction
    lyapunov_derivative: Fraction
    lyapunov_balance_residual: Fraction
    shifted_dirichlet_energy: Fraction
    shifted_dirichlet_derivative: Fraction
    shifted_dirichlet_balance_residual: Fraction
    epi_dirichlet_energy: Fraction
    epi_dirichlet_derivative: Fraction
    pressure_identity_residual: Vector
    mean_conservation_residual: Fraction


def observe_cycle_capacity_balance(
    epi, capacity, *, epi_weight=None, vf_weight=None
) -> CycleCapacityBalance:
    """Derive a fixed-cycle balance without reading or changing a graph.

    Require at least three nodes, equal vector lengths, positive capacities
    and EPI coefficient, and a nonnegative capacity coefficient. Explicit
    weights are effective channel coefficients used as supplied. Missing
    coefficients use the existing normalized full four-channel defaults;
    the EPI/capacity pair is never renormalized on its own.
    General positive explicit pairs are formal algebraic inputs; canonical
    correspondence additionally requires the actual normalized runtime
    channel coefficients, which this detached observer does not establish.

    Exact rationals are preserved and other real inputs retain the shared
    represented-binary64 rational value. The result assumes identical EPI
    and capacity neighbor walks, as on a unit-conductance simple cycle,
    zero remaining channel pressures, no clipping, and held capacity. It
    does not establish those conditions on a live graph or identify a
    subsequent numerical or canonical operator trajectory.
    """
    values, frequency = _vector(epi, "epi"), _vector(capacity, "capacity")
    count = len(values)
    if count < 3 or len(frequency) != count:
        raise ValueError("a cycle requires matching vectors of length at least three")
    if any(value <= 0 for value in frequency):
        raise ValueError("capacity must be strictly positive at every node")
    if epi_weight is None or vf_weight is None:
        defaults = normalize_weights(
            DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo")
        )
        if epi_weight is None:
            epi_weight = defaults["epi"]
        if vf_weight is None:
            vf_weight = defaults["vf"]
    e, f = _rational(epi_weight, "epi_weight"), _rational(vf_weight, "vf_weight")
    if e <= 0 or f < 0:
        raise ValueError("epi_weight must be positive and vf_weight nonnegative")
    ratio = f / e
    laplacian = _laplacian_matrix(count)
    metric = tuple(2 / value for value in frequency)
    shifted = tuple(x + ratio * nu for x, nu in zip(values, frequency))
    metric_sum = sum(metric, Fraction(0))
    conserved = _dot(metric, values)
    consensus = _dot(metric, shifted) / metric_sum
    harmonic = 2 * count / metric_sum
    equilibrium = tuple(consensus - ratio * nu for nu in frequency)
    lx, ln, ly = (_laplacian_action(vector) for vector in (values, frequency, shifted))
    pressure = tuple(-e * a - f * b for a, b in zip(lx, ln))
    rate = tuple(nu * p for nu, p in zip(frequency, pressure))
    equilibrium_pressure = tuple(
        -e * a - f * b for a, b in zip(_laplacian_action(equilibrium), ln)
    )
    deviation = tuple(x - target for x, target in zip(values, equilibrium))
    lyapunov = _dot(metric, tuple(value**2 for value in deviation)) / 2
    lyapunov_rate = _dot(tuple(h * z for h, z in zip(metric, deviation)), rate)
    lyapunov_residual = lyapunov_rate + 2 * e * _dirichlet(deviation)
    shifted_rate = 2 * _dot(ly, rate)
    shifted_residual = shifted_rate + 2 * e * _dot(
        frequency, tuple(value**2 for value in ly)
    )
    pressure_residual = tuple(p + e * value for p, value in zip(pressure, ly))
    mean_residual = _dot(metric, rate)
    if (
        any(equilibrium_pressure) or any(pressure_residual)
        or lyapunov_residual or shifted_residual or mean_residual
        or lyapunov_rate > 0 or shifted_rate > 0
    ):
        raise RuntimeError("exact cycle capacity balance lost its identities")
    return CycleCapacityBalance(
        epi=values, capacity=frequency, epi_weight=e, vf_weight=f,
        shift_ratio=ratio, laplacian=laplacian, metric_weights=metric,
        shifted_epi=shifted, conserved_epi_total=conserved,
        shifted_consensus=consensus, harmonic_capacity=harmonic,
        equilibrium_epi=equilibrium, equilibrium_pressure=equilibrium_pressure,
        pressure=pressure, epi_rate=rate, deviation=deviation,
        lyapunov_value=lyapunov, lyapunov_derivative=lyapunov_rate,
        lyapunov_balance_residual=lyapunov_residual,
        shifted_dirichlet_energy=_dirichlet(shifted),
        shifted_dirichlet_derivative=shifted_rate,
        shifted_dirichlet_balance_residual=shifted_residual,
        epi_dirichlet_energy=_dirichlet(values),
        epi_dirichlet_derivative=2 * _dot(lx, rate),
        pressure_identity_residual=pressure_residual,
        mean_conservation_residual=mean_residual,
    )
