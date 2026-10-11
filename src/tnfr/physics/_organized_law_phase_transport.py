"""Analytical phase transport from the fixed historical form observations.

No readings, trajectories or coefficient designs are evaluated here. The
bound applies to the exact heat-plus-constant-force bridge through any one
compatible record. It neither reconstructs an unobserved tail nor certifies
cross-law separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from ._organized_computable_prediction import (
    _READING_ERROR,
    _ForecastStage,
    _forecast_stages,
)
from ._organized_law_observation import (
    _admit_carrier_stage,
    _bound_organized_observation_law,
)


def _historical_cells() -> tuple[_ForecastStage, ...]:
    """Rebuild the fixed sampling domain, stopping at its last reading."""
    stages = tuple(_admit_carrier_stage(stage) for stage in _forecast_stages()[1:])
    if any(stage is None for stage in stages):
        raise ArithmeticError("historical stages must be explicit")
    normalized = tuple(stage for stage in stages if stage is not None)
    plateaus = tuple(stage for stage in normalized if stage.name == "plateau")
    if not plateaus:
        raise ArithmeticError("historical transport requires declared readings")
    times = tuple(
        time
        for stage in plateaus
        for time in (stage.start, stage.start + stage.duration)
    )
    cells = tuple(stage for stage in normalized if stage.start < times[-1])
    if len(cells) != len(times) - 1:
        raise ArithmeticError("each historical cell must have two observed endpoints")
    for cell, start, end in zip(cells, times, times[1:]):
        if cell.start != start or cell.start + cell.duration != end:
            raise ArithmeticError(
                "historical cells must retain the sampling boundaries"
            )
        if cell.duration > 1:
            raise ArithmeticError(
                "the spectral Green bound requires cells of length <=1"
            )
    if any(stage.child is not None and stage.increment != 0 for stage in plateaus):
        raise ArithmeticError("readings require zero command-form offset on plateaus")
    return cells


@dataclass(frozen=True)
class _OrganizedLawPhaseTransport:
    deformation: Q
    historical_start: Q
    historical_end: Q
    observation_times: tuple[Q, ...]
    cell_durations: tuple[Q, ...]
    command_variation: Q
    command_weighted_variation: Q
    cell_cube_sum: Q
    source_form_error_upper: Q
    sample_error_upper: Q
    carrier_form_upper: Q
    endpoint_noise_phase_error: Q
    command_phase_error: Q
    endogenous_phase_error: Q
    historical_phase_increment_error: Q
    cell_end_error_bounds: tuple[Q, ...]
    absolute_initial_phase_drift_upper: Q
    green_kernel_factor: Q = Q(8, 5)
    edge_phase_row_squared_norm: Q = Q(5, 4)
    reading_error: Q = _READING_ERROR
    scope: str = "conditional_historical_phase_transport_not_a_law_discriminator"


def _bound_organized_law_phase_transport(
    *, deformation: Any
) -> _OrganizedLawPhaseTransport:
    """Bound historical carrier edge-phase increments around a common record bridge.

    Set A=L5/4, Q_h=int_0^h exp(-sA) ds and V(t)=Q_t Q_h^-1.
    The bridge is (I-V)y_left+V y_right, so it exactly retains arbitrary
    heat relaxation and constant forcing. With exact endpoints its error
    solves e''+A e'=w', e(0)=e(h)=0. On 0<=mu<=1 and h<=1 the scalar
    Dirichlet Green magnitude is at most (8/5) times the zero-drift Green.
    Its time integral is s(h-s)/2. The C5 bounds ||B||_2<=2 and
    ||row(B' A)||_2=sqrt(5)/2 give the three errors below, using
    ||w'||_2 <= gamma*L*(sqrt(2)*|q'|+2*gamma*sqrt(5)*M)/2.

    The cubic command contributes sqrt(10)*gamma^2*L*|a|*h^2/25;
    sqrt(10)<19/6 makes this rational. Each endpoint perturbation has
    Euclidean norm <=sqrt(5)*sample_error. Both bridge weight norms are
    <=1, giving the conservative 5*gamma*h*sample_error phase allowance.
    Taking a full-cell absolute integral also bounds any partial cell.
    """
    law = _bound_organized_observation_law(deformation=deformation)
    if not law.admitted:
        raise ArithmeticError("phase transport requires own-law source admission")
    cells = _historical_cells()
    start, end = cells[0].start, cells[-1].start + cells[-1].duration
    times = (Q(0),) + tuple(cell.start + cell.duration - start for cell in cells)
    durations = tuple(cell.duration for cell in cells)
    amplitudes = tuple(
        abs(cell.increment) if cell.child is not None else Q(0) for cell in cells
    )
    variation = sum(amplitudes, Q(0))
    weighted = sum((a * h**2 for a, h in zip(amplitudes, durations)), Q(0))
    cubes = sum((h**3 for h in durations), Q(0))
    g, lipschitz = law.gamma_upper, law.current_lipschitz
    sample_error = _READING_ERROR + law.source_form_error_upper
    form = law.initial_form_upper + g * law.current_upper * law.duration / 2
    noise_factor = 5 * g * sample_error
    command_factor = Q(19, 150) * g**2 * lipschitz
    endogenous_factor = g**3 * lipschitz * form / 3
    noise = noise_factor * (end - start)
    command = command_factor * weighted
    endogenous = endogenous_factor * cubes
    prefixes = []
    cumulative = Q(0)
    for amplitude, duration in zip(amplitudes, durations):
        cumulative += (
            noise_factor * duration
            + command_factor * amplitude * duration**2
            + endogenous_factor * duration**3
        )
        prefixes.append(cumulative)
    return _OrganizedLawPhaseTransport(
        law.deformation,
        start,
        end,
        times,
        durations,
        variation,
        weighted,
        cubes,
        law.source_form_error_upper,
        sample_error,
        form,
        noise,
        command,
        endogenous,
        noise + command + endogenous,
        tuple(prefixes),
        4 * g * law.initial_form_upper + 2 * g**2 * law.current_upper * law.duration,
    )
