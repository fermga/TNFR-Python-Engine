"""Exact conservative bridge memory after an admitted two-cycle observation.

The eight shell coordinates retain every linear bridge response on the stated
two-C6 target. They do not replace nonlinear nodal state or declare a sampled
initial perturbation. Rational coordinate elimination retains the complete
hidden source and kernel without evaluating a trajectory or exponential.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from ..mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_matrix_product,
)
from ..mathematics.linear_observation import (
    LinearCoordinateMemory,
    LinearObservation,
    derive_coordinate_memory,
    derive_linear_observation,
)
from .relational_sine_comparison import _validate_comparison_labels
from .relational_sine_resonance import (
    SineBridgeChannelAssessment,
    _admit_two_cycle_bridge,
    _bridge_full_tangent,
)

__all__ = ("SineBridgeMemoryAssessment", "assess_sine_bridge_memory")

Matrix = tuple[tuple[Q, ...], ...]


@dataclass(frozen=True)
class SineBridgeMemoryAssessment:
    """Exact bridge memory with an explicit unobserved initial-state source.

    Coordinates are all four form shell increments followed by their phase
    counterparts. Visible indices (0, 4) retain the two bridge contrasts.
    For any declared nodal tangent initial state z0, projection_rows*z0 gives
    the eight shell coordinates; coordinate_memory's visible/hidden indices
    then select y0 and h0. The captured comparison's state is not substituted
    for z0 or assumed to be the critical target.
    """

    channels: SineBridgeChannelAssessment
    left_cycle: tuple[Any, ...]
    right_cycle: tuple[Any, ...]
    projection_rows: Matrix
    nodal_lift: Matrix
    full_tangent_generator: Matrix
    coordinate_generator: Matrix
    energy_metric: Matrix
    coordinate_memory: LinearCoordinateMemory
    visible_observation: LinearObservation
    hidden_initial_projection_rows: Matrix
    initial_source_rows: Matrix
    kernel_derivative_at_zero: Matrix
    kernel_second_derivative_at_zero: Matrix
    static_visible_generator: Matrix
    low_frequency_rate_matrix: Matrix
    scope: tuple[str, ...] = (
        "exact_two_C6_unit_support_with_one_aligned_bridge_and_unit_capacities",
        "declared_internal_target_turns_abs_one_sixth_beta_one_zero_loss",
        "tau_equals_t_over_pi_for_the_admitted_unit_exchange_coefficient",
        "eight_shell_coordinates_preserve_every_full_linear_bridge_response",
        "discarded_symmetry_sectors_are_unobservable_at_this_tangent_port",
        "nodal_lift_is_an_invariant_section_not_a_reconstruction_of_all_states",
        "common_origins_are_not_reconstructed_from_the_bridge_response",
        "hidden_initial_state_is_not_set_to_zero_or_inferred_from_captured_source",
        "exact_kernel_B_exp_D_tau_C_and_source_B_exp_D_tau_h0_are_retained",
        "hidden_zero_shell_preparation_is_the_admitted_energy_transpose_lift",
        "static_generator_is_zero_frequency_resolvent_algebra_not_a_memory_integral",
        "low_frequency_rate_matrix_is_not_an_installed_local_loss_law",
        "no_trajectory_fit_exponential_evaluation_nonlinear_or_physical_claim",
    )

    def to_dict(self):
        """Project exact matrices and premises without authenticating observations."""
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.channels.source)
        return {
            "schema": "tnfr.relational-sine-bridge-memory.v1",
            "report": _project(self),
        }


def _two_channels(matrix):
    """Place one rectangular rational map on both channel diagonals."""
    columns = len(matrix[0])
    zero = (Q(0),) * columns
    return tuple(row + zero for row in matrix) + tuple(zero + row for row in matrix)


def _transpose(matrix):
    return tuple(zip(*matrix))


def _subtract(left, right):
    return tuple(
        tuple(a - b for a, b in zip(row, other)) for row, other in zip(left, right)
    )


def assess_sine_bridge_memory(
    source, *, left_cycle, right_cycle, target_phase_turns
) -> SineBridgeMemoryAssessment:
    """Derive exact two-C6 bridge memory from full nodal tangent coordinates.

    The ordered disjoint six-node cycles start at the bridge endpoints and
    cover the full support. Every internal target gap has absolute turn 1/6;
    both winding signs are admitted. Capacity, beta and normalized exchange
    equal one, with zero form loss. These are this reader's model hypotheses,
    not necessary conditions for memory on a general graph.

    Let d_k be right minus left averages at cycle distance k=0..3. Retain
    (d0,d1-d0,d2-d1,d3-d2) in each channel. Exact intertwining authenticates
    the supplied support's tangent reduction. Memory and its hidden source
    then use the existing coordinate owner, with no observed response fitted.
    """
    channels, left, right, cycles = _admit_two_cycle_bridge(
        source,
        left_cycle=left_cycle,
        right_cycle=right_cycle,
        target_phase_turns=target_phase_turns,
    )
    admitted = channels.source
    size = len(admitted.nodes)
    _, _, full, full_metric = _bridge_full_tangent(channels, Q(1, 2))
    shell_positions = ((0,), (1, 5), (2, 4), (3,))
    differences = []
    lift = [[Q(0) for _ in range(4)] for _ in range(size)]
    for distance, shell in enumerate(shell_positions):
        row = [Q(0)] * size
        for sign, cycle in zip((-1, 1), cycles):
            for position in shell:
                row[cycle[position]] = Q(sign, len(shell))
                for column in range(distance + 1):
                    lift[cycle[position]][column] = Q(sign, 2)
        differences.append(tuple(row))
    increments = (differences[0],) + tuple(
        tuple(value - previous for value, previous in zip(row, prior))
        for row, prior in zip(differences[1:], differences[:-1])
    )
    projection = _two_channels(increments)
    lift = _two_channels(tuple(tuple(row) for row in lift))
    product = exact_matrix_product
    generator = product(product(projection, full), lift)
    identity = tuple(tuple(Q(int(i == j)) for j in range(8)) for i in range(8))
    if (
        product(projection, lift) != identity
        or product(projection, full) != product(generator, projection)
        or product(full, lift) != product(lift, generator)
    ):
        raise ArithmeticError("exact bridge shell intertwining failed")
    metric = product(product(_transpose(lift), full_metric), lift)
    if product(metric, projection) != product(_transpose(lift), full_metric):
        raise ArithmeticError("bridge shell projection is not energy-orthogonal")
    weighted = product(metric, generator)
    if any(weighted[i][j] != -weighted[j][i] for i in range(8) for j in range(8)):
        raise ArithmeticError("bridge coordinate storage is not conserved")
    memory = derive_coordinate_memory(generator, (0, 4))
    output = tuple(tuple(Q(int(i == j)) for j in range(8)) for i in (0, 4))
    observation = derive_linear_observation(generator, output)
    hidden_rows = tuple(projection[i] for i in memory.hidden_indices)
    a, b = memory.visible_generator, memory.hidden_to_visible
    c, d = memory.visible_to_hidden, memory.hidden_generator
    inverse = exact_matrix_inverse(d)
    static = _subtract(a, product(product(b, inverse), c))
    rate_correction = product(product(product(b, inverse), inverse), c)
    rate_matrix = tuple(
        tuple(Q(int(i == j)) + value for j, value in enumerate(row))
        for i, row in enumerate(rate_correction)
    )
    return SineBridgeMemoryAssessment(
        channels=channels,
        left_cycle=left,
        right_cycle=right,
        projection_rows=projection,
        nodal_lift=lift,
        full_tangent_generator=full,
        coordinate_generator=generator,
        energy_metric=metric,
        coordinate_memory=memory,
        visible_observation=observation,
        hidden_initial_projection_rows=hidden_rows,
        initial_source_rows=product(b, hidden_rows),
        kernel_derivative_at_zero=product(product(b, d), c),
        kernel_second_derivative_at_zero=product(product(product(b, d), d), c),
        static_visible_generator=static,
        low_frequency_rate_matrix=rate_matrix,
    )
