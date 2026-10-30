"""Detached currents and cycle readings of the complete phase-sensitive law.

The supplied state is observed at one instant. The unchanged field supplies
the state derivative for the current chain rule; no state is advanced and
no source association, invariant domain or closed collective law is inferred.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._mixed_phase_sensitive_interface import _mixed_phase_sensitive_field
from ._sine_coupling_successor import _derive_coupling_successor_geometry
from .relational_observations import _ordered

_REGIONS = tuple(tuple(range(start, start + 9)) for start in (0, 9, 18, 27))
_CYCLES = tuple(
    tuple((start + j, start + (j + 1) % 9) for j in range(9))
    for start in (0, 9, 18, 27)
)


@dataclass(frozen=True)
class _MixedCycleCurrent:
    """Interval observations of supplied forms, continuous lifts and held weights.

    Edge pairs use increasing node order; positive current leaves the first
    endpoint. Each carried C9 instead follows its original increasing cyclic
    slot orientation, including the final wrap edge. The phase means are
    relative to the carried winding template, not circular resultants.
    """

    forms: tuple[I, ...]
    phase_lifts: tuple[I, ...]
    first_new_edge_conductances: tuple[Q, ...]
    second_new_edge_conductances: tuple[Q, ...]
    edge_pairs: tuple[tuple[int, int], ...]
    form_conductances: tuple[Q, ...]
    phase_coefficients: tuple[Q, ...]
    edge_currents: tuple[I, ...]
    edge_current_derivatives: tuple[I, ...]
    edge_phase_currents: tuple[I, ...]
    current_divergence: tuple[I, ...]
    phase_current_divergence: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    region_mean_forms: tuple[I, ...]
    region_mean_phases: tuple[I, ...]
    region_mean_form_rates: tuple[I, ...]
    region_mean_phase_rates: tuple[I, ...]
    cycle_currents: tuple[I, ...]
    cycle_current_derivatives: tuple[I, ...]
    gamma_bounds: I
    region_nodes: tuple[tuple[int, ...], ...] = _REGIONS
    cycle_edges: tuple[tuple[tuple[int, int], ...], ...] = _CYCLES
    full_state_dimension: int = 72
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    internal_divisor: Q = Q(9)
    current_normalization: str = "(wplus*(x_i-x_j)+gamma*sigma*sin(theta_i-theta_j))/9"
    continuity_equation: str = "x_prime=-div(J)"
    phase_continuity_equation: str = "theta_prime=-div(P)"
    cycle_normalization: str = "Gamma_a=sum_oriented_C9(J)/9"
    current_units: str = "form per structural tau"
    phase_current_units: str = "phase per structural tau"
    cycle_derivative_units: str = "form per structural tau squared"
    current_selected_by_supplied_law: bool = True
    current_selected_by_continuity_alone: bool = False
    supplied_state_authenticated_as_source: bool = False
    supplied_state_membership_certified: bool = False
    collective_derivative_closed_on_readings: bool = False
    new_field_installed: bool = False
    trajectory_evaluated: bool = False
    new_measurement_acquired: bool = False
    physical_current_identified: bool = False
    clock: str = "tau=(1023/1024)*t; unchanged complete phase-sensitive rows"


def _observe_mixed_cycle_current(
    *, forms, phase_lifts, first_new_edge_conductances, second_new_edge_conductances
):
    """Read the law-selected edge current and its instantaneous derivative.

    All 72 state coordinates and both complete held-weight vectors are
    normalized before any geometry, trigonometry or field work. States use
    the shared rational-interval domain; held weights use the complete
    field's exact/represented real admission and ordered 63-edge association.

    Internal sine coefficients are one, independently of form conductances.
    A matching edge has form conductance nine and sine coefficient nine in
    the common divisor-nine representation. Coefficients are held, so their
    time derivatives are zero. No phase wrapping or new frame is installed.
    """
    raw_forms = _ordered(forms, "forms", limit=37)
    raw_phases = _ordered(phase_lifts, "phase_lifts", limit=37)
    first = _ordered(
        first_new_edge_conductances, "first_new_edge_conductances", limit=64
    )
    second = _ordered(
        second_new_edge_conductances, "second_new_edge_conductances", limit=64
    )
    if len(raw_forms) != 36 or len(raw_phases) != 36:
        raise ValueError("forms and phase_lifts each require all 36 coordinates")
    if len(first) != 63 or len(second) != 63:
        raise ValueError(
            "each held-weight vector requires all 63 new-edge conductances"
        )
    forms = tuple(I.coerce(value) for value in raw_forms)
    phases = tuple(I.coerce(value) for value in raw_phases)
    weights = tuple(
        tuple(_exact(value, f"{name}[{j}]") for j, value in enumerate(values))
        for name, values in (
            ("first_new_edge_conductances", first),
            ("second_new_edge_conductances", second),
        )
    )
    if any(not Q(3, 4) <= value <= 1 for row in weights for value in row):
        raise ValueError("new-edge conductances must lie in [3/4,1]")

    geometry = _derive_coupling_successor_geometry(completion="common_sense")
    old_pairs = tuple(
        sorted(
            tuple(sorted((start + j, start + (j + 1) % 9)))
            for start in (0, 9)
            for j in range(9)
        )
    )
    turns = tuple(Q((1 + j // 9) * (j % 9 - 4), 9) for j in range(18))
    phase_bins = tuple(int((9 * turn) % 9) for turn in turns)
    expected_pairs = {
        (a, b)
        for a in range(18)
        for b in range(a + 1, 18)
        if min((phase_bins[b] - phase_bins[a]) % 9, (phase_bins[a] - phase_bins[b]) % 9)
        in (1, 2)
        or (a // 9 != b // 9 and phase_bins[a] == phase_bins[b])
    }
    if not (
        geometry.support_certified
        and geometry.nodes == tuple(range(18))
        and geometry.degree == 9
        and geometry.degrees == (9,) * 18
        and geometry.held_capacities == (Q(1),) * 18
        and geometry.internal_edges == tuple((a, b, Q(1)) for a, b in old_pairs)
        and len(geometry.new_edge_pairs) == 63
        and geometry.new_edge_pairs == tuple(sorted(expected_pairs - set(old_pairs)))
        and len(geometry.edge_pairs) == len(set(geometry.edge_pairs)) == 81
        and set(geometry.edge_pairs) == set(old_pairs) | set(geometry.new_edge_pairs)
        and geometry.target_phase_turns == turns
    ):
        raise ValueError(
            "the carried support, unit cycle weights or template is unavailable"
        )

    edges = []
    for offset, row in zip((0, 18), weights, strict=True):
        internal = geometry.internal_edges + tuple(
            (a, b, value)
            for (a, b), value in zip(geometry.new_edge_pairs, row, strict=True)
        )
        edges.extend((a + offset, b + offset, value, Q(1)) for a, b, value in internal)
    edges.extend((j, j + 18, Q(9), Q(9)) for j in range(18))
    edges = tuple(sorted(edges))

    field = _mixed_phase_sensitive_field(
        first_new_edge_conductances=weights[0], second_new_edge_conductances=weights[1]
    )
    rates = field(forms + phases)
    form_rates, phase_rates = rates[:36], rates[36:]
    pi = pi_interval()
    gamma = 1 / (1023 * pi)
    currents, derivatives, phase_currents = [], [], []
    divergence = [I(0) for _ in range(36)]
    phase_divergence = [I(0) for _ in range(36)]
    for a, b, weight, sigma in edges:
        gap = phases[a] - phases[b]
        current = (weight * (forms[a] - forms[b]) + gamma * sigma * sin(gap)) / 9
        derivative = (
            weight * (form_rates[a] - form_rates[b])
            + gamma * sigma * cos(gap) * (phase_rates[a] - phase_rates[b])
        ) / 9
        currents.append(current)
        derivatives.append(derivative)
        phase_current = -gamma * weight * (forms[a] - forms[b]) / 9
        phase_currents.append(phase_current)
        divergence[a] += current
        divergence[b] -= current
        phase_divergence[a] += phase_current
        phase_divergence[b] -= phase_current

    # The weighted form part telescopes on these specifically unit-weight C9s.
    # Retaining this exact cancellation avoids an interval dependence artifact.
    cycle_currents = tuple(
        gamma * sum((sin(phases[a] - phases[b]) for a, b in cycle), I(0)) / 81
        for cycle in _CYCLES
    )
    cycle_derivatives = tuple(
        gamma
        * sum(
            (
                cos(phases[a] - phases[b]) * (phase_rates[a] - phase_rates[b])
                for a, b in cycle
            ),
            I(0),
        )
        / 81
        for cycle in _CYCLES
    )
    template = tuple(2 * pi * turn for turn in geometry.target_phase_turns) * 2

    def means(values):
        return tuple(sum((values[j] for j in region), I(0)) / 9 for region in _REGIONS)

    return _MixedCycleCurrent(
        forms=forms,
        phase_lifts=phases,
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
        edge_pairs=tuple((a, b) for a, b, _, _ in edges),
        form_conductances=tuple(weight for _, _, weight, _ in edges),
        phase_coefficients=tuple(sigma for _, _, _, sigma in edges),
        edge_currents=tuple(currents),
        edge_current_derivatives=tuple(derivatives),
        edge_phase_currents=tuple(phase_currents),
        current_divergence=tuple(divergence),
        phase_current_divergence=tuple(phase_divergence),
        form_rates=tuple(form_rates),
        phase_rates=tuple(phase_rates),
        region_mean_forms=means(forms),
        region_mean_phases=means(
            tuple(a - b for a, b in zip(phases, template, strict=True))
        ),
        region_mean_form_rates=means(form_rates),
        region_mean_phase_rates=means(phase_rates),
        cycle_currents=cycle_currents,
        cycle_current_derivatives=cycle_derivatives,
        gamma_bounds=gamma,
    )
