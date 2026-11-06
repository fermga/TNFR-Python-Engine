"""Local temporal modes of the unchanged complete aligned pair.

The support owner observes the explicit equilibrium only. First-order jets
differentiate the existing complete field; the uniform spectral conclusion
uses a symmetric quadratic pencil, without diagonalizing either held matrix.
Neither operation acquires a source state or advances a trajectory.
"""

from dataclasses import dataclass, replace
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I, cos, pi_interval
from ._mixed_cycle_current import _MixedCycleCurrent, _observe_mixed_cycle_current
from ._mixed_phase_sensitive_interface import _mixed_phase_sensitive_field
from .relational_observations import _ordered

_GL, _G = Q(1, 3300), Q(1, 3000)
_L_BOUNDS, _K_BOUNDS = (Q(3), Q(36)), (Q(4, 9), Q(36))
_TURNS = tuple(Q((1 + j // 9) * (j % 9 - 4), 9) for j in range(18)) * 2
_CLOCK = "tau=(1023/1024)*t; unchanged complete phase-sensitive rows"


def _aligned_member(first_new_edge_conductances, second_new_edge_conductances):
    """Reuse the current owner's exact support and ordered-weight admission.

    Both complete raw vectors are normalized before template arithmetic or
    the static observation. Its support checks retain every original unit
    C9 edge, the common-sense completion and the original winding template.
    This reuse performs a static field observation, not a source reassessment.
    """
    first = _ordered(
        first_new_edge_conductances, "first_new_edge_conductances", limit=64
    )
    second = _ordered(
        second_new_edge_conductances, "second_new_edge_conductances", limit=64
    )
    if len(first) != 63 or len(second) != 63:
        raise ValueError("each held-weight vector requires all 63 new conductances")
    weights = tuple(
        tuple(_exact(value, f"{name}[{j}]") for j, value in enumerate(row))
        for name, row in (
            ("first_new_edge_conductances", first),
            ("second_new_edge_conductances", second),
        )
    )
    if any(not Q(3, 4) <= value <= 1 for row in weights for value in row):
        raise ValueError("new-edge conductances must lie in [3/4,1]")
    pi = pi_interval()
    template = tuple(2 * pi * turn for turn in _TURNS)
    observation = _observe_mixed_cycle_current(
        forms=(0,) * 36,
        phase_lifts=template,
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
    )
    if not isinstance(observation, _MixedCycleCurrent):
        raise TypeError("the static template observation has an invalid type")

    def exact_row(values, name, limit):
        return tuple(
            _exact(value, f"{name}[{j}]")
            for j, value in enumerate(_ordered(values, name, limit=limit))
        )

    pairs = tuple(
        _ordered(pair, "edge pair", limit=3)
        for pair in _ordered(observation.edge_pairs, "edge pairs", limit=181)
    )
    if any(
        len(pair) != 2
        or any(type(node) is not int for node in pair)
        or not 0 <= pair[0] < pair[1] < 36
        for pair in pairs
    ):
        raise ValueError("edge pairs require ordered ordinary node indices")
    observation = replace(
        observation,
        edge_pairs=pairs,
        first_new_edge_conductances=exact_row(
            observation.first_new_edge_conductances, "first observed weights", 64
        ),
        second_new_edge_conductances=exact_row(
            observation.second_new_edge_conductances, "second observed weights", 64
        ),
        physical_capacities=exact_row(
            observation.physical_capacities, "physical capacities", 37
        ),
        form_conductances=exact_row(
            observation.form_conductances, "form conductances", 181
        ),
        phase_coefficients=exact_row(
            observation.phase_coefficients, "phase coefficients", 181
        ),
    )
    if not (
        observation.forms == (I(0),) * 36
        and observation.phase_lifts == template
        and observation.first_new_edge_conductances == weights[0]
        and observation.second_new_edge_conductances == weights[1]
        and observation.full_state_dimension == 72
        and observation.physical_capacities == (Q(1),) * 36
        and observation.internal_divisor == 9
        and observation.clock == _CLOCK
        and observation.gamma_bounds == 1 / (1023 * pi)
        and len(observation.edge_pairs) == len(set(observation.edge_pairs)) == 180
        and len(observation.form_conductances) == 180
        and len(observation.phase_coefficients) == 180
        and observation.edge_pairs == tuple(sorted(observation.edge_pairs))
    ):
        raise ValueError("the complete aligned member is unavailable")

    # Bind the returned primitive support to the phase-bin graph whose bare
    # gap is four. Counts and a cached spectral verdict would not suffice.
    bins = tuple(int((9 * turn) % 9) for turn in _TURNS[:18])
    internal_pairs = tuple(
        (a, b)
        for a in range(18)
        for b in range(a + 1, 18)
        if min((bins[b] - bins[a]) % 9, (bins[a] - bins[b]) % 9) in (1, 2)
        or (a // 9 != b // 9 and bins[a] == bins[b])
    )
    expected_pairs = tuple(
        sorted(
            tuple(
                (a + offset, b + offset)
                for offset in (0, 18)
                for a, b in internal_pairs
            )
            + tuple((j, j + 18) for j in range(18))
        )
    )
    if observation.edge_pairs != expected_pairs:
        raise ValueError("the phase-bin support for the uniform gap differs")

    # Re-admit the consumed conductance/coefficient association. Exact support
    # reconstruction itself remains with the current owner's shared admission.
    old = {
        tuple(sorted((start + j, start + (j + 1) % 9)))
        for start in (0, 9, 18, 27)
        for j in range(9)
    }
    new = tuple(
        tuple(
            pair
            for pair in observation.edge_pairs
            if pair[0] // 18 == pair[1] // 18 == copy and pair not in old
        )
        for copy in range(2)
    )
    if tuple(map(len, new)) != (63, 63):
        raise ValueError("the ordered internal conductance association is unavailable")
    expected_weights = {pair: Q(1) for pair in old}
    for pairs, row in zip(new, weights, strict=True):
        expected_weights.update(zip(pairs, row, strict=True))
    expected_weights.update({(j, j + 18): Q(9) for j in range(18)})
    if not (
        set(observation.edge_pairs) == set(expected_weights)
        and observation.form_conductances
        == tuple(expected_weights[pair] for pair in observation.edge_pairs)
        and observation.phase_coefficients
        == tuple(Q(9 if a // 18 != b // 18 else 1) for a, b in observation.edge_pairs)
    ):
        raise ValueError("the complete form and bare-phase coefficients differ")

    # Odd sine terms cancel in exact opposite rational phase gaps. Interval
    # residuals containing zero would not, by themselves, prove equilibrium.
    phase_rows = [{} for _ in range(36)]
    for (a, b), sigma in zip(
        observation.edge_pairs, observation.phase_coefficients, strict=True
    ):
        for left, right in ((a, b), (b, a)):
            gap = (_TURNS[right] - _TURNS[left] + Q(1, 2)) % 1 - Q(1, 2)
            phase_rows[left][gap] = phase_rows[left].get(gap, Q(0)) + sigma
    if any(
        value != row.get(-gap, Q(0))
        for row in phase_rows
        for gap, value in row.items()
        if gap
    ):
        raise ValueError("the exact aligned sine balance is unavailable")
    return observation


def _mixed_aligned_tangent(
    *, first_new_edge_conductances, second_new_edge_conductances
):
    """Compile the genuine field's tangent at the explicit aligned equilibrium.

    The returned callable admits exactly 72 interval/exact direction entries,
    forms followed by continuous phase lifts. It returns their first-order
    image in structural tau. These are formal directional jets, not solution
    jets or a time step, and the full common coordinates remain present.
    """
    member = _aligned_member(first_new_edge_conductances, second_new_edge_conductances)
    field = _mixed_phase_sensitive_field(
        first_new_edge_conductances=member.first_new_edge_conductances,
        second_new_edge_conductances=member.second_new_edge_conductances,
    )
    equilibrium = member.forms + member.phase_lifts

    def tangent(values):
        raw = _ordered(values, "direction", limit=73)
        if len(raw) != 72:
            raise ValueError("direction requires all 72 form and phase entries")
        direction = tuple(I.coerce(value) for value in raw)
        variables = tuple(
            Jet((base, value))
            for base, value in zip(equilibrium, direction, strict=True)
        )
        rows = tuple(field(variables))
        if len(rows) != 72 or any(
            not isinstance(row, Jet) or row.order != 1 for row in rows
        ):
            raise ValueError("the complete field must retain 72 first-order jets")
        return tuple(row.coeffs[1] for row in rows)

    return tangent


@dataclass(frozen=True)
class _MixedLocalModes:
    """Uniform local certificate for an admitted held-weight family member.

    Pole intervals are open and expressed per structural tau. Counts include
    algebraic multiplicity. Repeated poles are allowed; semisimplicity does
    not assert Euclidean orthogonality or simultaneous diagonalization of L,K.
    """

    template_observation: _MixedCycleCurrent
    gamma_bounds: I
    minimum_phase_cosine_bounds: I
    form_laplacian_bounds: tuple[Q, Q]
    phase_hessian_bounds: tuple[Q, Q]
    discriminant_margin: Q
    separator: Q
    separator_pencil_upper_bound: Q
    slow_pole_interval: tuple[Q, Q] | None
    fast_pole_interval: tuple[Q, Q] | None
    slow_mode_count: int | None
    fast_mode_count: int | None
    common_mode_count: int | None
    real_negative_poles_certified: bool
    semisimple_poles_certified: bool
    common_modes_semisimple_certified: bool
    linear_cycle_readout_zero_certified: bool
    uniform_held_weight_family_certified: bool
    status: str
    full_state_dimension: int = 72
    spatial_quotient_dimension: int = 35
    quotient_dimension: int = 70
    common_coordinates: tuple[str, str] = ("mean_form", "mean_phase_lift")
    pole_intervals_are_open: bool = True
    physical_capacities: tuple[Q, ...] = (Q(1),) * 36
    internal_divisor: Q = Q(9)
    original_time_rate_factor: Q = Q(1023, 1024)
    commutation_assumed: bool = False
    euclidean_orthogonal_modes_claimed: bool = False
    supplied_weights_authenticated_as_source: bool = False
    acquired_source_at_equilibrium_claimed: bool = False
    spectrum_numerically_evaluated: bool = False
    trajectory_evaluated: bool = False
    new_field_installed: bool = False
    nonlinear_nonoscillation_claimed: bool = False
    coordinate_monotonicity_claimed: bool = False
    physical_wave_identified: bool = False
    clock: str = _CLOCK


def _assess_mixed_local_modes(
    *, first_new_edge_conductances, second_new_edge_conductances
):
    """Certify real semisimple local poles without an eigenspectrum evaluation.

    On the 35-dimensional global-zero leaf, the exact tangent has the symmetric
    pencil P(z)=z**2 L**(-1)+(z/9)I+gamma**2 K/81. The fixed support gives
    3I<=L<=36I and (4/9)I<=K<=36I for every legal held weight. The latter
    uses cos(4*pi/9)>1/9 and the previously proved bare-copy gap four.

    The discriminant is uniformly positive for every candidate eigenvector;
    P(-1/6)<0 separates 35 slow and 35 fast roots. The branch sign of P' on
    each kernel excludes Jordan chains. The two conserved common directions
    split off as semisimple zero modes. This proof concerns the aligned local
    tangent; no interval residual, numerical diagonalization or old source
    verdict substitutes for it.
    """
    member = _aligned_member(first_new_edge_conductances, second_new_edge_conductances)
    cosine = cos(4 * pi_interval() / 9)
    delta = 1 - 4 * _G**2 * _K_BOUNDS[1] / _L_BOUNDS[0]
    separator_upper = -delta / 108
    certified = (
        _GL < member.gamma_bounds.lo <= member.gamma_bounds.hi < _G
        and cosine.lo > Q(1, 9)
        and 0 < _L_BOUNDS[0] <= _L_BOUNDS[1]
        and 0 < _K_BOUNDS[0] <= _K_BOUNDS[1]
        and delta > 0
        and separator_upper < 0
    )
    return _MixedLocalModes(
        template_observation=member,
        gamma_bounds=member.gamma_bounds,
        minimum_phase_cosine_bounds=cosine,
        form_laplacian_bounds=_L_BOUNDS,
        phase_hessian_bounds=_K_BOUNDS,
        discriminant_margin=delta,
        separator=Q(-1, 6),
        separator_pencil_upper_bound=separator_upper,
        slow_pole_interval=(Q(-1, 6), Q(0)) if certified else None,
        fast_pole_interval=(Q(-4), Q(-1, 6)) if certified else None,
        slow_mode_count=35 if certified else None,
        fast_mode_count=35 if certified else None,
        common_mode_count=2 if certified else None,
        real_negative_poles_certified=certified,
        semisimple_poles_certified=certified,
        common_modes_semisimple_certified=certified,
        linear_cycle_readout_zero_certified=certified,
        uniform_held_weight_family_certified=certified,
        status=("uniform_real_semisimple_local_modes" if certified else "unavailable"),
    )
