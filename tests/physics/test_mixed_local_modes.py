"""Independent static controls of the aligned complete-law temporal tangent.

The held weights below are constructed legal family members, not a replay of
the acquired pair. No temporal spectrum, trajectory or new source is evaluated.
"""

import os
from dataclasses import FrozenInstanceError, asdict, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._interval_taylor import sin as jet_sin
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_local_modes as owner
from tnfr.physics import _sine_flow as flow_owner
from tnfr.research.artifact_io import encode_exact_tree, write_json_once

G, GL = Q(1, 3000), Q(1, 3300)
ALPHA, TOL = Q(1023, 1024), Q(1, 10**190)
TURNS = tuple(Q((1 + j // 9) * (j % 9 - 4), 9) for j in range(18)) * 2
CYCLES = tuple(
    tuple((k + j, k + (j + 1) % 9) for j in range(9)) for k in (0, 9, 18, 27)
)


def _mp(mp, value):
    if isinstance(value, (int, Q, np.integer)):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second, strict=True))


def _inner(mp, first, second):
    return sum(mp.conj(a) * b for a, b in zip(first, second, strict=True))


def _center(values):
    values = tuple(values)
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def _support():
    bins = tuple(int((9 * turn) % 9) for turn in TURNS[:18])
    old = {
        tuple(sorted((start + j, start + (j + 1) % 9)))
        for start in (0, 9)
        for j in range(9)
    }
    pairs = {
        (a, b)
        for a in range(18)
        for b in range(a + 1, 18)
        if min((bins[b] - bins[a]) % 9, (bins[a] - bins[b]) % 9) in (1, 2)
        or (a // 9 != b // 9 and bins[a] == bins[b])
    }
    return tuple(sorted(old)), tuple(sorted(pairs - old))


def _weights():
    # Fixed exact unequal weights throughout the admitted family; no source fit.
    return tuple(
        tuple(Q(3, 4) + Q((7 * j + side) % 17, 64) for j in range(63))
        for side in (0, 1)
    )


def _edges(weights):
    old, new = _support()
    result = []
    for side in (0, 1):
        weighted = tuple((a, b, Q(1)) for a, b in old) + tuple(
            (a, b, weight) for (a, b), weight in zip(new, weights[side], strict=True)
        )
        result.extend((a + 18 * side, b + 18 * side, w, Q(1)) for a, b, w in weighted)
    result.extend((j, j + 18, Q(9), Q(9)) for j in range(18))
    return tuple(sorted(result))


def _laplacian(mp, values, weights, *, phase=False):
    result = [mp.mpf(0)] * 36
    for a, b, weight, sigma in _edges(weights):
        coefficient = (
            _mp(mp, sigma) * mp.cos(2 * mp.pi * _mp(mp, TURNS[a] - TURNS[b]))
            if phase
            else _mp(mp, weight)
        )
        current = coefficient * (values[a] - values[b])
        result[a] += current
        result[b] -= current
    return tuple(result)


def _oracle(mp, direction, weights):
    x = tuple(_mp(mp, value) for value in direction[:36])
    y = tuple(_mp(mp, value) for value in direction[36:])
    lx = _laplacian(mp, x, weights)
    ky = _laplacian(mp, y, weights, phase=True)
    gamma = 1 / (1023 * mp.pi)
    return tuple((-a - gamma * b) / 9 for a, b in zip(lx, ky, strict=True)) + tuple(
        gamma * a / 9 for a in lx
    )


def _arguments(weights):
    return dict(
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
    )


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_source_or_spectrum():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("local-mode controls cannot acquire, integrate or diagonalize")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (np.linalg, "eig"),
            (np.linalg, "eigh"),
            (np.linalg, "eigvals"),
            (np.linalg, "eigvalsh"),
        ):
            patch.setattr(module, name, forbidden)
        # Allow only the existing row factory for static directional jets.
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 220

    def forbidden(*args, **kwargs):
        pytest.fail("local-mode controls do not evaluate temporal roots or flows")

    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "exp",
            "expm",
            "eig",
            "eigh",
            "polyroots",
            "odefun",
            "invertlaplace",
        ):
            patch.setattr(context, name, forbidden)
        yield context


@pytest.fixture(scope="module")
def member():
    weights = _weights()
    report = owner._assess_mixed_local_modes(**_arguments(weights))
    tangent = owner._mixed_aligned_tangent(**_arguments(weights))
    basis = tuple(tuple(Q(j == column) for j in range(72)) for column in range(72))
    columns = tuple(tangent(direction) for direction in basis)
    destination = os.environ.get("TNFR_LOCAL_MODES_RECORD_PATH")
    if destination:
        write_json_once(
            destination,
            encode_exact_tree(
                {
                    "schema": "tnfr.mixed-local-modes-control-observations.v1",
                    "certificate": asdict(report),
                    "finite_inputs": {
                        "held_weights": weights,
                        "directions": "all72 standard basis directions",
                        "direction_count": 72,
                        "weight_membership": "constructed legal family member",
                    },
                    "precision": {
                        "oracle_decimal_digits": 220,
                        "oracle_residual_allowance": TOL,
                        "owner_interval_bits": 128,
                    },
                    "source_reassessed": False,
                    "spectrum_numerically_evaluated": False,
                    "new_trajectories": 0,
                }
            ),
        )
    return dict(
        weights=weights, report=report, tangent=tangent, basis=basis, columns=columns
    )


def test_complete_jet_tangent_matches_independent_laplacian_and_hessian(mp, member):
    weights, report = member["weights"], member["report"]
    observation = report.template_observation
    expected = _edges(weights)
    assert observation.edge_pairs == tuple((a, b) for a, b, _, _ in expected)
    assert observation.form_conductances == tuple(w for _, _, w, _ in expected)
    assert observation.phase_coefficients == tuple(s for _, _, _, s in expected)
    assert len(expected) == 180
    for direction, column in zip(member["basis"], member["columns"], strict=True):
        assert len(column) == 72
        oracle = _oracle(mp, direction, weights)
        for row, value in zip(column, oracle, strict=True):
            _contains(mp, row, value)
    # The zero phase/phase block and opposite form/phase block are both checked.
    assert all(
        row.contains(0) for column in member["columns"][36:] for row in column[36:]
    )
    for j in range(36):
        assert member["columns"][j][j].hi < 0
        assert member["columns"][36 + j][j].hi < 0
        assert member["columns"][j][36 + j].lo > 0
    # The full equilibrium is an exact odd-gap cancellation, not merely a
    # small rounded residual. Opposite rational turn gaps balance nodewise.
    for node in range(36):
        gaps = {}
        for a, b, _, sigma in expected:
            if node not in (a, b):
                continue
            other = b if node == a else a
            gap = (TURNS[other] - TURNS[node] + Q(1, 2)) % 1 - Q(1, 2)
            gaps[gap] = gaps.get(gap, Q(0)) + sigma
        assert all(value == gaps.get(-gap, Q(0)) for gap, value in gaps.items() if gap)
    assert all(
        row.contains(0) for row in observation.form_rates + observation.phase_rates
    )


def test_common_coordinates_are_two_separate_semisimple_zero_directions(mp, member):
    common_form = (Q(1),) * 36 + (Q(0),) * 36
    common_phase = common_form[36:] + common_form[:36]
    for vector in (common_form, common_phase):
        rows = member["tangent"](vector)
        assert all(row.contains(0) for row in rows)
        assert max(row.abs_max for row in rows) < Q(1, 10**30)
    for column in member["columns"]:
        assert sum(column[:36], I(0)).contains(0)
        assert sum(column[36:], I(0)).contains(0)
    # Independent left and right common vectors pair as 36I. A generalized
    # vector mapped to a nonzero common vector would contradict these sums.
    assert _dot(common_form, common_form) == _dot(common_phase, common_phase) == 36
    assert _dot(common_form, common_phase) == 0
    direction = tuple(Q((5 * j) % 11 - 5, 64) for j in range(72))
    actual = member["tangent"](direction)
    oracle = _oracle(mp, direction, member["weights"])
    for row, value in zip(actual, oracle, strict=True):
        _contains(mp, ALPHA * row, _mp(mp, ALPHA) * value)
    assert member["report"].original_time_rate_factor == ALPHA
    assert member["report"].common_coordinates == ("mean_form", "mean_phase_lift")
    assert member["report"].common_modes_semisimple_certified


def test_actual_graph_quadratic_bounds_and_tangent_storage_use_bare_cosines(mp, member):
    weights = member["weights"]
    edges = _edges(weights)
    assert all(Q(3, 4) <= w <= 1 for row in weights for w in row)
    for node in range(36):
        degree = sum(w for a, b, w, _ in edges if node in (a, b))
        assert degree <= 18
    assert mp.sin(mp.pi / 18) > _mp(mp, Q(1, 9))
    for a, b, _, sigma in edges:
        coefficient = _mp(mp, sigma) * mp.cos(2 * mp.pi * _mp(mp, TURNS[a] - TURNS[b]))
        assert coefficient > _mp(mp, Q(1, 9))
    probes = (
        (Q(1),) * 18 + (Q(-1),) * 18,
        (Q(1), Q(-1)) + (Q(0),) * 34,
        _center(Q((7 * j) % 13, 16) for j in range(36)),
    )
    for probe in probes:
        vector = tuple(_mp(mp, value) for value in probe)
        norm2 = _dot(vector, vector)
        lv = _laplacian(mp, vector, weights)
        kv = _laplacian(mp, vector, weights, phase=True)
        assert 3 * norm2 <= _dot(vector, lv) <= 36 * norm2
        assert _mp(mp, Q(4, 9)) * norm2 <= _dot(vector, kv) <= 36 * norm2
    # K is independent of the held form conductances. Its graph identity is
    # checked at every edge, with the distinct complete matching coefficient9.
    unit = ((Q(1),) * 63,) * 2
    vector = tuple(_mp(mp, value) for value in probes[-1])
    assert _laplacian(mp, vector, weights, phase=True) == _laplacian(
        mp, vector, unit, phase=True
    )
    assert _laplacian(mp, vector, weights) != _laplacian(mp, vector, unit)
    x, y = probes[-1], probes[1]
    rows = member["tangent"](x + y)
    lx = _laplacian(mp, tuple(_mp(mp, value) for value in x), weights)
    ky = _laplacian(mp, tuple(_mp(mp, value) for value in y), weights, phase=True)
    exact_rows = _oracle(mp, x + y, weights)
    derivative = _dot(lx, exact_rows[:36]) + _dot(ky, exact_rows[36:])
    assert abs(derivative + _dot(lx, lx) / 9) < _mp(mp, TOL)
    # Interval coefficients enclose the same positive quadratic storage row.
    coefficient_intervals = tuple(
        I(Q(mp.nstr(value, 205)) - TOL, Q(mp.nstr(value, 205)) + TOL)
        for value in lx + ky
    )
    enclosed = sum(
        (a * b for a, b in zip(coefficient_intervals, rows, strict=True)), I(0)
    )
    _contains(mp, enclosed, -_dot(lx, lx) / 9)


def test_a_legal_noncommuting_member_keeps_the_family_certificate(mp):
    _, new = _support()
    position = new.index((0, 2))
    first = tuple(Q(3, 4) if j == position else Q(1) for j in range(63))
    weights = first, (Q(1),) * 63
    vector = tuple(mp.mpf(j == 3) for j in range(36))
    lk = _laplacian(mp, _laplacian(mp, vector, weights, phase=True), weights)
    kl = _laplacian(mp, _laplacian(mp, vector, weights), weights, phase=True)
    commutator = tuple(a - b for a, b in zip(lk, kl, strict=True))
    expected = -mp.cos(2 * mp.pi / 9) / 4
    assert abs(commutator[0] - expected) < _mp(mp, TOL)
    assert abs(commutator[0]) > mp.mpf("0.1")
    report = owner._assess_mixed_local_modes(**_arguments(weights))
    assert report.real_negative_poles_certified and report.semisimple_poles_certified
    assert not report.commutation_assumed
    assert not report.supplied_weights_authenticated_as_source
    assert report.slow_mode_count == report.fast_mode_count == 35


def test_hermitian_pencil_separates_branches_without_a_common_eigenbasis(mp, member):
    weights = member["weights"]
    gamma = 1 / (1023 * mp.pi)
    delta = Q(1) - 48 * G**2
    assert delta == Q(187499, 187500)
    # v=L*u prices v*L^-1*v=u*L*u without ever inverting a singular full L.
    u = _center(mp.mpc(_mp(mp, Q((3 * j) % 7, 16)), (5 * j) % 11) for j in range(36))
    v = _laplacian(mp, u, weights)
    a = _inner(mp, u, v)
    norm2 = _inner(mp, v, v)
    k = _inner(mp, v, _laplacian(mp, v, weights, phase=True))
    assert abs(mp.im(a)) < _mp(mp, TOL)
    assert abs(mp.im(norm2)) < _mp(mp, TOL)
    assert abs(mp.im(k)) < _mp(mp, TOL)
    a, norm2, k = mp.re(a), mp.re(norm2), mp.re(k)
    b, c = norm2 / 9, gamma**2 * k / 81
    assert a > 0 and b > 0 and c > 0
    assert b**2 - 4 * a * c >= _mp(mp, delta) * b**2
    assert a / 36 - b / 6 + c <= -_mp(mp, delta) * norm2 / 108
    for position in (Q(-1, 12), Q(-1, 1000)):
        assert 2 * _mp(mp, position) * a + b > 0
    # Exact constructed 2D noncommuting pencil: this is an algebraic witness
    # for Jordan-chain solvability, not another 36-node spectrum calculation.
    gamma_q = G
    pole = -(gamma_q**2)
    inverse = (Q(1, 3), Q(1, 6))
    stiffness = ((10 - 27 * gamma_q**2, Q(-1)), (Q(-1), 10 - Q(27, 2) * gamma_q**2))
    kernel = (Q(1), Q(1))
    pencil_kernel = tuple(
        pole**2 * inverse[j] + pole / 9 + gamma_q**2 * sum(stiffness[j]) / 81
        for j in range(2)
    )
    assert pencil_kernel == (Q(0), Q(0))
    derivative_kernel = tuple(2 * pole * value + Q(1, 9) for value in inverse)
    assert _dot(kernel, derivative_kernel) == Q(2, 9) - gamma_q**2 > 0
    assert 3 * stiffness[0][1] != stiffness[0][1] * 6
    # For every possible generalized vector w, kernel*P*w=0, whereas the
    # required -kernel*P'*kernel is strictly negative. Multiplicity is allowed.
    for generalized in ((Q(0), Q(0)), (Q(2), Q(-3))):
        pw = tuple(
            gamma_q**2 * (generalized[j] - generalized[1 - j]) / 81 for j in range(2)
        )
        assert _dot(kernel, pw) == 0
    assert member["report"].separator_pencil_upper_bound == -delta / 108


def test_cycle_current_has_no_linear_readout_despite_nonzero_stationary_value(
    mp, member
):
    gamma_i = 1 / (1023 * pi_interval())
    direction = tuple(Q((7 * j) % 13 - 6, 64) for j in range(72))
    phases = tuple(
        Jet((2 * pi_interval() * turn, I(direction[36 + j])))
        for j, turn in enumerate(TURNS)
    )
    for cycle, winding in zip(CYCLES, (1, 2, 1, 2), strict=True):
        current = (
            sum((jet_sin(phases[a] - phases[b]) for a, b in cycle), Jet((I(0), I(0))))
            * gamma_i
            / 81
        )
        exact = -mp.sin(2 * mp.pi * winding / 9) / (9 * 1023 * mp.pi)
        _contains(mp, current.coeffs[0], exact)
        assert current.coeffs[0].hi < 0
        assert current.coeffs[1].contains(0)
        assert current.coeffs[1].abs_max < Q(1, 10**30)
        # Equal edge cosines telescope for every direction, not only this one.
        gradient = {node: 0 for node in range(36)}
        for a, b in cycle:
            gradient[a] += 1
            gradient[b] -= 1
        assert set(gradient.values()) == {0}
    assert any(not row.contains(0) for row in member["tangent"](direction))
    assert member["report"].linear_cycle_readout_zero_certified


def test_all_raw_weights_and_directions_precede_static_scientific_work(
    monkeypatch, member
):
    arguments = _arguments(member["weights"])

    def forbidden(*args, **kwargs):
        pytest.fail("invalid raw input reached static field, template or trigonometry")

    invalid = (
        True,
        np.bool_(False),
        None,
        "0",
        float("nan"),
        float("inf"),
        complex(1, 2),
    )
    with monkeypatch.context() as patch:
        for name in (
            "pi_interval",
            "_observe_mixed_cycle_current",
            "_mixed_phase_sensitive_field",
        ):
            patch.setattr(owner, name, forbidden)
        for factory in (owner._assess_mixed_local_modes, owner._mixed_aligned_tangent):
            for name in arguments:
                for bad in invalid + (
                    I(1),
                    Q(3, 4) - Q(1, 10**100),
                    Q(1) + Q(1, 10**100),
                ):
                    for position in (0, 62):
                        changed = tuple(
                            bad if j == position else value
                            for j, value in enumerate(arguments[name])
                        )
                        with pytest.raises((TypeError, ValueError, OverflowError)):
                            factory(**dict(arguments, **{name: changed}))
                for bad in ((1,) * 62, (1,) * 64, repeat(1), {0, 1}, {0: 1}, "1" * 63):
                    with pytest.raises((TypeError, ValueError)):
                        factory(**dict(arguments, **{name: bad}))
    # Compile against a forbidden captured field only after valid admission;
    # every invalid direction must fail before either Jet or that field runs.
    with monkeypatch.context() as patch:
        patch.setattr(
            owner,
            "_aligned_member",
            lambda *args: member["report"].template_observation,
        )
        patch.setattr(owner, "_mixed_phase_sensitive_field", lambda **kwargs: forbidden)
        tangent = owner._mixed_aligned_tangent(**arguments)
        patch.setattr(owner, "Jet", forbidden)
        for bad in invalid + (0.25,):
            for position in (0, 71):
                values = tuple(bad if j == position else Q(0) for j in range(72))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    tangent(values)
        for bad in ((0,) * 71, (0,) * 73, repeat(0), {0, 1}, {0: 0}, "0" * 72):
            with pytest.raises((TypeError, ValueError)):
                tangent(bad)


def test_consumed_support_and_law_cannot_be_replaced_by_cached_flags(
    monkeypatch, member
):
    arguments = _arguments(member["weights"])
    observation = member["report"].template_observation
    wrong_support = tuple(sorted(((0, 3),) + observation.edge_pairs[1:]))
    for changed in (
        replace(observation, edge_pairs=wrong_support),
        replace(
            observation,
            edge_pairs=((False, observation.edge_pairs[0][1]),)
            + observation.edge_pairs[1:],
        ),
        replace(
            observation,
            form_conductances=(Q(1, 2),) + observation.form_conductances[1:],
        ),
        replace(
            observation, form_conductances=(True,) + observation.form_conductances[1:]
        ),
        replace(
            observation, phase_coefficients=(Q(2),) + observation.phase_coefficients[1:]
        ),
        replace(
            observation, phase_coefficients=(True,) + observation.phase_coefficients[1:]
        ),
        replace(observation, first_new_edge_conductances=(Q(1),) * 63),
        replace(observation, physical_capacities=(Q(2),) * 36),
        replace(
            observation,
            physical_capacities=(True,) + observation.physical_capacities[1:],
        ),
        replace(observation, internal_divisor=Q(1)),
        replace(observation, full_state_dimension=2),
        replace(observation, clock="t=tau"),
        replace(observation, forms=(I(1),) * 36),
        replace(observation, phase_lifts=(I(0),) * 36),
        replace(observation, gamma_bounds=I(G)),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(
                owner, "_observe_mixed_cycle_current", lambda **kwargs: changed
            )
            with pytest.raises((TypeError, ValueError)):
                owner._assess_mixed_local_modes(**arguments)
    # Bounds cannot become an oscillation claim when their own premise fails.
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_aligned_member", lambda *args: observation)
        patch.setattr(owner, "cos", lambda value: I(0, Q(1, 9)))
        unavailable = owner._assess_mixed_local_modes(**arguments)
    assert unavailable.status == "unavailable"
    assert unavailable.slow_pole_interval is unavailable.fast_pole_interval is None
    assert unavailable.slow_mode_count is unavailable.fast_mode_count is None
    assert unavailable.common_mode_count is None
    assert not unavailable.real_negative_poles_certified
    assert not unavailable.semisimple_poles_certified
    assert not unavailable.common_modes_semisimple_certified
    assert not unavailable.linear_cycle_readout_zero_certified
    assert not unavailable.uniform_held_weight_family_certified
    assert not unavailable.nonlinear_nonoscillation_claimed


def test_immutable_local_certificate_retains_open_intervals_and_physical_boundaries(
    member, record_property
):
    report = member["report"]
    assert report.status == "uniform_real_semisimple_local_modes"
    assert report.form_laplacian_bounds == (Q(3), Q(36))
    assert report.phase_hessian_bounds == (Q(4, 9), Q(36))
    assert GL < report.gamma_bounds.lo <= report.gamma_bounds.hi < G
    assert report.minimum_phase_cosine_bounds.lo > Q(1, 9)
    assert report.discriminant_margin == Q(187499, 187500)
    assert report.separator == Q(-1, 6)
    assert report.slow_pole_interval == (Q(-1, 6), Q(0))
    assert report.fast_pole_interval == (Q(-4), Q(-1, 6))
    assert report.pole_intervals_are_open
    assert report.spatial_quotient_dimension == 35
    assert report.quotient_dimension == 70 and report.full_state_dimension == 72
    assert (
        report.slow_mode_count + report.fast_mode_count + report.common_mode_count == 72
    )
    for name in (
        "commutation_assumed",
        "euclidean_orthogonal_modes_claimed",
        "supplied_weights_authenticated_as_source",
        "acquired_source_at_equilibrium_claimed",
        "spectrum_numerically_evaluated",
        "trajectory_evaluated",
        "new_field_installed",
        "nonlinear_nonoscillation_claimed",
        "coordinate_monotonicity_claimed",
        "physical_wave_identified",
    ):
        assert getattr(report, name) is False
    with pytest.raises(FrozenInstanceError):
        report.slow_mode_count = 0
    for name in (
        "source",
        "phase_lifts",
        "gamma",
        "clock",
        "horizon",
        "noise",
        "observed_modes",
    ):
        with pytest.raises(TypeError):
            owner._assess_mixed_local_modes(
                **dict(_arguments(member["weights"]), **{name: 0})
            )
    for name in (
        "discriminant_margin",
        "separator",
        "separator_pencil_upper_bound",
        "slow_pole_interval",
        "fast_pole_interval",
        "slow_mode_count",
        "fast_mode_count",
        "common_mode_count",
        "real_negative_poles_certified",
        "semisimple_poles_certified",
        "common_modes_semisimple_certified",
        "linear_cycle_readout_zero_certified",
        "uniform_held_weight_family_certified",
        "spectrum_numerically_evaluated",
        "trajectory_evaluated",
    ):
        record_property(name, str(getattr(report, name)))
