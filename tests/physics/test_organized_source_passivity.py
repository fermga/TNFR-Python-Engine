"""Independent ideal-word work checks without events, sources or trajectories."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_source_capture as capture
from tnfr.physics import _organized_source_passivity as passivity
from tnfr.physics import _organized_source_port as port

# Build the declared support independently of the observer's edge inventory.
EDGES = tuple(
    (9 * child + slot, 9 * child + (slot + 1) % 9)
    for child in range(5)
    for slot in range(9)
) + tuple(
    (9 * a + slot, 9 * b + slot)
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
    for slot in range(9)
)
TURNS = (0,) * 45
FORM = tuple(
    Q(1, 5) + {0: Q(1, 64), 1: -Q(1, 128), 7: Q(1, 256), 20: -Q(1, 512)}.get(n, 0)
    for n in range(45)
)
CAPACITIES = tuple({0: Q(1, 2), 1: Q(3, 4), 10: Q(2)}.get(n, Q(1)) for n in range(45))


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Ideal-word work cannot execute events, flow or source producers")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (word_execution, "execute_network_operator_stage"),
            (capture, "_assess_organized_source_capture"),
            (capture, "_assess_organized_source_attachment"),
            (port, "_assess_organized_source_attachment"),
            (port, "_assess_organized_source_emission"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


@pytest.fixture(scope="module")
def phases(mp):
    return tuple(
        Q(
            float(
                2 * mp.pi / 9 * (n % 9 - 4)
                + mp.mpf(1) / 40
                + ({0: mp.mpf(1) / 100, 1: mp.mpf(1) / 300}.get(n, 0))
            )
        )
        for n in range(45)
    )


def _assess(initial_phases, **overrides):
    inputs = dict(
        form=FORM,
        phases=initial_phases,
        phase_turns=TURNS,
        capacities=CAPACITIES,
        reception_mix=Q(1, 2),
        phase_coefficient=Q(0.3),
        silence_capacity_factor=Q(1, 2),
    )
    inputs.update(overrides)
    return passivity._assess_organized_source_word_passivity(**inputs)


@pytest.fixture(scope="module")
def reports(phases):
    return {
        coefficient: _assess(phases, phase_coefficient=coefficient)
        for coefficient in (Q(0), Q(0.3), Q(1))
    }


@pytest.fixture(scope="module")
def report(reports):
    return reports[Q(0.3)]


def _gradient(form):
    result = [Q(0)] * 45
    for a, b in EDGES:
        result[a] += form[a] - form[b]
        result[b] += form[b] - form[a]
    return tuple(result)


def _form_storage(form):
    return sum(((form[b] - form[a]) ** 2 / 2 for a, b in EDGES), Q(0))


def _mean(form, capacities):
    return sum((x / nu for x, nu in zip(form, capacities)), Q(0)) / sum(
        (1 / nu for nu in capacities), Q(0)
    )


def test_actual_support_and_independent_reception_energy(report):
    geometry = report.domain.observation.unit_capacity_reference.geometry
    assert len(EDGES) == len(geometry.weighted_edges) == 90
    assert {frozenset((a, b)) for a, b in EDGES} == {
        frozenset((a, b)) for a, b, _ in geometry.weighted_edges
    }
    assert all(weight == 1 for _, _, weight in geometry.weighted_edges)
    q = _gradient(FORM)
    assert report.domain.observation.unit_capacity_reference.form_gradient == q
    successor = list(FORM)
    for target in (0, 1):
        neighbors = [b if a == target else a for a, b in EDGES if target in (a, b)]
        assert len(neighbors) == 5
        successor[target] = (FORM[target] + sum(FORM[n] for n in neighbors) / 5) / 2
    assert report.reception_form == tuple(successor)
    exact_work = _form_storage(successor) - _form_storage(FORM)
    assert report.reception_work == exact_work < 0
    assert exact_work <= report.reception_work_upper_bound < 0
    assert report.reception_symmetric_loss > 0
    assert report.reception_antisymmetric_loss > 0
    assert exact_work == -(
        report.reception_symmetric_loss + report.reception_antisymmetric_loss
    )
    assert min(FORM) <= min(successor) <= max(successor) <= max(FORM)


@pytest.mark.parametrize("beta", (Q(1, 4), Q(1, 2), Q(1)))
@pytest.mark.parametrize("mode", ("symmetric", "antisymmetric"))
def test_reception_principal_matrix_modes_have_independent_exact_loss(
    phases, beta, mode
):
    amplitude = Q(1, 128)
    sign = 1 if mode == "symmetric" else -1
    form = (Q(1, 5) + amplitude, Q(1, 5) + sign * amplitude) + (Q(1, 5),) * 43
    result = _assess(phases, form=form, reception_mix=beta)
    assert result.admitted
    q = _gradient(form)
    eigenvalue = 4 if mode == "symmetric" else 6
    assert q[:2] == (eigenvalue * amplitude, sign * eigenvalue * amplitude)
    expected = list(form)
    expected[0] -= beta * q[0] / 5
    expected[1] -= beta * q[1] / 5
    work = _form_storage(expected) - _form_storage(form)
    assert result.reception_form == tuple(expected)
    assert result.reception_work == work < 0
    active = getattr(result, f"reception_{mode}_loss")
    inactive = getattr(
        result,
        (
            "reception_antisymmetric_loss"
            if mode == "symmetric"
            else "reception_symmetric_loss"
        ),
    )
    assert active == -work and inactive == 0
    if mode == "antisymmetric":
        assert result.reception_work_upper_bound == work
    else:
        assert work < result.reception_work_upper_bound


@pytest.mark.parametrize("coefficient", (Q(0), Q(0.3), Q(1)))
def test_phase_joint_work_encloses_independent_high_precision_all_edge_reset(
    reports, phases, mp, coefficient
):
    result = reports[coefficient]
    angles = tuple(_mp(mp, value) for value in phases)
    displacements = [mp.mpf(0)] * 45
    isolated = []
    for target in (0, 1):
        neighbors = [b if a == target else a for a, b in EDGES if target in (a, b)]
        z = sum(mp.exp(mp.j * (angles[n] - angles[target])) for n in neighbors)
        argument = mp.arg(z)
        _contains(mp, result.relative_resultant_real_bounds[target], z.real)
        _contains(mp, result.relative_resultant_imaginary_bounds[target], z.imag)
        _contains(mp, result.phase_angle_bounds[target], argument)
        assert z.real > 4 and abs(argument) < mp.mpf(1) / 8
        displacements[target] = _mp(mp, coefficient) * argument
        _contains(mp, result.phase_displacement_bounds[target], displacements[target])
        isolated.append(
            sum(
                mp.cos(angles[n] - angles[target])
                - mp.cos(angles[n] - angles[target] - displacements[target])
                for n in neighbors
            )
        )
        _contains(mp, result.isolated_phase_work_bounds[target], isolated[-1])
    before = sum(1 - mp.cos(angles[b] - angles[a]) for a, b in EDGES)
    after = sum(
        1 - mp.cos(angles[b] + displacements[b] - angles[a] - displacements[a])
        for a, b in EDGES
    )
    work = after - before
    correction = work - sum(isolated)
    _contains(mp, result.phase_work_bounds, work)
    _contains(mp, result.shared_edge_correction_bounds, correction)
    assert result.work_decomposition_residual_bounds.contains(0)
    assert result.work_decomposition_residual_bounds.width < Q(1, 2**100)
    assert work <= _mp(mp, result.phase_work_upper_bound)
    assert _mp(mp, result.reception_work) + work <= _mp(
        mp, result.word_work_upper_bound
    )
    if coefficient:
        assert correction != 0
        assert result.strict_phase_loss_certified and work < 0
        assert result.phase_work_upper_bound < 0
    else:
        assert result.phase_work_bounds == result.shared_edge_correction_bounds == I(0)
        assert result.isolated_phase_work_bounds == (I(0), I(0))
        assert result.phase_work_upper_bound == 0
        assert not result.strict_phase_loss_certified


def test_independent_target_angles_and_nonzero_weighted_means_are_not_projected(
    report, phases
):
    assert report.admitted and report.all_reset_prefixes_passive
    assert report.retained_windings == (1,) * 5
    assert report.status == "certified_conditional_ideal_word_passivity"
    assert not report.unresolved_conditions
    reference = report.domain.observation.unit_capacity_reference
    assert reference.form == FORM and reference.phases == phases
    assert report.domain.observation.capacities == CAPACITIES
    assert sum(FORM) != 0
    assert sum(x / nu for x, nu in zip(FORM, CAPACITIES)) != 0
    a0, a1 = report.phase_angle_bounds
    assert (a0 + a1).hi < 0
    assert not (a0 + a1).contains(0)
    expected_capacities = (Q(1, 4), Q(3, 8)) + CAPACITIES[2:]
    assert report.final_capacities == expected_capacities
    assert report.silence_work == 0
    assert report.form_mean_before == _mean(FORM, CAPACITIES)
    assert report.form_mean_after_reception == _mean(report.reception_form, CAPACITIES)
    assert report.form_mean_after_silence == _mean(
        report.reception_form, expected_capacities
    )
    assert report.reception_mean_jump == (
        report.form_mean_after_reception - report.form_mean_before
    )
    assert report.silence_mean_jump == (
        report.form_mean_after_silence - report.form_mean_after_reception
    )
    assert report.reception_mean_jump != 0 and report.silence_mean_jump != 0


def test_uniform_form_gives_zero_reception_work_without_false_rail_admission(phases):
    result = _assess(phases, form=(Q(2),) * 45)
    assert result.admitted and result.domain.geometric_retention_certified
    assert not result.domain.form_rails_certified
    assert result.reception_work == result.reception_work_upper_bound == 0
    assert result.reception_symmetric_loss == result.reception_antisymmetric_loss == 0
    assert not result.strict_reception_loss_certified
    assert result.strict_phase_loss_certified
    assert result.form_mean_before == result.form_mean_after_silence == 2
    assert result.reception_mean_jump == result.silence_mean_jump == 0
    assert (
        "passivity_does_not_supply_form_rails_runtime_policies_or_event_occurrence"
        in (result.scope)
    )


@pytest.mark.parametrize("control", ("wrong_turn", "high_form_energy", "zero_winding"))
def test_actual_geometric_domain_rejects_without_borrowing_phase_or_work_verdicts(
    phases, control
):
    overrides = {}
    if control == "wrong_turn":
        overrides["phase_turns"] = (1,) + (0,) * 44
    elif control == "high_form_energy":
        overrides["form"] = (Q(1),) + (Q(0),) * 44
    else:
        phases = (Q(0),) * 45
    result = _assess(phases, **overrides)
    assert not result.admitted
    assert not result.domain.geometric_retention_certified
    assert result.unresolved_conditions
    assert result.reception_work is None and result.phase_work_bounds is None
    assert result.word_work_upper_bound is None and result.final_capacities is None
    assert result.retained_windings is None
    assert not result.all_reset_prefixes_passive
    if control == "high_form_energy":
        assert result.domain.in_acute_cell
        assert not result.domain.below_energy_ceiling
    elif control == "zero_winding":
        assert result.domain.below_energy_ceiling
        assert not result.domain.in_acute_cell


@pytest.mark.parametrize(
    "field,invalid",
    (
        ("reception_mix", 0),
        ("reception_mix", Q(5, 4)),
        ("reception_mix", True),
        ("reception_mix", float("nan")),
        ("reception_mix", Q(1, 3)),
        ("phase_coefficient", -1),
        ("phase_coefficient", Q(5, 4)),
        ("phase_coefficient", True),
        ("phase_coefficient", float("inf")),
        ("phase_coefficient", Q(1, 3)),
        ("silence_capacity_factor", 0),
        ("silence_capacity_factor", 1),
        ("silence_capacity_factor", -Q(1, 2)),
        ("silence_capacity_factor", True),
        ("silence_capacity_factor", float("nan")),
        ("silence_capacity_factor", Q(1, 3)),
    ),
)
def test_invalid_or_unrepresented_factors_reject_before_state_observation(
    monkeypatch, phases, field, invalid
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid factor reached geometric observation")

    monkeypatch.setattr(
        passivity, "_assess_organized_source_weighted_mean_retention", forbidden
    )
    with pytest.raises((TypeError, ValueError)):
        _assess(phases, **{field: invalid})


@pytest.mark.parametrize("field", ("form", "phases", "capacities"))
@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), 1j))
def test_invalid_physical_primitives_reject_before_geometry(
    monkeypatch, phases, field, invalid
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid primitive reached geometric construction")

    monkeypatch.setattr(port, "_derive_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(phases, **{field: (invalid,) + (Q(1),) * 44})


@pytest.mark.parametrize("field", ("form", "phases", "capacities", "phase_turns"))
def test_wrong_vector_length_rejects_before_geometry(monkeypatch, phases, field):
    def forbidden(*args, **kwargs):
        pytest.fail("Wrong vector length reached geometric construction")

    monkeypatch.setattr(port, "_derive_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(phases, **{field: (0,) * 44})


@pytest.mark.parametrize("invalid", (0, -Q(1, 2)))
def test_nonpositive_capacity_cannot_claim_reciprocal_continuation(
    monkeypatch, phases, invalid
):
    def forbidden(*args, **kwargs):
        pytest.fail("Nonpositive capacity reached phase observation")

    monkeypatch.setattr(capture, "_observe_organized_source_raw_capacity", forbidden)
    with pytest.raises(ValueError):
        _assess(phases, capacities=(invalid,) + CAPACITIES[1:])


@pytest.mark.parametrize("invalid", (True, Q(0), 0.0, "0"))
def test_noninteger_lift_turns_reject_before_phase_observation(
    monkeypatch, phases, invalid
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid turn reached phase observation")

    monkeypatch.setattr(capture, "_observe_organized_source_raw_capacity", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(phases, phase_turns=(invalid,) + (0,) * 44)
