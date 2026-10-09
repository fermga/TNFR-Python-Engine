"""Independent static controls for whole-set cycle recovery and trapping gates.

No recovery trajectory is integrated. Synthetic forecast reports exercise
endpoint/set-selection plumbing, not a claim of numerical propagation.
"""

import json
import pickle
from copy import copy
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.mathematics._rational_interval import I
from tnfr.physics._sine_admission import _admit_sine_source
from tnfr.physics.relational_sine_forecast import SineForecast
from tnfr.physics.relational_sine_pattern import (
    SineRelativeForecast,
    bound_relational_sine_pattern,
)
from tnfr.physics.relational_sine_recovery import (
    assess_sine_cycle_identity,
    certify_sine_cycle_recovery,
    certify_sine_sector_capture,
)
from tnfr.sdk import export_to_json, relational_report_to_dict

MODEL = RelationalExchangeModel(1, phase_domain="regular")
ERROR = Q(1, 4096)
RADIUS = Q(1, 16)
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)
REVERSIBLE_MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")


def _graph(*, size=5, sign=1, capacity=1):
    graph = nx.cycle_graph(size)
    for i in graph:
        form = Q((1, -1, 0, 0, 0)[i], 1024) if size == 5 else Q(0)
        graph.nodes[i].update(EPI=float(form), theta=sign * 5 * i / 4, nu_f=capacity)
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _pattern(graph=None, *, model=MODEL, form_errors=None, phase_errors=None):
    graph = _graph() if graph is None else graph
    size = len(graph)
    return bound_relational_sine_pattern(
        graph,
        reference_node=0,
        reference_model=model,
        form_error_bounds=(ERROR,) * size if form_errors is None else form_errors,
        phase_error_bounds=(ERROR,) * size if phase_errors is None else phase_errors,
    )


def _certificate(pattern=None, **changes):
    arguments = dict(cycle=tuple(range(5)), winding=1, radius=RADIUS)
    arguments.update(changes)
    return certify_sine_cycle_recovery(
        _pattern() if pattern is None else pattern, **arguments
    )


@pytest.fixture(scope="module")
def admitted_pattern():
    return _pattern()


@pytest.mark.parametrize("consumer", ("recovery", "identity"))
@pytest.mark.parametrize(
    "field,value",
    (
        ("capacity", ()),
        ("capacity", (1, 1, True, 1, 1)),
        ("capacity", (1, 1, -1, 1, 1)),
        ("law", "different_complete_law"),
        ("neighbors", ((1,),) * 5),
        ("degrees", (2, 2, 1, 2, 2)),
        ("nominal_form", (0, 0, True, 0, 0)),
        ("nominal_phase", (0, 1, 2, 3)),
        ("form_error_bounds", (-ERROR,) + (ERROR,) * 4),
        ("phase_error_bounds", (-ERROR,) + (ERROR,) * 4),
    ),
)
def test_recovery_and_identity_revalidate_source_primitives(
    admitted_pattern, consumer, field, value
):
    source = admitted_pattern
    if consumer == "identity":
        source = replace(source, reference_model=REVERSIBLE_MODEL)
    source = replace(source, **{field: value})
    with pytest.raises(ADMISSION_ERRORS):
        (_certificate if consumer == "recovery" else _identity)(source)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    assert _mp(bound.lo) <= value <= _mp(bound.hi)


@pytest.mark.parametrize("winding", (-1, 1))
def test_noisy_deformed_cycle_is_admitted_with_independent_constants(winding):
    pattern = _pattern(_graph(sign=winding))
    report = _certificate(pattern, winding=winding)
    assert report.admitted
    assert report.norm_margin > 0
    assert report.energy_margin > 0
    assert report.hypothesis_failures == report.unresolved_conditions == ()
    assert report.norm_squared_upper_bound <= Q(2184757, 1284505600) < RADIUS**2
    assert report.excess_storage_upper_bound <= Q(59951, 308281344) < Q(1, 2560)
    with mp.workdps(90):
        gap = 4 * mp.sin(mp.pi / 5) ** 2
        _contains(report.spectral_gap_bounds, gap)
        assert 0 < _mp(report.spectral_gap_lower_bound) <= gap
        cosine = mp.cos(2 * mp.pi / 5 + mp.sqrt(2) * _mp(RADIUS))
        assert 0 < _mp(report.cosine_lower_bound) <= cosine
        assert (
            _mp(report.barrier_lower_bound)
            <= gap * min(1, cosine) * _mp(RADIUS) ** 2 / 2
        )
    assert report.observation_time is None
    assert report.input_forecast_admitted is None


def test_all_declared_residual_corners_are_bounded_by_the_recovery_budget():
    pattern = _pattern()
    report = _certificate(pattern)
    with mp.workdps(90):
        beta, n = _mp(MODEL.storage_scale), len(pattern.nodes)
        reference = tuple(2 * mp.pi * i / n for i in range(n))
        target_energy = beta * n * (1 - mp.cos(2 * mp.pi / n))
        for signs in product((-1, 1), repeat=n):
            # Independent common offsets disappear only after full centering.
            form = tuple(
                _mp(value) + 7 + sign * _mp(ERROR)
                for value, sign in zip(pattern.nominal_form, signs)
            )
            phase = tuple(
                _mp(value) - 3 - sign * _mp(ERROR)
                for value, sign in zip(pattern.nominal_phase, signs)
            )
            mean_form = sum(form) / n
            deviations = tuple(
                value - target for value, target in zip(phase, reference)
            )
            mean_deviation = sum(deviations) / n
            form_norm = sum((value - mean_form) ** 2 for value in form)
            phase_norm = sum((value - mean_deviation) ** 2 for value in deviations)
            energy = (
                sum(
                    (form[i] - form[j]) ** 2 / 2
                    + beta * (1 - mp.cos(phase[j] - phase[i]))
                    for i, j in pattern.edges
                )
                - target_energy
            )
            assert form_norm <= _mp(report.form_norm_squared_upper_bound)
            assert phase_norm <= _mp(report.phase_norm_squared_upper_bound)
            assert form_norm + phase_norm <= _mp(report.norm_squared_upper_bound)
            assert energy <= _mp(report.excess_storage_upper_bound)
            assert energy < _mp(report.barrier_lower_bound)


def test_orientation_reversal_and_negated_winding_preserve_target_geometry():
    pattern = _pattern()
    forward = _certificate(pattern)
    reversed_cycle = _certificate(pattern, cycle=tuple(reversed(range(5))), winding=-1)
    assert forward.admitted and reversed_cycle.admitted
    assert forward.norm_squared_upper_bound == reversed_cycle.norm_squared_upper_bound
    assert (
        forward.excess_storage_upper_bound == reversed_cycle.excess_storage_upper_bound
    )
    assert forward.barrier_lower_bound == reversed_cycle.barrier_lower_bound


def test_common_origins_and_common_lift_turn_do_not_change_recovery():
    graph = _graph()
    original = _certificate(_pattern(graph))
    for node in graph:
        graph.nodes[node]["EPI"] += 7
        graph.nodes[node]["theta"] -= 3
    translated = _certificate(_pattern(graph), phase_turns=(2,) * 5)
    assert translated.admitted
    assert original.norm_squared_upper_bound == translated.norm_squared_upper_bound
    assert original.excess_storage_upper_bound == translated.excess_storage_upper_bound


def test_independent_wrapped_representative_needs_its_declared_lift_turn():
    graph = _graph()
    graph.nodes[2]["theta"] -= 2 * float(mp.pi)
    pattern = _pattern(graph)
    missing = _certificate(pattern)
    lifted = _certificate(pattern, phase_turns=(0, 0, 1, 0, 0))
    assert not missing.admitted
    assert lifted.admitted
    assert lifted.phase_turns == (0, 0, 1, 0, 0)


def test_zero_capacity_is_not_admitted_by_a_positive_capacity_limit():
    graph = _graph()
    graph.nodes[0]["nu_f"] = graph.nodes[2]["nu_f"] = 0
    pattern = _pattern(graph)
    report = _certificate(pattern)
    assert not report.admitted
    assert report.hypothesis_failures
    # Distinct actual frozen forms obstruct uniform-form recovery, however
    # close they initially are. A small geometric budget cannot override it.
    assert pattern.nominal_form[0] != pattern.nominal_form[2]
    assert pattern.form_rate_bounds[0] == pattern.form_rate_bounds[2] == I(0)


def test_zero_damping_keeps_nonzero_excess_instead_of_guaranteeing_recovery():
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    pattern = _pattern(model=model)
    report = _certificate(pattern)
    assert not report.admitted
    assert report.hypothesis_failures
    assert pattern.continuous_loss_bounds == I(0)
    assert pattern.form_storage_bounds.lo > 0


def test_tiny_strictly_positive_capacity_preserves_geometric_admission():
    ordinary = _certificate()
    slow = _certificate(_pattern(_graph(capacity=2.0**-100)))
    assert slow.admitted
    assert slow.norm_margin == ordinary.norm_margin
    assert slow.energy_margin == ordinary.energy_margin
    assert all(bound.lo > 0 for bound in slow.capacity_bounds)
    # No uniform rate estimate as capacity tends to zero is asserted.


def test_large_residual_set_is_unavailable_not_a_proof_of_instability():
    report = _certificate(_pattern(form_errors=(Q(1, 4),) * 5))
    assert not report.admitted
    assert report.unresolved_conditions
    assert report.hypothesis_failures == ()


def test_too_large_declared_recovery_radius_cannot_use_an_acute_hessian():
    report = _certificate(radius=1)
    assert not report.admitted
    assert report.cosine_lower_bound is None


def test_quarter_turn_cycle_target_is_not_strictly_acute():
    pattern = _pattern(_graph(size=4))
    with pytest.raises(ValueError, match="4\\*abs\\(winding\\)<n"):
        _certificate(pattern, cycle=(0, 1, 2, 3))


@pytest.mark.parametrize("change", ("chord", "incomplete", "duplicate", "wrong_order"))
def test_cycle_must_match_every_node_and_every_support_edge(change):
    graph, cycle = _graph(), tuple(range(5))
    if change == "chord":
        graph.add_edge(0, 2)
    elif change == "incomplete":
        cycle = (0, 1, 2, 3)
    elif change == "duplicate":
        cycle = (0, 1, 2, 3, 3)
    else:
        cycle = (0, 1, 3, 2, 4)
    with pytest.raises(ADMISSION_ERRORS):
        _certificate(_pattern(graph), cycle=cycle)


@pytest.mark.parametrize(
    "keyword,value",
    (
        ("winding", True),
        ("winding", 1.0),
        ("radius", 0),
        ("radius", -1),
        ("radius", True),
        ("radius", float("inf")),
        ("phase_turns", (0, 0)),
        ("phase_turns", (0, 0, False, 0, 0)),
        ("phase_turns", (0, 0, Q(1, 2), 0, 0)),
    ),
)
def test_invalid_family_or_chart_declarations_reject(keyword, value):
    with pytest.raises(ADMISSION_ERRORS):
        _certificate(**{keyword: value})


def _initial_forecast_report(pattern):
    """Synthetic unavailable solver report, still at its admitted initial time."""
    initial = (
        pattern.relative_form_bounds
        + pattern.relative_phase_bounds
        + (I(pattern.capacity[-1]),)
    )
    full = SineForecast(
        model=pattern.reference_model,
        neighbors=pattern.neighbors,
        visible_capacity=pattern.capacity[:-1],
        initial_box=initial,
        observation_time=Q(2),
        end_time=Q(3),
        time_step=Q(1, 8),
        order=6,
        steps=(),
        validated_end_time=Q(2),
        endpoint=initial,
        failed_tube=None,
        status="unavailable",
        reasons=("synthetic_initial_endpoint_plumbing_only",),
    )
    return SineRelativeForecast(
        pattern=pattern,
        full_forecast=full,
        relative_form_bounds=pattern.relative_form_bounds,
        relative_phase_bounds=pattern.relative_phase_bounds,
        reference_form_displacement_bounds=I(0),
        reference_phase_displacement_bounds=I(0),
    )


def test_correlated_measurement_set_is_not_the_cartesian_forecast_endpoint():
    pattern = _pattern(form_errors=(Q(1, 64), 0, 0, 0, 0), phase_errors=(0,) * 5)
    measured = _certificate(pattern)
    boxed = _certificate(_initial_forecast_report(pattern))
    assert measured.admitted
    assert not boxed.admitted
    assert boxed.excess_storage_upper_bound > measured.excess_storage_upper_bound
    assert boxed.uncertainty_scope != measured.uncertainty_scope
    assert boxed.observation_time == 2
    assert boxed.input_forecast_admitted is False
    assert boxed.input_forecast_requested_end_time == 3


def test_recovery_at_partial_endpoint_does_not_promote_requested_forecast_horizon():
    forecast = _initial_forecast_report(
        _pattern(form_errors=(0,) * 5, phase_errors=(0,) * 5)
    )
    report = _certificate(forecast)
    assert report.admitted
    assert not forecast.admitted
    assert report.input_forecast_admitted is False
    assert report.observation_time == forecast.full_forecast.validated_end_time == 2
    assert report.input_forecast_requested_end_time == 3


@pytest.mark.parametrize("size", (3, 6, 51))
def test_consensus_family_uses_full_cycle_spectral_gap(size):
    graph = nx.cycle_graph(size)
    for i in graph:
        graph.nodes[i].update(EPI=7, theta=-3, nu_f=2.0**-i)
    pattern = _pattern(graph, form_errors=(0,) * size, phase_errors=(0,) * size)
    report = _certificate(pattern, cycle=tuple(graph), winding=0)
    assert report.admitted
    assert report.norm_squared_upper_bound == 0
    assert report.excess_storage_upper_bound == 0
    with mp.workdps(90):
        _contains(report.spectral_gap_bounds, 4 * mp.sin(mp.pi / size) ** 2)
    assert report.source is pattern
    if size == 51:
        # Primitive admission must not impose the separate generic geometry
        # budget on the preexisting closed-form full-cycle certificate.
        with pytest.raises(ValueError, match="2 to 32 ordered nodes"):
            pattern.certify_pattern_recovery(
                target_phase_turns=(Q(0),) * size, radius=RADIUS
            )
        with pytest.raises(ValueError, match="2 to 32 ordered nodes"):
            pattern.certify_sector_capture(edge_turn_offsets=(0,) * size)


def test_endpoint_capacity_interval_must_exclude_zero_for_whole_family():
    source = _initial_forecast_report(_pattern())
    uncertain = replace(
        source.full_forecast, endpoint=source.full_forecast.endpoint[:-1] + (I(0, 1),)
    )
    unavailable = _certificate(replace(source, full_forecast=uncertain))
    assert not unavailable.admitted
    assert "strictly_positive_held_capacity_required" in unavailable.hypothesis_failures
    positive = replace(
        uncertain, endpoint=uncertain.endpoint[:-1] + (I(Q(1, 2**100), 1),)
    )
    admitted = _certificate(replace(source, full_forecast=positive))
    assert admitted.admitted


def test_exact_positive_capacities_survive_outward_rounding_to_zero_lower_bound():
    tiny = Q(1, 2**200)
    static = _certificate(_pattern(_graph(capacity=float(tiny))))
    assert static.admitted
    assert static.exact_held_capacity == (tiny,) * 5
    assert all(value.lo == 0 for value in static.capacity_bounds)
    # Exact held visible capacities are not uncertain augmented coordinates.
    graph = _graph()
    graph.nodes[0]["nu_f"] = float(tiny)
    source = _initial_forecast_report(_pattern(graph))
    visible = _certificate(source)
    assert visible.admitted
    assert visible.exact_held_capacity[0] == tiny
    assert visible.capacity_bounds[0].lo == 0
    assert visible.exact_held_capacity[-1] is None
    # The final augmented coordinate genuinely contains zero after interval
    # materialization; source nominal positivity cannot remove that point.
    graph.nodes[4]["nu_f"] = float(tiny)
    augmented = _certificate(_initial_forecast_report(_pattern(graph)))
    assert augmented.capacity_bounds[-1].lo == 0
    assert not augmented.admitted
    assert "strictly_positive_held_capacity_required" in augmented.hypothesis_failures


@pytest.mark.parametrize("change", ("model", "support", "layout"))
def test_forecast_model_support_and_full_coordinate_layout_must_match(change):
    source = _initial_forecast_report(_pattern())
    full = source.full_forecast
    if change == "model":
        full = replace(full, model=RelationalExchangeModel(2, phase_domain="regular"))
    elif change == "support":
        full = replace(full, neighbors=((1,),) + full.neighbors[1:])
    else:
        full = replace(full, endpoint=full.endpoint[:-1])
    with pytest.raises(ValueError, match="must match"):
        _certificate(replace(source, full_forecast=full))


@pytest.mark.parametrize("consumer", ("shared", "recovery", "sector", "identity"))
@pytest.mark.parametrize(
    "field,value,model",
    (
        ("storage_scale", True, MODEL),
        ("epi_weight", False, REVERSIBLE_MODEL),
        ("phase_weight", True, REVERSIBLE_MODEL),
    ),
)
def test_forecast_model_equality_cannot_mask_boolean_coefficients(
    consumer, field, value, model
):
    source = _initial_forecast_report(_pattern(model=model))
    declared = copy(model)
    object.__setattr__(declared, field, value)
    assert declared == model  # Numerical equality does not admit a physical scalar.
    source = replace(
        source, full_forecast=replace(source.full_forecast, model=declared)
    )
    readers = {
        "shared": _admit_sine_source,
        "recovery": _certificate,
        "sector": lambda item: certify_sine_sector_capture(
            item, edge_turn_offsets=(0, -1, 0, 0, 0)
        ),
        "identity": _identity,
    }
    with pytest.raises(TypeError, match="boolean"):
        readers[consumer](source)


@pytest.mark.parametrize(
    "field,value",
    (("storage_scale", 2), ("epi_weight", Q(1, 3)), ("phase_weight", Q(2, 3))),
)
def test_shared_forecast_admission_checks_each_declared_model_coefficient(field, value):
    source = _initial_forecast_report(_pattern())
    declared = copy(source.full_forecast.model)
    object.__setattr__(declared, field, value)
    source = replace(
        source, full_forecast=replace(source.full_forecast, model=declared)
    )
    with pytest.raises(ValueError, match="model must match"):
        _admit_sine_source(source)


@pytest.mark.parametrize("declaration", ("pattern", "full_forecast"))
@pytest.mark.parametrize("consumer", ("recovery", "sector"))
def test_equivalent_forecast_neighbor_order_preserves_certificate(
    declaration, consumer
):
    source = _initial_forecast_report(_pattern())
    declared = getattr(source, declaration)
    rows = tuple(tuple(reversed(row)) for row in declared.neighbors)
    reordered = replace(source, **{declaration: replace(declared, neighbors=rows)})
    read = (
        _certificate
        if consumer == "recovery"
        else lambda item: certify_sine_sector_capture(
            item, edge_turn_offsets=(0, -1, 0, 0, 0)
        )
    )
    original, actual = read(source), read(reordered)
    assert original.admitted and actual.admitted
    assert actual.source is reordered
    assert replace(actual, source=original.source) == original


@pytest.mark.parametrize(
    "field,value",
    (
        ("observation_time", True),
        ("observation_time", -1),
        ("validated_end_time", 1),
        ("validated_end_time", 4),
        ("end_time", 2),
        ("freeze_hidden", True),
    ),
)
def test_forecast_time_and_held_capacity_domain_are_revalidated(
    admitted_pattern, field, value
):
    source = _initial_forecast_report(admitted_pattern)
    full = replace(source.full_forecast, **{field: value})
    with pytest.raises(ADMISSION_ERRORS):
        _certificate(replace(source, full_forecast=full))


def test_export_and_capture_provenance_remain_detached(tmp_path):
    graph = _graph()
    pattern = _pattern(graph)
    before = pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )
    report = pattern.certify_cycle_recovery(
        cycle=tuple(range(5)), winding=1, radius=RADIUS
    )
    assert report.admitted
    assert (
        pickle.dumps(
            (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
            protocol=5,
        )
        == before
    )
    payload = report.to_dict()
    path = tmp_path / "whole-set-recovery.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert report.source is pattern


def _identity(source=None, **changes):
    arguments = dict(
        cycle=tuple(range(5)),
        winding=1,
        radius=RADIUS,
        excess_ceiling=Q(1, 4096),
        form_mean_bounds=(-1, 1),
    )
    arguments.update(changes)
    return assess_sine_cycle_identity(
        _pattern(model=REVERSIBLE_MODEL) if source is None else source,
        **arguments,
    )


def test_reversible_noisy_winding_pattern_is_trapped_in_a_recurrent_family():
    report = _identity()
    assert report.family_admitted
    assert report.source_set_trapping_certified
    assert report.relative_source_family_membership == "certified_inside"
    assert report.family_almost_everywhere_recurrence_certified
    assert report.absolute_source_family_membership == (
        "unavailable_common_form_origin_unobserved"
    )
    assert report.individual_recurrence_status == "unavailable_for_chosen_state"
    assert report.phase_norm_squared_upper_bound > 0
    assert report.target_phase_turns == tuple(Q(i, 5) for i in range(5))
    assert report.family_barrier_margin > 0
    assert report.source_family_excess_margin > 0
    assert report.hypothesis_failures == report.family_unresolved_conditions == ()
    assert report.source_trapping_unresolved_conditions == ()
    pattern = report.source
    with mp.workdps(90):
        target = tuple(2 * mp.pi * i / 5 for i in range(5))
        target_energy = 5 * (1 - mp.cos(2 * mp.pi / 5))
        exact_barrier = (
            4
            * mp.sin(mp.pi / 5) ** 2
            * mp.cos(2 * mp.pi / 5 + mp.sqrt(2) * _mp(RADIUS))
            * _mp(RADIUS) ** 2
            / 2
        )
        assert _mp(report.excess_ceiling) < _mp(report.barrier_lower_bound)
        assert _mp(report.barrier_lower_bound) <= exact_barrier
        for signs in product((-1, 1), repeat=5):
            x = tuple(
                _mp(value) + sign * _mp(ERROR)
                for value, sign in zip(pattern.nominal_form, signs)
            )
            theta = tuple(
                _mp(value) - sign * _mp(ERROR)
                for value, sign in zip(pattern.nominal_phase, signs)
            )
            delta = tuple(value - reference for value, reference in zip(theta, target))
            centered_norm = sum((value - sum(x) / 5) ** 2 for value in x) + sum(
                (value - sum(delta) / 5) ** 2 for value in delta
            )
            excess = (
                sum(
                    (x[i] - x[j]) ** 2 / 2 + 1 - mp.cos(theta[j] - theta[i])
                    for i, j in pattern.edges
                )
                - target_energy
            )
            gaps = tuple(
                mp.arg(mp.exp(1j * (theta[(i + 1) % 5] - theta[i]))) for i in range(5)
            )
            assert (
                centered_norm <= _mp(report.norm_squared_upper_bound) < _mp(RADIUS) ** 2
            )
            assert 0 < excess <= _mp(report.excess_storage_upper_bound)
            assert excess < _mp(report.excess_ceiling)
            assert all(abs(gap) < mp.pi / 2 for gap in gaps)
            assert abs(sum(gaps) / (2 * mp.pi) - 1) < mp.mpf("1e-80")


def test_reversible_trapping_does_not_inherit_dissipative_recovery_admission():
    reversible = _pattern(model=REVERSIBLE_MODEL)
    trapped = _identity(reversible)
    no_recovery = _certificate(reversible)
    assert trapped.source_set_trapping_certified
    assert reversible.continuous_loss_bounds == I(0)
    assert not no_recovery.admitted
    assert "positive_epi_weight_required" in no_recovery.hypothesis_failures
    dissipative = _pattern()
    recovery = _certificate(dissipative)
    no_recurrence = _identity(dissipative)
    assert recovery.admitted
    assert not no_recurrence.family_admitted
    assert not no_recurrence.family_almost_everywhere_recurrence_certified
    assert no_recurrence.hypothesis_failures
    # Geometry is shared; its use does not select a different complete law.
    assert trapped.norm_squared_upper_bound == recovery.norm_squared_upper_bound
    assert trapped.barrier_lower_bound == recovery.barrier_lower_bound


def test_identity_origins_oriented_winding_and_declared_lifts_retain_their_scope():
    graph = _graph()
    source = _pattern(graph, model=REVERSIBLE_MODEL)
    forward = _identity(source)
    reverse = _identity(source, cycle=tuple(reversed(range(5))), winding=-1)
    assert reverse.source_set_trapping_certified
    assert reverse.norm_squared_upper_bound == forward.norm_squared_upper_bound
    for node in graph:
        graph.nodes[node]["EPI"] += 1000
        graph.nodes[node]["theta"] -= 3
    translated = _identity(
        _pattern(graph, model=REVERSIBLE_MODEL), phase_turns=(2,) * 5
    )
    assert translated.relative_source_family_membership == "certified_inside"
    assert translated.absolute_source_family_membership == (
        "unavailable_common_form_origin_unobserved"
    )
    assert translated.norm_squared_upper_bound == forward.norm_squared_upper_bound
    assert translated.excess_storage_upper_bound == forward.excess_storage_upper_bound
    # The nominal origin is far outside (-1,1), but the original observation
    # contains arbitrary common offsets. It neither proves nor refutes the
    # separate absolute-mean restriction of the finite-volume family.
    assert min(translated.source.nominal_form) > 999
    graph.nodes[2]["theta"] -= 2 * float(mp.pi)
    wrapped = _pattern(graph, model=REVERSIBLE_MODEL)
    missing = _identity(wrapped)
    lifted = _identity(wrapped, phase_turns=(0, 0, 1, 0, 0))
    assert missing.family_admitted
    assert not missing.source_set_trapping_certified
    assert missing.relative_source_family_membership == "unresolved"
    assert lifted.source_set_trapping_certified


def test_identity_family_ceiling_source_trapping_and_set_membership_are_distinct():
    small_family = _identity(excess_ceiling=Q(1, 2**20))
    assert small_family.family_admitted
    assert small_family.source_set_trapping_certified
    assert small_family.relative_source_family_membership == "unresolved"
    assert small_family.source_family_excess_margin < 0
    large_family = _identity(excess_ceiling=1)
    assert not large_family.family_admitted
    assert not large_family.family_almost_everywhere_recurrence_certified
    assert large_family.source_set_trapping_certified
    assert large_family.family_barrier_margin < 0
    wide_observation = _identity(
        _pattern(model=REVERSIBLE_MODEL, form_errors=(Q(1, 4),) * 5)
    )
    assert wide_observation.family_admitted
    assert wide_observation.family_almost_everywhere_recurrence_certified
    assert not wide_observation.source_set_trapping_certified
    assert wide_observation.relative_source_family_membership == "unresolved"
    # An upper bound failing to fit does not prove that every source state
    # lies outside. Nor does the separate family theorem repair membership.
    assert (
        wide_observation.individual_recurrence_status == "unavailable_for_chosen_state"
    )


def test_exact_consensus_target_and_nonzero_deformation_are_not_the_same_state():
    graph = nx.cycle_graph(5)
    for i in graph:
        graph.nodes[i].update(EPI=7, theta=-3, nu_f=2.0**-i)
    exact = _identity(
        _pattern(
            graph, model=REVERSIBLE_MODEL, form_errors=(0,) * 5, phase_errors=(0,) * 5
        ),
        winding=0,
    )
    assert exact.family_admitted and exact.source_set_trapping_certified
    assert exact.norm_squared_upper_bound == exact.excess_storage_upper_bound == 0
    assert exact.target_phase_turns == (Q(0),) * 5
    graph.nodes[0]["EPI"] += Q(1, 1024)
    perturbed = _identity(
        _pattern(
            graph, model=REVERSIBLE_MODEL, form_errors=(0,) * 5, phase_errors=(0,) * 5
        ),
        winding=0,
    )
    assert perturbed.source_set_trapping_certified
    assert perturbed.norm_squared_upper_bound > 0
    assert perturbed.excess_storage_upper_bound > 0
    assert perturbed.individual_recurrence_status == "unavailable_for_chosen_state"


def test_identity_positive_capacity_and_endpoint_provenance_are_not_optional():
    tiny = Q(1, 2**200)
    slow = _identity(_pattern(_graph(capacity=float(tiny)), model=REVERSIBLE_MODEL))
    assert slow.family_admitted and slow.source_set_trapping_certified
    assert all(bound.lo == 0 for bound in slow.capacity_bounds)
    assert slow.exact_held_capacity == (tiny,) * 5
    frozen = _identity(_pattern(_graph(capacity=0), model=REVERSIBLE_MODEL))
    assert not frozen.family_admitted
    assert "strictly_positive_held_capacity_required" in frozen.hypothesis_failures
    with pytest.raises(ADMISSION_ERRORS):
        _identity(_pattern(_graph(capacity=-1), model=REVERSIBLE_MODEL))
    source = _initial_forecast_report(
        _pattern(model=REVERSIBLE_MODEL, form_errors=(0,) * 5, phase_errors=(0,) * 5)
    )
    endpoint = _identity(source)
    assert endpoint.source_set_trapping_certified
    assert endpoint.observation_time == 2
    assert endpoint.input_forecast_admitted is False
    assert endpoint.input_forecast_requested_end_time == 3
    assert endpoint.absolute_source_family_membership == (
        "unavailable_common_form_origin_unobserved"
    )
    touches_zero = replace(
        source.full_forecast, endpoint=source.full_forecast.endpoint[:-1] + (I(0, 1),)
    )
    rejected = _identity(replace(source, full_forecast=touches_zero))
    assert not rejected.family_admitted
    corrupt = replace(source.full_forecast, endpoint=source.full_forecast.endpoint[:-1])
    with pytest.raises(ValueError, match="must match"):
        _identity(replace(source, full_forecast=corrupt))


@pytest.mark.parametrize(
    "change",
    (
        {"excess_ceiling": 0},
        {"excess_ceiling": True},
        {"excess_ceiling": float("inf")},
        {"form_mean_bounds": (0, 0)},
        {"form_mean_bounds": (2, 1)},
        {"form_mean_bounds": (False, 1)},
        {"form_mean_bounds": (0,)},
        {"form_mean_bounds": {0, 1}},
    ),
)
def test_identity_rejects_invalid_or_zero_volume_family_declarations(change):
    with pytest.raises(ADMISSION_ERRORS):
        _identity(**change)


def test_identity_full_support_chart_and_export_remain_detached(tmp_path):
    graph = _graph()
    source = _pattern(graph, model=REVERSIBLE_MODEL)
    before = pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )
    report = _identity(source)
    assert (
        pickle.dumps(
            (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
            protocol=5,
        )
        == before
    )
    too_wide = _identity(radius=1)
    assert not too_wide.family_admitted
    assert not too_wide.source_set_trapping_certified
    graph.add_edge(0, 2)
    with pytest.raises(ValueError, match="cycle"):
        _identity(_pattern(graph, model=REVERSIBLE_MODEL))
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-cycle-identity.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    path = tmp_path / "reversible-cycle-identity.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
