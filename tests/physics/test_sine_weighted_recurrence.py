"""Independent compact-family bounds; no selected trajectory is evaluated."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_support import (
    SineSupportExchangeComparison,
    bound_relational_sine_support_exchange,
)
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

FORMS = (Q(-1, 4), Q(1, 8), Q(1, 2))
PHASES = (Q(0), Q(1, 4), Q(-1, 8))


def _source(*, weights=(2, 3), capacities=(2, 3, 5), zero_chord=False):
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    for node, (form, phase, capacity) in enumerate(zip(FORMS, PHASES, capacities)):
        graph.add_node(node, EPI=float(form), theta=float(phase), nu_f=float(capacity))
    graph.add_edge(0, 1, weight=weights[0])
    graph.add_edge(1, 2, weight=weights[1])
    if zero_chord:
        graph.add_edge(0, 2, weight=0)
    return bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            2, epi_weight=0, phase_domain="regular"
        ),
    )


@pytest.fixture(scope="module")
def source():
    return _source()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _report(source, **changes):
    arguments = {"energy_ceiling": Q(3, 2), "form_mean_bounds": (Q(-1, 2), Q(3, 4))}
    arguments.update(changes)
    return source.recurrence_bounds(**arguments)


@pytest.mark.parametrize("exchange", (Q(1), Q(3, 2)))
def test_weighted_path_compactness_and_full_field_invariants(source, mp, exchange):
    model = copy(source.reference_model)
    object.__setattr__(model, "phase_weight", exchange)
    report = _report(replace(source, reference_model=model))
    assert report.form_mean_weights == (Q(1, 2), Q(2, 3), Q(1, 5))
    assert report.normalized_mean_weights == (Q(15, 41), Q(20, 41), Q(6, 41))
    assert report.weighted_form_mean == Q(7, 164)
    assert report.positive_conductance_edge_indices == (0, 1)
    assert report.positive_conductance_weight_floor == 2
    assert report.path_length_upper_bound == 2
    _contains(mp, report.form_radius_bounds, mp.sqrt(3))
    assert report.form_coordinate_bounds == I(
        Q(-1, 2) - report.form_radius_bounds.hi,
        Q(3, 4) + report.form_radius_bounds.hi,
    )
    # Independent complete rows: bare sine currents, weighted form gradient,
    # actual bare degrees and beta=2. Their cross work cancels pointwise.
    gradient = (Q(-3, 4), Q(-3, 8), Q(9, 8))
    currents = (
        mp.sin(mp.mpf(1) / 4),
        -mp.sin(mp.mpf(1) / 4) - mp.sin(mp.mpf(3) / 8),
        mp.sin(mp.mpf(3) / 8),
    )
    mobility = (Q(2), Q(3, 2), Q(5))
    form_rates = tuple(
        _mp(mp, exchange * factor) * current / mp.pi
        for factor, current in zip(mobility, currents)
    )
    phase_rates = tuple(
        _mp(mp, exchange * factor * value / 2) / mp.pi
        for factor, value in zip(mobility, gradient)
    )
    for intervals, values in (
        (report.source.form_rates, form_rates),
        (report.source.phase_rates, phase_rates),
    ):
        for interval, value in zip(intervals, values):
            _contains(mp, interval, value)
    mean_rate = sum(
        _mp(mp, weight) * rate
        for weight, rate in zip(report.normalized_mean_weights, form_rates)
    )
    work = sum(
        _mp(mp, q) * dx - 2 * current * dtheta
        for q, current, dx, dtheta in zip(gradient, currents, form_rates, phase_rates)
    )
    assert abs(mean_rate) < mp.mpf("1e-140")
    assert abs(work) < mp.mpf("1e-140")
    assert report.weighted_mean_rate_residual_bounds.lo <= 0
    assert report.weighted_mean_rate_residual_bounds.hi >= 0
    assert report.divergence == 0  # Both rows omit their own differentiated coordinate.
    expected_storage = Q(45, 128) + 2 * (
        2 - mp.cos(mp.mpf(1) / 4) - mp.cos(mp.mpf(3) / 8)
    )
    _contains(mp, report.captured_storage_bounds, expected_storage)
    assert report.snapshot_family_membership == "inside"
    assert report.captured_energy_exact is None
    assert report.almost_everywhere_recurrence_certified
    assert report.finite_positive_family_volume_certified
    assert report.individual_recurrence_status == "unavailable_for_chosen_state"
    assert report.clock == "t"
    with pytest.raises(FrozenInstanceError):
        report.energy_ceiling = Q(9)


def test_zero_weight_chord_changes_bare_mean_but_not_positive_path_requirement(source):
    report = _report(_source(zero_chord=True))
    assert report.source.degrees == (2, 2, 2)
    assert report.form_mean_weights == (Q(1), Q(2, 3), Q(2, 5))
    assert report.weighted_form_mean != _report(source).weighted_form_mean
    positive = tuple(
        report.source.edges[i] for i in report.positive_conductance_edge_indices
    )
    assert positive == ((0, 1), (1, 2))
    assert report.positive_conductance_weight_floor == 2
    assert report.source.pressure[0] != source.pressure[0]


@pytest.mark.parametrize("weights", ((0, 2), (2, 0), (0, 0)))
def test_connected_bare_support_cannot_replace_positive_conductance_connectivity(
    weights,
):
    with pytest.raises(ValueError, match="connected positive-conductance"):
        _report(_source(weights=weights))


def test_zero_weight_pair_has_an_exact_unbounded_affine_flow_despite_conserved_mean(mp):
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    graph.add_node(0, EPI=0.0, theta=0.0, nu_f=2.0)
    graph.add_node(1, EPI=0.0, theta=0.5, nu_f=3.0)
    graph.add_edge(0, 1, weight=0)
    source = bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_domain="regular"
        ),
    )
    # W=0 makes both phases constant. For every real t the exact solution is
    # x(t)=t*(2*sin(1/2)/pi,-3*sin(1/2)/pi). It has constant H and mean,
    # yet its forms are unbounded. This is an analytic solution, not a run.
    speed = mp.sin(mp.mpf(1) / 2) / mp.pi
    assert speed > 0
    for enclosure, value in zip(source.form_rates, (2 * speed, -3 * speed)):
        _contains(mp, enclosure, value)
    assert source.phase_rates == (I(0), I(0))
    assert Q(1, 2) * 2 + Q(1, 3) * -3 == 0
    _contains(mp, source.storage, 1 - mp.cos(mp.mpf(1) / 2))
    with pytest.raises(ValueError, match="connected positive-conductance"):
        _report(source)


def test_exact_boundary_and_outside_membership_do_not_change_family_theorem(source):
    equal_phases = replace(source, phase=(Q(7, 8),) * 3)
    boundary = _report(equal_phases, energy_ceiling=Q(45, 128))
    assert boundary.captured_energy_exact == Q(45, 128)
    assert boundary.captured_storage_bounds == I(Q(45, 128))
    assert boundary.snapshot_family_membership == "inside"
    assert boundary.snapshot_membership_reasons == ()
    for report, reason in (
        (
            _report(equal_phases, energy_ceiling=Q(45, 128) - Q(1, 2**80)),
            "exact_energy_exceeds_declared_ceiling",
        ),
        (
            _report(source, form_mean_bounds=(1, 2)),
            "weighted_mean_outside_declared_slab",
        ),
        (
            _report(source, energy_ceiling=Q(1, 100)),
            "energy_lower_bound_exceeds_declared_ceiling",
        ),
    ):
        assert report.snapshot_family_membership == "outside"
        assert report.snapshot_membership_reasons == (reason,)
        assert report.almost_everywhere_recurrence_certified
        assert report.individual_recurrence_status == "unavailable_for_chosen_state"


def test_unresolved_trigonometric_membership_remains_unresolved(source):
    midpoint = (source.storage.lo + source.storage.hi) / 2
    assert source.storage.lo < midpoint < source.storage.hi
    report = _report(source, energy_ceiling=midpoint)
    assert report.snapshot_family_membership == "unresolved"
    assert report.snapshot_membership_reasons == (
        "energy_ceiling_membership_not_resolved",
    )
    assert report.almost_everywhere_recurrence_certified


def test_form_translation_and_arbitrary_lifts_do_not_change_circular_family_bound(
    source,
):
    original = _report(source)
    shifted = replace(
        source,
        epi=tuple(value + 7 for value in source.epi),
        phase=tuple(value + Q(10**50) for value in source.phase),
    )
    report = _report(shifted, form_mean_bounds=(Q(13, 2), Q(31, 4)))
    assert report.weighted_form_mean == original.weighted_form_mean + 7
    assert report.form_radius_bounds == original.form_radius_bounds
    assert report.form_coordinate_bounds == original.form_coordinate_bounds + 7
    assert report.captured_storage_bounds == original.captured_storage_bounds
    assert report.snapshot_family_membership == original.snapshot_family_membership
    equilibrium = _report(replace(source, epi=(Q(0),) * 3, phase=(Q(0),) * 3))
    assert equilibrium.snapshot_family_membership == "inside"
    assert equilibrium.individual_recurrence_status == "unavailable_for_chosen_state"


@pytest.mark.parametrize(
    "changes",
    (
        {"energy_ceiling": 0},
        {"energy_ceiling": -1},
        {"energy_ceiling": True},
        {"energy_ceiling": float("inf")},
        {"energy_ceiling": float("nan")},
        {"form_mean_bounds": (0, 0)},
        {"form_mean_bounds": (1, -1)},
        {"form_mean_bounds": (False, 1)},
        {"form_mean_bounds": (0, float("inf"))},
        {"form_mean_bounds": (0,)},
        {"form_mean_bounds": (0, 1, 2)},
        {"form_mean_bounds": {0, 1}},
    ),
)
def test_family_requires_finite_positive_energy_and_ordered_positive_width_mean(
    source, changes
):
    with pytest.raises((TypeError, ValueError)):
        _report(source, **changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"capacity": (0, 3, 5)},
        {"capacity": (True, 3, 5)},
        {"phase": (False, Q(1, 4), Q(-1, 8))},
        {"epi": (float("nan"), Q(1, 8), Q(1, 2))},
        {"conductances": (Q(-1), Q(3))},
        {"law": "different_law"},
        {
            "reference_model": RelationalExchangeModel(
                2, epi_weight=1, phase_domain="regular"
            )
        },
    ),
)
def test_invalid_source_primitives_cannot_use_cached_retention_values(source, changes):
    with pytest.raises((TypeError, ValueError)):
        _report(replace(source, storage=I(0), degrees=(99,) * 3, **changes))


def test_fresh_direct_and_sdk_export_rebuild_every_consumed_field(source, tmp_path):
    report = _report(source)
    poisoned_source = replace(
        source,
        degrees=(99,) * 3,
        form_storage=Q(0),
        storage=I(0),
        form_gradient=(Q(0),) * 3,
        form_rates=(I(0),) * 3,
    )
    assert _report(poisoned_source) == report
    poisoned = replace(
        report,
        source=poisoned_source,
        weighted_form_mean=Q(100),
        positive_conductance_weight_floor=Q(99),
        form_radius_bounds=I(0),
        captured_storage_bounds=I(999),
        snapshot_family_membership="outside",
        divergence=Q(99),
        almost_everywhere_recurrence_certified=False,
        individual_recurrence_status="certified_recurrent",
    )
    assert poisoned.to_dict() == report.to_dict()
    generic = relational_report_to_dict(poisoned)
    assert generic["report"] == report.to_dict()["report"]
    path = tmp_path / "recurrence.json"
    export_to_json(generic, path)
    assert json_loads(path.read_text(encoding="utf-8")) == generic
    original = path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        export_to_json(
            relational_report_to_dict(replace(poisoned, energy_ceiling=False)), path
        )
    assert path.read_bytes() == original


def test_untrusted_source_and_report_do_not_dispatch_custom_methods(source):
    class Fake:
        def recurrence_bounds(self, **kwargs):
            raise AssertionError("untrusted method dispatch")

    report = _report(source)
    with pytest.raises(TypeError):
        SineSupportExchangeComparison.recurrence_bounds(
            Fake(), energy_ceiling=1, form_mean_bounds=(-1, 1)
        )
    with pytest.raises(TypeError):
        replace(report, source=Fake()).to_dict()
    with pytest.raises(TypeError):
        type(report).to_dict(Fake())
