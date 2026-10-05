"""Complete-state symmetry obstructions, distinct from energy or confinement."""

import math
from dataclasses import replace
from fractions import Fraction as Q
from functools import partial

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.mathematics._validated_taylor import flow_jets
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import certify_sine_prepared_entry
from tnfr.physics.relational_sine_forecast import _sine_flow
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_symmetry import assess_sine_cycle_symmetry
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

MODEL = RelationalExchangeModel(
    1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
)
REFLECTION = (0, 4, 3, 2, 1)
CYCLE = (0, 1, 2, 3, 4)
REFLECTED_FORM = tuple(1023 * value for value in (8, 4, -8, -8, 4))


def _graph(*, form=REFLECTED_FORM, phase=None, capacity=None, graph=None):
    graph = nx.cycle_graph(5) if graph is None else graph.copy()
    size = len(graph)
    phase = (0,) * size if phase is None else phase
    capacity = (1,) * size if capacity is None else capacity
    for node, x, theta, nu in zip(graph, form, phase, capacity):
        graph.nodes[node].update(EPI=x, theta=theta, nu_f=nu)
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _source(*, model=MODEL, **kwargs):
    return bound_relational_sine_exchange(_graph(**kwargs), reference_model=model)


def _assess(source, *, permutation=REFLECTION, cycle=CYCLE):
    return assess_sine_cycle_symmetry(
        source, permutation_indices=permutation, cycle=cycle
    )


def _jets(source):
    positions = {node: index for index, node in enumerate(source.nodes)}
    neighbors = [[] for _ in source.nodes]
    for left, right in source.edges:
        i, j = positions[left], positions[right]
        neighbors[i].append(j)
        neighbors[j].append(i)
    return flow_jets(
        tuple(I(value) for value in (*source.epi, *source.phase, source.capacity[-1])),
        2,
        partial(
            _sine_flow,
            neighbors=tuple(map(tuple, neighbors)),
            visible_capacity=source.capacity[:-1],
            model=source.reference_model,
        ),
    )


def _zero(value):
    assert value.lo <= 0 <= value.hi
    assert value.abs_max < Q(1, 10**25)


@pytest.fixture(scope="module")
def reflected_source():
    return _source()


@pytest.fixture(scope="module")
def acquired_entry():
    # The existing frozen analytic producer is reused once, without a new
    # trajectory, preparation search or adjustment of the observation time.
    source = _source(form=tuple(4092 * (i - 2) for i in range(5)))
    return certify_sine_prepared_entry(
        source, scaled_time=100, edge_turn_offsets=(0, -1, 0, 0, 0)
    )


def test_equal_budget_does_not_select_the_same_organization(
    reflected_source, acquired_entry
):
    report = _assess(reflected_source)
    assert report.status == "certified" and report.unresolved == ()
    assert report.trajectory_symmetry_certified
    assert report.zero_winding_when_nonantipodal
    assert report.nonzero_winding_limit_excluded
    assert report.cycle_orientation_reversed
    assert report.mapped_cycle_chain == tuple(-v for v in report.cycle_chain)
    assert report.initial_form_storage == acquired_entry.initial_form_storage
    for source in (reflected_source, acquired_entry.source):
        # Independent edge sum and weighted conserved means, not stored labels.
        form = dict(zip(source.nodes, source.epi))
        energy = sum((form[j] - form[i]) ** 2 / 2 for i, j in source.edges)
        assert energy == 160 * 1023**2
        assert sum(source.epi) == sum(source.phase) == 0
        assert source.continuous_loss > 0
        assert any(value.abs_max > 0 for value in source.phase_rates)
    assert acquired_entry.admitted
    assert acquired_entry.initial_cycle_periods == (0,)
    assert acquired_entry.capture.cycle_periods == (1,)


def test_full_two_row_field_preserves_reflection_with_heterogeneous_capacities():
    form = tuple(map(Q, (3, -2, 1, 1, -2)))
    phase = tuple(map(Q, (0, 2, -1, -1, 2)))
    capacity = tuple(map(Q, (1, 2, 3, 3, 2)))
    model = RelationalExchangeModel(
        2, epi_weight=Q(3, 4), phase_weight=Q(1, 4), phase_domain="regular"
    )
    source = _source(form=form, phase=phase, capacity=capacity, model=model)
    report, jets = _assess(source), _jets(source)
    assert report.trajectory_symmetry_certified
    e, w = map(Q, model.effective_weights)
    for i, reflected in enumerate(REFLECTION):
        neighbors = ((i - 1) % 5, (i + 1) % 5)
        q = sum((form[i] - form[j] for j in neighbors), Q(0))
        current = sum((sin(I(phase[j] - phase[i])) for j in neighbors), I(0))
        expected_form = capacity[i] / 2 * (-e * q + w / pi_interval() * current)
        expected_phase = (
            capacity[i] * w * q / (2 * Q(model.storage_scale)) / pi_interval()
        )
        _zero(jets[i][1] - expected_form)
        _zero(jets[5 + i][1] - expected_phase)
        for row in (0, 5):
            for order in (0, 1, 2):
                _zero(jets[row + i][order] - jets[row + reflected][order])
    # One fixed edge is synchronized; this does not require all edges acute.
    assert phase[1] - phase[0] > pi_interval().hi / 2
    _zero(jets[7][1] - jets[8][1])


def test_combined_sign_reflection_does_not_impose_zero_winding(acquired_entry):
    source = acquired_entry.source
    reverse = (4, 3, 2, 1, 0)
    report = _assess(source, permutation=reverse)
    assert report.support_automorphism and report.cycle_orientation_reversed
    assert report.phase_lift_preserved and report.capacity_preserved
    assert not report.form_preserved
    assert not report.trajectory_symmetry_certified
    assert not report.zero_winding_when_nonantipodal
    assert not report.nonzero_winding_limit_excluded
    jets = _jets(source)
    for row in (0, 5):
        for i, reflected in enumerate(reverse):
            for order in (0, 1, 2):
                _zero(jets[row + i][order] + jets[row + reflected][order])
    assert acquired_entry.capture.cycle_periods == (1,)


def test_global_sign_reversal_is_a_covariance_of_both_rows():
    form = tuple(map(Q, (1, 4, -2, 3, -1)))
    phase = (Q(0), Q(1, 2), Q(-1, 3), Q(4, 3), Q(-2, 3))
    capacity = (1, 2, 3, 4, 5)
    forward = _jets(_source(form=form, phase=phase, capacity=capacity))
    reverse = _jets(
        _source(
            form=tuple(-v for v in form),
            phase=tuple(-v for v in phase),
            capacity=capacity,
        )
    )
    for i in range(10):
        for order in (0, 1, 2):
            _zero(forward[i][order] + reverse[i][order])


def test_nonacute_and_large_raw_gaps_do_not_invalidate_the_conditional_obstruction():
    phase = (0, 4, -3, -3, 4)
    report = _assess(_source(phase=phase))
    assert report.trajectory_symmetry_certified
    assert report.zero_winding_when_nonantipodal
    assert max(abs(phase[j] - phase[i]) for i, j in nx.cycle_graph(5).edges) > math.pi
    wrapped = [
        (phase[(i + 1) % 5] - phase[i] + math.pi) % (2 * math.pi) - math.pi
        for i in range(5)
    ]
    assert abs(sum(wrapped)) < 1e-14
    # This conclusion only concerns times without antipodal cycle edges.
    assert report.nonzero_winding_limit_excluded


@pytest.mark.parametrize("field", ("epi", "phase", "capacity"))
def test_tiny_exact_asymmetry_is_not_rounded_into_a_fixed_state(
    reflected_source, field
):
    values = list(getattr(reflected_source, field))
    values[1] += Q(1, 10**40)
    report = _assess(replace(reflected_source, **{field: tuple(values)}))
    assert report.support_automorphism and report.cycle_orientation_reversed
    assert not report.trajectory_symmetry_certified
    assert not report.zero_winding_when_nonantipodal
    residuals = getattr(
        report,
        {
            "epi": "form_residuals",
            "phase": "phase_lift_residuals",
            "capacity": "capacity_residuals",
        }[field],
    )
    assert any(value != 0 for value in residuals)


def test_floating_two_pi_is_not_an_exact_circular_symmetry(reflected_source):
    phase = (0, 2 * math.pi, 0, 0, 0)
    report = _assess(replace(reflected_source, phase=phase))
    assert not report.phase_lift_preserved
    assert not report.trajectory_symmetry_certified


def test_orientation_preserving_action_does_not_supply_a_winding_obstruction():
    source = _source(form=(2,) * 5, phase=(Q(1, 3),) * 5)
    report = _assess(source, permutation=(1, 2, 3, 4, 0))
    assert report.trajectory_symmetry_certified
    assert not report.cycle_orientation_reversed
    assert not report.zero_winding_when_nonantipodal
    assert not report.nonzero_winding_limit_excluded


def test_swapping_distinct_cycles_does_not_force_either_period_to_zero():
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edge(0, 5)
    source = _source(graph=graph, form=(0,) * 10)
    permutation = tuple((i + 5) % 10 for i in range(10))
    report = _assess(source, permutation=permutation)
    assert report.trajectory_symmetry_certified
    assert not report.cycle_orientation_reversed
    assert not report.zero_winding_when_nonantipodal


def test_support_symmetry_must_include_the_environment_not_only_the_selected_cycle():
    graph = nx.cycle_graph(5)
    graph.add_edge(1, 5)
    report = _assess(_source(graph=graph, form=(0,) * 6), permutation=(*REFLECTION, 5))
    assert not report.support_automorphism
    assert not report.trajectory_symmetry_certified
    assert not report.zero_winding_when_nonantipodal


def test_missing_mapped_cycle_edges_are_unavailable_not_a_zero_integer_chain():
    source = _source(form=(0,) * 5)
    report = _assess(source, permutation=(0, 2, 1, 3, 4))
    assert not report.support_automorphism
    assert report.mapped_cycle_chain is None
    assert not report.cycle_orientation_reversed
    assert not report.zero_winding_when_nonantipodal


def test_common_origins_relabeling_and_cycle_orientation_keep_the_same_obstruction():
    form = tuple(value + Q(17, 3) for value in REFLECTED_FORM)
    phase = (Q(-11, 7),) * 5
    graph = _graph(form=form, phase=phase)
    labels = {i: f"node-{4 - i}" for i in graph}
    reordered = nx.Graph()
    reordered.graph.update(graph.graph)
    reordered.add_nodes_from((i, graph.nodes[i]) for i in (3, 0, 4, 2, 1))
    reordered.add_edges_from(graph.edges)
    relabeled = nx.relabel_nodes(reordered, labels)
    source = bound_relational_sine_exchange(relabeled, reference_model=MODEL)
    index = {node: i for i, node in enumerate(source.nodes)}
    original = {label: i for i, label in labels.items()}
    permutation = tuple(
        index[labels[REFLECTION[original[node]]]] for node in source.nodes
    )
    cycle = tuple(labels[i] for i in range(5))
    forward = _assess(source, permutation=permutation, cycle=cycle)
    backward = _assess(source, permutation=permutation, cycle=tuple(reversed(cycle)))
    assert forward.zero_winding_when_nonantipodal
    assert backward.zero_winding_when_nonantipodal
    assert forward.cycle_chain == tuple(-v for v in backward.cycle_chain)
    assert forward.initial_form_storage == 160 * 1023**2
    direct = source.assess_cycle_symmetry(permutation_indices=permutation, cycle=cycle)
    assert direct == forward


def test_zero_loss_and_zero_capacity_do_not_destroy_exact_equivariance():
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    source = _source(model=model, capacity=(1, 0, 2, 2, 0))
    report = _assess(source)
    assert report.trajectory_symmetry_certified
    assert report.zero_winding_when_nonantipodal
    assert source.continuous_loss == 0


def test_cached_derived_fields_cannot_supply_or_remove_the_obstruction(
    reflected_source,
):
    altered = replace(
        reflected_source,
        form_storage=Q(-123),
        form_gradient=(Q(99),) * 5,
        form_rates=(I(999),) * 5,
        phase_rates=(I(-999),) * 5,
    )
    report = _assess(altered)
    assert report.trajectory_symmetry_certified
    assert report.initial_form_storage == 160 * 1023**2


def test_an_uncertain_set_fixed_as_a_set_does_not_fix_every_member():
    pattern = bound_relational_sine_pattern(
        _graph(),
        reference_node=0,
        reference_model=MODEL,
        form_error_bounds=(Q(1, 16),) * 5,
        phase_error_bounds=(Q(1, 65536),) * 5,
    )
    with pytest.raises((TypeError, ValueError)):
        _assess(pattern)


@pytest.mark.parametrize(
    "permutation,cycle",
    (
        ((0, 1, 2, 3, 3), CYCLE),
        ((0, 1, 2, 3), CYCLE),
        ((0, True, 2, 3, 4), CYCLE),
        ((0, 1, 2, 3, 9), CYCLE),
        (REFLECTION, (0, 1)),
        (REFLECTION, (0, 1, 2, 3, 4, 0)),
        (REFLECTION, (0, 2, 1, 3, 4)),
        (REFLECTION, (0, 1, 2, 3, "unknown")),
    ),
)
def test_malformed_permutations_and_cycles_reject(reflected_source, permutation, cycle):
    with pytest.raises((TypeError, ValueError)):
        _assess(reflected_source, permutation=permutation, cycle=cycle)


def test_source_law_and_primitive_domains_are_revalidated(reflected_source):
    invalid = (
        replace(reflected_source, law="native_arg"),
        replace(reflected_source, reference_model=RelationalExchangeModel(1)),
        replace(reflected_source, capacity=(-1, 1, 1, 1, 1)),
        replace(reflected_source, epi=(True,) * 5),
        replace(reflected_source, phase=(float("nan"),) * 5),
        replace(reflected_source, degrees=(1,) * 5),
        replace(reflected_source, edges=((0, 1), (1, 0))),
    )
    for source in invalid:
        with pytest.raises((TypeError, ValueError)):
            _assess(source)


def test_sdk_keeps_exact_source_and_conditional_winding_scope(
    reflected_source, tmp_path
):
    report = _assess(reflected_source)
    direct = report.to_dict()
    generic = relational_report_to_dict(report)
    assert generic["schema"] == "tnfr.relational-report.v1"
    assert generic["report_type"] == "SineCycleSymmetryAssessment"
    assert generic["report"] == direct["report"]
    path = tmp_path / "symmetry.json"
    export_to_json(report, path)
    assert json_loads(path.read_text(encoding="utf-8")) == direct
    assert direct["report"]["initial_form_storage"] == {
        "numerator": 160 * 1023**2,
        "denominator": 1,
    }
    assert direct["report"]["zero_winding_when_nonantipodal"] is True
    assert "acute_trapping_certified" not in direct["report"]
