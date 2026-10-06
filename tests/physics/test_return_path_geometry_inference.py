"""Inverse-domain, correlation and admission contracts for return-path geometry."""

from dataclasses import dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics import phase_cycle_geometry as owner
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

LEFT, RIGHT = tuple(range(5)), tuple(range(5, 10))


def _source(*, reverse=False, label=lambda i: i):
    graph = nx.Graph()
    graph.add_nodes_from(
        label(i) for i in (reversed(range(11)) if reverse else range(11))
    )
    graph.add_edges_from(
        (label(i), label(j))
        for i, j in (
            [(offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)]
            + [(0, 10), (10, 5), (1, 6)]
        )
    )
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


@pytest.fixture(scope="module")
def source():
    return _source()


def _assess(source, **changes):
    arguments = dict(
        left_cycle=LEFT,
        right_cycle=RIGHT,
        mediator=10,
        special_turn_bounds=(Q(1, 8), Q(1, 8)),
        form_direction=tuple(Q(i == 3) for i in range(11)),
        observation_origin="supplied_mathematical_interval",
    )
    arguments.update(changes)
    return owner.assess_return_path_geometry_response(source, **arguments)


@pytest.fixture(scope="module")
def report(source):
    return _assess(source)


def test_single_exact_geometric_coordinate_infers_inverse_with_outward_uncertainty(
    report,
):
    # This point is a different mathematical input from the frozen reserved
    # epsilon=1 control. Its inverse is checked without a production root.
    with mp.workdps(80):
        angle = mp.pi / 4
        a, b, c = mp.cos(angle / 4), mp.sin(angle), mp.sin(2 * angle / 3)
        epsilon = -(a - b - c) / (a**3 - b**3 - c**3)
        lower = (
            mp.mpf(report.coefficient_lower.numerator)
            / report.coefficient_lower.denominator
        )
        upper = (
            mp.mpf(report.coefficient_upper.numerator)
            / report.coefficient_upper.denominator
        )
        assert 0 < lower < epsilon < upper
        assert upper - lower < mp.mpf("1e-30")
    assert report.coefficient_status == "bounded"
    assert report.clock == "tau=t/pi"
    assert report.response_status == "bounded_initial_tangent_acceleration"
    assert report.special_turn_bounds == (Q(1, 8), Q(1, 8))
    assert all(value.lo > 0 for value in report.edge_curvature_bounds)


def test_observation_interval_maps_monotonically_and_keeps_shared_affine_coordinates(
    source, report
):
    wider = _assess(source, special_turn_bounds=(Q(124, 1000), Q(126, 1000)))
    assert wider.coefficient_lower < report.coefficient_lower
    assert wider.coefficient_upper > report.coefficient_upper
    assert wider.named_cycle_periods == (1, -1, 0)
    for (i, j), (constant, slope), offset in zip(
        wider.geometry.edges,
        wider.edge_turn_affine_coefficients,
        wider.edge_integer_offsets,
    ):
        ai, bi = wider.nodal_turn_affine_coefficients[i]
        aj, bj = wider.nodal_turn_affine_coefficients[j]
        assert (aj - ai - offset, bj - bi) == (constant, slope)
    assert (
        "outer_parameter_boxes_enclose_candidates_but_do_not_define_independent_equilibria"
        in wider.scope
    )


def test_lower_observation_before_sine_root_clips_only_coefficient_lower(source):
    report = _assess(source, special_turn_bounds=(0, Q(1, 8)))
    assert report.coefficient_status == "bounded"
    assert report.coefficient_lower == 0 < report.coefficient_upper
    assert report.geometric_outer_turn_bounds == (Q(1, 12), Q(1, 8))
    assert report.endpoint_classifications == (
        "below_sine_root",
        "finite_positive_coefficient",
    )


@pytest.mark.parametrize("bounds", [(Q(1, 8), Q(1, 5)), (0, 1)])
def test_observation_reaching_cubic_limit_has_no_finite_response_bound(source, bounds):
    report = _assess(source, special_turn_bounds=bounds)
    assert report.coefficient_status == "unbounded_above"
    assert report.coefficient_lower is not None
    assert report.coefficient_upper is None
    assert report.response_acceleration_bounds is report.edge_curvature_bounds is None
    assert report.response_status == "unavailable"


@pytest.mark.parametrize("bounds", [(-1, 0), (0, Q(1, 12)), (Q(1, 5), 1), (1, 2)])
def test_disjoint_geometric_intervals_are_incompatible_not_zero_coefficient(
    source, bounds
):
    report = _assess(source, special_turn_bounds=bounds)
    assert report.coefficient_status == "incompatible"
    assert report.coefficient_lower is report.coefficient_upper is None
    assert report.geometric_outer_turn_bounds is None
    assert report.response_acceleration_bounds is None


def test_unresolved_outward_sign_abstains_instead_of_fitting(source, monkeypatch):
    monkeypatch.setattr(
        owner, "_return_path_storage_terms", lambda turns: (I(-1, 1), I(1, 2))
    )
    report = _assess(source)
    assert report.coefficient_status == "unresolved_interval_arithmetic"
    assert report.coefficient_lower is report.coefficient_upper is None
    assert report.response_acceleration_bounds is None
    assert report.endpoint_linear_residual_bounds == (I(-1, 1), I(-1, 1))


def test_inverse_reader_does_not_run_a_forward_coefficient_root(source, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("inverse admission must not generate its observation")

    monkeypatch.setattr(owner, "_enclose_decreasing_phase_root", forbidden)
    assert _assess(source).coefficient_status == "bounded"


def test_uniform_form_direction_is_exactly_in_the_nodal_nullspace(source):
    report = _assess(source, form_direction=(Q(7, 3),) * 11)
    assert report.phase_velocity_direction == (0,) * 11
    assert report.response_acceleration_bounds == (I(0),) * 11


def test_signed_form_direction_changes_the_signed_response_without_changing_inference(
    source, report
):
    negative = _assess(
        source, form_direction=tuple(-value for value in report.form_direction)
    )
    assert negative.coefficient_lower == report.coefficient_lower
    assert negative.coefficient_upper == report.coefficient_upper
    assert negative.phase_velocity_direction == tuple(
        -value for value in report.phase_velocity_direction
    )
    assert negative.response_acceleration_bounds == tuple(
        -value for value in report.response_acceleration_bounds
    )


def test_node_order_and_scalar_labels_do_not_select_the_response(source, report):
    reverse = _source(reverse=True, label=lambda i: f"node-{i}")
    mapped = _assess(
        reverse,
        left_cycle=tuple(f"node-{i}" for i in LEFT),
        right_cycle=tuple(f"node-{i}" for i in RIGHT),
        mediator="node-10",
        form_direction=tuple(Q(node == "node-3") for node in reverse.nodes),
    )
    by_label = dict(zip(mapped.source.nodes, mapped.response_acceleration_bounds))
    assert (
        tuple(by_label[f"node-{i}"] for i in source.nodes)
        == report.response_acceleration_bounds
    )
    assert mapped.coefficient_lower == report.coefficient_lower


def test_source_cached_response_and_snapshot_are_not_inverse_evidence(source, report):
    changed = replace(
        source,
        epi=tuple(float(i) for i in range(11)),
        phase=(Q(1, 7),) * 11,
        form_rates=(I(999),) * 11,
        phase_rates=(I(999),) * 11,
        storage=I(999),
    )
    rebuilt = _assess(changed)
    assert rebuilt.source.epi == tuple(map(Q, range(11)))
    assert rebuilt.target_epi == (0,) * 11
    assert rebuilt.coefficient_lower == report.coefficient_lower
    assert rebuilt.response_acceleration_bounds == report.response_acceleration_bounds


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), "0.125", 1j])
def test_invalid_geometric_scalars_are_not_coerced(source, value):
    with pytest.raises((ValueError, TypeError)):
        _assess(source, special_turn_bounds=(value, Q(1, 8)))


@pytest.mark.parametrize("values", [(Q(1, 8),), (1, 0), {0, 1}, "01"])
def test_geometry_requires_two_ordered_endpoints(source, values):
    with pytest.raises((ValueError, TypeError)):
        _assess(source, special_turn_bounds=values)


def test_represented_float_endpoints_retain_exact_binary_value(source):
    report = _assess(source, special_turn_bounds=(0.124, 0.126))
    assert report.special_turn_bounds == tuple(
        Q.from_float(value) for value in (0.124, 0.126)
    )


@pytest.mark.parametrize("origin", [None, True, "derived_from_response"])
def test_observation_origin_requires_an_explicit_available_kind(source, origin):
    with pytest.raises(ValueError, match="observation_origin"):
        _assess(source, observation_origin=origin)


def test_measured_origin_is_a_declaration_and_does_not_change_inference(source, report):
    measured = _assess(source, observation_origin="measured_interval")
    assert measured.coefficient_lower == report.coefficient_lower
    assert (
        "observation_origin_is_a_declaration_not_provenance_authentication"
        in measured.scope
    )


@pytest.mark.parametrize(
    "direction", [(0,) * 10, (True,) + (0,) * 10, (float("nan"),) + (0,) * 10]
)
def test_invalid_perturbation_is_rejected_even_if_geometry_is_incompatible(
    source, direction
):
    with pytest.raises((ValueError, TypeError)):
        _assess(source, special_turn_bounds=(0, Q(1, 12)), form_direction=direction)


def test_invalid_source_primitives_and_changed_model_cannot_inherit_the_inverse(source):
    with pytest.raises((ValueError, TypeError)):
        _assess(replace(source, capacity=(True,) + source.capacity[1:]))
    with pytest.raises(ValueError, match="unit capacity"):
        _assess(replace(source, capacity=(2,) + source.capacity[1:]))
    with pytest.raises((ValueError, TypeError)):
        _assess(replace(source, epi=(float("nan"),) + source.epi[1:]))
    with pytest.raises(ValueError, match="zero loss"):
        _assess(
            replace(
                source,
                reference_model=RelationalExchangeModel(
                    1, epi_weight=1, phase_weight=1, phase_domain="regular"
                ),
            )
        )


def test_detached_export_retains_availability_and_validates_all_labels(source, report):
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.return-path-geometry-response.v1"
    payload["report"]["form_direction"][3]["numerator"] = 999
    assert report.form_direction[3] == 1
    unbounded = _assess(source, special_turn_bounds=(0, 1)).to_dict()["report"]
    assert (
        unbounded["coefficient_upper"]
        is unbounded["response_acceleration_bounds"]
        is None
    )

    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    with pytest.raises(TypeError, match="JSON scalar node labels"):
        replace(report, mediator=OpaqueLabel(10)).to_dict()
