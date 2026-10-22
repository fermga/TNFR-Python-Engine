"""Static geometry and admission controls for the conditional two-port probe.

Static controls evaluate no response. Post-freeze interface controls replace
the target producer with an explicit mock and check conditional arithmetic;
they acquire no scientific target or trajectory. Retained acquisition evidence
has a separate owner and is never regenerated.
"""

from fractions import Fraction as Q
from inspect import signature
from types import SimpleNamespace
from typing import get_type_hints

import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_two_port_probe as owner


@pytest.fixture(scope="module")
def geometry():
    return owner._probe_geometry()


def _independent_support():
    graph = nx.disjoint_union(nx.cycle_graph(9), nx.cycle_graph(9))
    graph.add_edges_from(((0, 9), (1, 10)))
    return graph


def test_joined_and_disconnected_supports_retain_their_own_degree_weights(geometry):
    graph = _independent_support()
    edges = tuple(sorted(tuple(sorted(edge)) for edge in graph.edges))
    degrees = tuple(graph.degree[i] for i in range(18))
    assert geometry["geometry"].nodes == tuple(range(18))
    assert geometry["geometry"].edges == edges
    assert geometry["degrees"] == degrees
    assert sum(degrees[:9]) == sum(degrees[9:]) == 20
    assert geometry["receiver_weights"] == tuple(
        Q(degrees[i], 20) if i >= 9 else Q(0) for i in range(18)
    )
    graph.remove_edges_from(((0, 9), (1, 10)))
    assert nx.number_connected_components(graph) == 2
    assert geometry["disconnected_edges"] == tuple(
        sorted(tuple(sorted(edge)) for edge in graph.edges)
    )
    assert geometry["disconnected_degrees"] == tuple(graph.degree[i] for i in graph)
    assert geometry["disconnected_receiver_weights"] == tuple(
        Q(1, 9) if i >= 9 else Q(0) for i in range(18)
    )
    assert geometry["donor_mask"] == tuple(Q(i < 9) for i in range(18))


def test_normalized_heat_kernel_and_exact_dual_norm_constants(geometry):
    graph = _independent_support()
    degree = tuple(graph.degree[i] for i in graph)
    laplacian = tuple(
        tuple(Q(degree[i] if i == j else -int(graph.has_edge(i, j))) for j in graph)
        for i in graph
    )
    normalized = tuple(
        tuple(value / degree[i] for value in row) for i, row in enumerate(laplacian)
    )
    transition = tuple(
        tuple(Q(int(i == j)) - value for j, value in enumerate(row))
        for i, row in enumerate(normalized)
    )
    assert geometry["laplacian"] == laplacian
    assert geometry["normalized_laplacian"] == normalized
    assert all(min(row) >= 0 and sum(row) == 1 for row in transition)
    for i in graph:
        for j in graph:
            assert degree[i] * transition[i][j] == degree[j] * transition[j][i]
            assert geometry["normalized_gap_slack_matrix"][i][j] == (
                laplacian[i][j]
                - Q(1, 90) * (degree[i] * int(i == j) - Q(degree[i] * degree[j], 40))
            )
            assert geometry["normalized_upper_slack_matrix"][i][j] == (
                2 * degree[i] * int(i == j) - laplacian[i][j]
            )
    donor, receiver = geometry["donor_mask"], geometry["receiver_weights"]
    heat_first_coefficient = sum(
        receiver[i] * transition[i][j] * donor[j] for i in graph for j in graph
    )
    assert heat_first_coefficient == Q(1, 10)
    centered = tuple(value - Q(1, 2) for value in donor)
    assert sum(d * value**2 for d, value in zip(degree, centered)) == 10
    assert Q(19, 6) ** 2 > 10
    functional = tuple(
        sum(receiver[i] * normalized[i][j] for i in graph) for j in graph
    )
    norm_squared = sum(value**2 / d for value, d in zip(functional, degree))
    assert norm_squared == Q(1, 300) < Q(7, 120) ** 2
    centered_receiver = tuple(w - Q(d, 40) for w, d in zip(receiver, degree))
    receiver_norm_squared = sum(
        value**2 / d for value, d in zip(centered_receiver, degree)
    )
    assert receiver_norm_squared == Q(1, 40)
    assert 4 * receiver_norm_squared < Q(1, 3) ** 2


def test_exact_supplied_work_and_component_means_are_not_reset(geometry):
    # A deliberately nonzero common mean and unrelated signed residuals.
    form = tuple(Q(17, 5) + Q((i * 7) % 11 - 5, 200) for i in range(18))
    amplitude = Q(3, 100)
    donor = geometry["donor_mask"]
    after = tuple(value + amplitude * mask for value, mask in zip(form, donor))

    def kinetic(values, edges):
        return sum((values[i] - values[j]) ** 2 / 2 for i, j in edges)

    edges = geometry["geometry"].edges
    delta = kinetic(after, edges) - kinetic(form, edges)
    independent_cross = amplitude * (form[0] - form[9] + form[1] - form[10])
    assert delta == amplitude**2 + independent_cross
    degrees = geometry["degrees"]
    mean = sum(d * value for d, value in zip(degrees, form)) / 40
    after_mean = sum(d * value for d, value in zip(degrees, after)) / 40
    assert after_mean - mean == amplitude / 2
    residual_norm_squared = sum(
        d * (value - mean) ** 2 for d, value in zip(degrees, form)
    )
    assert independent_cross**2 <= Q(4, 3) * amplitude**2 * residual_norm_squared
    assert Q(4, 3) < Q(7, 6) ** 2
    assert kinetic(after, geometry["disconnected_edges"]) == kinetic(
        form, geometry["disconnected_edges"]
    )
    assert sum(after[:9], Q(0)) / 9 - sum(form[:9], Q(0)) / 9 == amplitude
    assert sum(after[9:], Q(0)) == sum(form[9:], Q(0))


def test_disconnected_receiver_readout_annihilates_every_edge_channel(geometry):
    # Each pressure edge cancels after the control's actual degree weighting,
    # independently of its current or the values of form and phase.
    weights = geometry["disconnected_receiver_weights"]
    degrees = geometry["disconnected_degrees"]
    for i, j in geometry["disconnected_edges"]:
        assert weights[i] / degrees[i] - weights[j] / degrees[j] == 0
        assert geometry["donor_mask"][i] - geometry["donor_mask"][j] == 0
    assert any(
        geometry["receiver_weights"][i] / geometry["degrees"][i]
        != geometry["receiver_weights"][j] / geometry["degrees"][j]
        for i, j in geometry["geometry"].edges
    )


def _arguments():
    # Nonreserved dummy primitives for rejection/interface controls only.
    return dict(
        form_radius=Q(0),
        phase_radius=Q(0),
        pulse_amplitude=Q(1, 16),
        probe_duration=Q(1, 2),
        readout_error_bound=Q(0),
        contrast_threshold=Q(0),
        work_allowance=Q(1),
    )


@pytest.mark.parametrize("field", tuple(_arguments()))
@pytest.mark.parametrize("value", (True, np.bool_(False), float("nan"), float("inf")))
def test_invalid_scalar_rejects_before_geometry_or_target(field, value, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached geometry or a scientific producer")

    monkeypatch.setattr(owner, "_probe_geometry", forbidden)
    monkeypatch.setattr(owner, "assess_sine_two_port_compatibility", forbidden)
    arguments = _arguments()
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        owner.assess_sine_two_port_probe(**arguments)


@pytest.mark.parametrize(
    "field,value",
    tuple((name, -Q(1, 9)) for name in _arguments())
    + (
        ("pulse_amplitude", 0),
        ("probe_duration", 0),
        ("probe_duration", Q(1001, 1000)),
        ("pulse_amplitude", np.longdouble("1e-400")),
    ),
)
def test_invalid_domain_or_lost_nonzero_scalar_rejects_before_target(
    field, value, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached a scientific producer")

    monkeypatch.setattr(owner, "_probe_geometry", forbidden)
    monkeypatch.setattr(owner, "assess_sine_two_port_compatibility", forbidden)
    arguments = _arguments()
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        owner.assess_sine_two_port_probe(**arguments)


@pytest.mark.parametrize(
    "field,value",
    (
        ("pulse_amplitude", Q(1, 10**500)),
        ("form_radius", Q(10**500)),
        ("probe_duration", 1),
    ),
)
def test_exact_rational_admission_preserves_extreme_finite_values_without_assessment(
    field, value, monkeypatch
):
    class GeometryReached(Exception):
        pass

    def stop_before_assessment():
        raise GeometryReached

    monkeypatch.setattr(owner, "_probe_geometry", stop_before_assessment)
    arguments = _arguments()
    arguments[field] = value
    with pytest.raises(GeometryReached):
        owner.assess_sine_two_port_probe(**arguments)


def test_api_requires_every_primitive_and_accepts_no_source_report():
    parameters = signature(owner.assess_sine_two_port_probe).parameters
    assert tuple(parameters) == tuple(_arguments())
    assert all(value.default is value.empty for value in parameters.values())
    assert all(value.kind is value.KEYWORD_ONLY for value in parameters.values())
    assert (
        get_type_hints(owner.SineTwoPortProbe)["target"]
        is owner.SineTwoPortCompatibility
    )
    assert not any("acquisition" in name or "source" in name for name in parameters)


@pytest.fixture
def mock_target(monkeypatch):
    """A declared interface fixture, not new equilibrium evidence."""
    target = SimpleNamespace(
        local_attraction_certified=True,
        acute_margin_turns_bounds=I(Q(1, 32)),
    )
    calls = []

    def supplied_mock(**kwargs):
        calls.append(kwargs)
        return target

    monkeypatch.setattr(owner, "assess_sine_two_port_compatibility", supplied_mock)
    return target, calls


def _controlled_arguments():
    return dict(
        form_radius=Q(0),
        phase_radius=Q(0),
        pulse_amplitude=Q(1, 4096),
        probe_duration=Q(1, 3),
        readout_error_bound=Q(0),
        contrast_threshold=Q(0),
        work_allowance=Q(1),
    )


def test_mocked_target_is_freshly_requested_at_the_declared_fixed_budget(mock_target):
    target, calls = mock_target
    report = owner.assess_sine_two_port_probe(**_controlled_arguments())
    assert report.target is target
    assert calls == [dict(classes=(2, 1), outer_refinements=32, inner_refinements=64)]
    assert report.target_admitted
    assert report.status == "certified_probe"
    assert report.response_certified and report.work_certified
    assert report.joined_identity_certified and report.recovery_certified
    assert report.unavailable_reasons == ()
    assert report.joined_form_mean_increment == report.pulse_amplitude / 2
    assert report.disconnected_form_mean_increments == (report.pulse_amplitude, 0)


def test_response_threshold_equality_abstains_without_discarding_other_gates(
    mock_target,
):
    arguments = _controlled_arguments()
    baseline = owner.assess_sine_two_port_probe(**arguments)
    arguments["contrast_threshold"] = baseline.recorded_contrast_bounds[0]
    report = owner.assess_sine_two_port_probe(**arguments)
    assert report.response_margin == 0
    assert not report.response_certified and report.status == "unavailable"
    assert report.work_certified and report.joined_identity_certified
    assert report.recovery_certified
    assert report.unavailable_reasons == (
        "recorded_contrast_not_strictly_above_threshold",
    )


def test_work_allowance_equality_is_admitted_and_smaller_allowance_is_unavailable(
    mock_target,
):
    arguments = _controlled_arguments()
    # At zero source form radius the supplied kinetic work is exactly a².
    exact_work = arguments["pulse_amplitude"] ** 2
    arguments["work_allowance"] = exact_work
    report = owner.assess_sine_two_port_probe(**arguments)
    assert report.joined_work_bounds == (exact_work, exact_work)
    assert report.work_allowance_margin == 0 and report.work_certified
    assert report.status == "certified_probe"
    arguments["work_allowance"] = 0
    failed = owner.assess_sine_two_port_probe(**arguments)
    assert failed.work_allowance_margin == -exact_work and not failed.work_certified
    assert failed.response_certified and failed.recovery_certified
    assert failed.unavailable_reasons == (
        "supplied_work_upper_bound_exceeds_allowance",
    )


@pytest.mark.parametrize("failure", ("attraction", "missing_margin", "small_margin"))
def test_unavailable_mock_target_preserves_candidates_without_certifying_bounds(
    mock_target, failure
):
    target, _ = mock_target
    if failure == "attraction":
        target.local_attraction_certified = False
    else:
        target.acute_margin_turns_bounds = (
            None if failure == "missing_margin" else I(Q(1, 64))
        )
    report = owner.assess_sine_two_port_probe(**_controlled_arguments())
    assert not report.target_admitted
    assert not report.response_certified
    assert not report.joined_identity_certified and not report.recovery_certified
    assert report.status == "unavailable"
    for field in (
        "whole_window_form_norm_upper_bound",
        "whole_window_phase_norm_upper_bound",
        "response_error_upper_bound",
        "joined_increment_bounds",
        "recorded_joined_increment_bounds",
        "recorded_contrast_bounds",
        "response_margin",
        "post_probe_excess_storage_upper_bound",
        "capture_storage_margin",
    ):
        assert getattr(report, field) is None
    assert report.whole_window_form_norm_candidate > 0
    assert report.whole_window_phase_norm_candidate > 0
    assert report.post_probe_excess_storage_candidate > 0
    assert report.capture_storage_margin_candidate > 0
    assert report.work_certified and report.positive_supplied_work_certified
    assert (
        report.disconnected_increment_bounds
        == report.disconnected_work_bounds
        == (0, 0)
    )


def test_exact_post_probe_storage_boundary_is_unavailable_even_inside_radius(
    mock_target,
):
    arguments = _controlled_arguments()
    # (1/1800)²+(1/900)² = 1/648000, independently matching the barrier.
    arguments.update(pulse_amplitude=Q(1, 1800), phase_radius=Q(1, 900))
    report = owner.assess_sine_two_port_probe(**arguments)
    assert report.post_probe_excess_storage_upper_bound == Q(1, 648000)
    assert report.capture_storage_margin == 0
    assert report.post_probe_radius_margin > 0
    assert report.response_certified and report.work_certified
    assert not report.joined_identity_certified and not report.recovery_certified
    assert report.unavailable_reasons == (
        "strict_post_probe_capture_storage_not_certified",
    )


def test_exact_post_probe_radius_boundary_is_not_accepted(mock_target):
    arguments = _controlled_arguments()
    arguments["pulse_amplitude"] = Q(1, 38)
    report = owner.assess_sine_two_port_probe(**arguments)
    assert report.post_probe_form_norm_upper_bound == Q(1, 12)
    assert report.post_probe_relative_norm_squared_upper_bound == Q(1, 144)
    assert report.post_probe_radius_margin == 0
    assert not report.joined_identity_certified and not report.recovery_certified
    assert "strict_post_probe_local_radius_not_certified" in report.unavailable_reasons


def test_duration_one_is_admitted_with_an_insufficient_elementary_lower_bound(
    mock_target,
):
    arguments = _controlled_arguments()
    arguments["probe_duration"] = Q(1)
    report = owner.assess_sine_two_port_probe(**arguments)
    assert report.coupled_denominator > 0
    assert report.ideal_heat_increment_bounds[0] == 0
    assert report.joined_increment_bounds[0] < 0
    assert not report.response_certified
    assert report.work_certified and report.recovery_certified
    assert report.unavailable_reasons == (
        "recorded_contrast_not_strictly_above_threshold",
    )


def test_signed_work_is_not_clipped_or_required_positive_by_the_allowance_gate(
    mock_target,
):
    arguments = _controlled_arguments()
    arguments.update(form_radius=Q(1, 1000), pulse_amplitude=Q(1, 10000))
    report = owner.assess_sine_two_port_probe(**arguments)
    assert report.joined_work_bounds[0] < 0 < report.joined_work_bounds[1]
    assert not report.positive_supplied_work_certified
    assert report.work_certified
    assert report.joined_increment_bounds[0] < 0
    assert report.joined_identity_certified and report.recovery_certified


def test_exact_tiny_pulse_preserves_linear_response_and_quadratic_work(mock_target):
    arguments = _controlled_arguments()
    base = owner.assess_sine_two_port_probe(**arguments)
    arguments["pulse_amplitude"] = Q(1, 10**500)
    tiny = owner.assess_sine_two_port_probe(**arguments)
    scale = tiny.pulse_amplitude / base.pulse_amplitude
    assert tiny.joined_increment_bounds == tuple(
        scale * value for value in base.joined_increment_bounds
    )
    assert tiny.joined_work_bounds == tuple(
        scale**2 * value for value in base.joined_work_bounds
    )
    assert (
        tiny.whole_window_form_norm_upper_bound
        == scale * base.whole_window_form_norm_upper_bound
    )
    assert (
        tiny.whole_window_phase_norm_upper_bound
        == scale * base.whole_window_phase_norm_upper_bound
    )
    assert tiny.recorded_contrast_bounds[0] > 0
    assert tiny.joined_work_bounds[0] > 0
    assert tiny.status == "certified_probe"


def test_readout_errors_are_two_per_increment_and_four_per_contrast(mock_target):
    arguments = _controlled_arguments()
    noiseless = owner.assess_sine_two_port_probe(**arguments)
    error = Q(1, 10**8)
    arguments["readout_error_bound"] = error
    noisy = owner.assess_sine_two_port_probe(**arguments)
    lower, upper = noiseless.joined_increment_bounds
    assert noisy.joined_increment_bounds == (lower, upper)
    assert noisy.recorded_joined_increment_bounds == (
        lower - 2 * error,
        upper + 2 * error,
    )
    assert noisy.recorded_disconnected_increment_bounds == (-2 * error, 2 * error)
    assert noisy.recorded_contrast_bounds == (lower - 4 * error, upper + 4 * error)
    assert noisy.response_margin == noiseless.response_margin - 4 * error
    assert noisy.joined_work_bounds == noiseless.joined_work_bounds
    assert noisy.capture_storage_margin == noiseless.capture_storage_margin
