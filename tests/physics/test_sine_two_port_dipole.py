"""Static graph, nonlinear metric and admission controls for the dipole gate.

No genuine target or frozen source producer is run by these tests. Synthetic
local states exercise identities independently of the reserved protocol.
Post-freeze interface controls use explicitly mocked target evidence; they
do not supply a scientific response, target or trajectory.
"""

from fractions import Fraction as Q
from inspect import signature
from types import SimpleNamespace

import mpmath
import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_two_port_dipole as owner
from tnfr.physics._sine_lyapunov import _sine_lyapunov_coefficients


@pytest.fixture(scope="module")
def geometry():
    return owner._dipole_geometry()


def _graphs():
    control = nx.disjoint_union(nx.cycle_graph(9), nx.cycle_graph(9))
    joined = control.copy()
    joined.add_edges_from(((0, 9), (1, 10)))
    return joined, control


def test_actual_support_dipole_and_component_means_are_independently_rebuilt(geometry):
    q = tuple(Q(int(i == 4) - int(i == 5)) for i in range(18))
    assert geometry["dipole"] == q
    for index, graph in enumerate(_graphs()):
        edges = tuple(sorted(tuple(sorted(edge)) for edge in graph.edges))
        actual_edges = (
            geometry["geometry"].edges if index == 0 else geometry["disconnected_edges"]
        )
        assert actual_edges == edges
        degrees = tuple(graph.degree[i] for i in graph)
        assert geometry["degrees_by_model"][index] == degrees
        assert sum(d * v**2 for d, v in zip(degrees, q)) == 4
        assert sum(v**2 / d for d, v in zip(degrees, q)) == 1
        for component in nx.connected_components(graph):
            assert sum(degrees[i] * q[i] for i in component) == 0
        changes = {(i, j): q[j] - q[i] for i, j in edges if q[i] != q[j]}
        assert changes == {(3, 4): 1, (4, 5): -2, (5, 6): 1}
        assert sum(value**2 for value in changes.values()) == 6


def test_each_spectral_slack_removes_exactly_its_actual_component_means(geometry):
    for index, graph in enumerate(_graphs()):
        degrees = tuple(graph.degree[i] for i in graph)
        components = tuple(nx.connected_components(graph))
        laplacian = geometry["laplacians"][index]
        for i in graph:
            for j in graph:
                expected = Q(degrees[i] if i == j else -int(graph.has_edge(i, j)))
                assert laplacian[i][j] == expected
                centered = Q(degrees[i] * int(i == j)) - sum(
                    (
                        Q(degrees[i] * degrees[j], sum(degrees[k] for k in component))
                        for component in components
                        if i in component and j in component
                    ),
                    Q(0),
                )
                assert (
                    geometry["normalized_gap_slack_matrices"][index][i][j]
                    == expected - centered / 90
                )
                assert (
                    geometry["normalized_upper_slack_matrices"][index][i][j]
                    == 2 * degrees[i] * int(i == j) - expected
                )
        inverse_root = np.diag(1 / np.sqrt(degrees))
        normalized = inverse_root @ np.array(laplacian, dtype=float) @ inverse_root
        eigenvalues = np.linalg.eigvalsh(normalized)
        assert sum(abs(value) < 1e-12 for value in eigenvalues) == len(components)
        assert eigenvalues[len(components)] > 1 / 90
        assert eigenvalues[-1] <= 2 + 1e-12


def test_nonlinear_degree_isometry_proves_two_rows_and_mixed_energy_cancellation(
    geometry,
):
    # A synthetic zero-phase equilibrium on the actual nonuniform-degree
    # support tests the shared mechanism, not the reserved winding target.
    graph = _graphs()[0]
    d = np.array([graph.degree[i] for i in graph], dtype=float)
    root = np.diag(np.sqrt(d))
    inverse_root = np.diag(1 / np.sqrt(d))
    laplacian = np.array(geometry["laplacians"][0], dtype=float)
    a = np.diag(1 / d) @ laplacian
    ahat = inverse_root @ laplacian @ inverse_root
    eigenvalues, eigenvectors = np.linalg.eigh(ahat)
    lambdas, basis = eigenvalues[1:], eigenvectors[:, 1:]
    x = np.array([((i * 7) % 13 - 6) / 100 for i in graph])
    y = np.array([((i * 5) % 17 - 8) / 50 for i in graph])
    x -= d @ x / sum(d)
    y -= d @ y / sum(d)
    gradient, hessian = np.zeros(18), np.zeros((18, 18))
    potential = 0.0
    for i, j in graph.edges:
        gap = y[i] - y[j]
        current, stiffness = np.sin(gap), np.cos(gap)
        gradient[i] += current
        gradient[j] -= current
        hessian[i, i] += stiffness
        hessian[j, j] += stiffness
        hessian[i, j] -= stiffness
        hessian[j, i] -= stiffness
        potential += 1 - np.cos(gap)
    hhat = inverse_root @ hessian @ inverse_root
    # Variable cosines make the phase Hessian and damping fail to commute.
    # The proof must use congruence bounds, not simultaneous eigenvectors.
    assert np.linalg.norm(ahat @ hhat - hhat @ ahat) > 1e-3
    gamma = 1 / 7
    form_rate = -a @ x - gamma * gradient / d
    phase_rate = gamma * a @ x
    xi = (basis.T @ root @ y) / np.sqrt(lambdas)
    v = gamma * np.sqrt(lambdas) * (basis.T @ root @ x)
    xi_rate = (basis.T @ root @ phase_rate) / np.sqrt(lambdas)
    v_rate = gamma * np.sqrt(lambdas) * (basis.T @ root @ form_rate)
    w_gradient = gamma**2 * np.sqrt(lambdas) * (basis.T @ inverse_root @ gradient)
    np.testing.assert_allclose(xi_rate, v, atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(
        v_rate, -lambdas * v - w_gradient, atol=1e-14, rtol=1e-12
    )
    assert np.isclose(np.linalg.norm(root @ y) ** 2, np.dot(lambdas * xi, xi))
    assert np.isclose(np.dot(v, v), gamma**2 * np.dot(x, laplacian @ x))
    mu, maximum, lower, upper, rate = _sine_lyapunov_coefficients(
        eta_lower=Q(1, 49),
        eta_upper=Q(1, 49),
        cosine_lower=Q(1, 2),
        gap_lower=Q(1, 90),
        rate_upper=Q(2),
    )
    epsilon = 1 / 360
    modified = (
        np.dot(v, v) / 2
        + gamma**2 * potential
        + epsilon * np.dot(xi, v)
        + epsilon * np.dot(xi, lambdas * xi) / 2
    )
    derivative = np.dot(
        w_gradient + epsilon * v + epsilon * lambdas * xi, xi_rate
    ) + np.dot(v + epsilon * xi, v_rate)
    cancellation = -np.dot(v, (lambdas - epsilon) * v) - epsilon * np.dot(
        xi, w_gradient
    )
    assert abs(derivative - cancellation) < 1e-14
    assert derivative <= -float(rate) * modified
    assert np.dot(v, v) / 4 + float(lower) * np.dot(xi, xi) <= modified
    assert modified <= 3 * np.dot(v, v) / 4 + float(upper) * np.dot(xi, xi)
    transformed_hessian = (
        gamma**2
        * np.diag(np.sqrt(lambdas))
        @ basis.T
        @ hhat
        @ basis
        @ np.diag(np.sqrt(lambdas))
    )
    spectrum = np.linalg.eigvalsh(transformed_hessian)
    assert spectrum[0] >= float(mu) and spectrum[-1] <= float(maximum)


def test_full_local_phase_jump_has_the_declared_signed_slope_and_exact_work(geometry):
    mp = mpmath.mp.clone()
    mp.dps = 70
    b, amplitude, gamma = mp.mpf(3) / 5, mp.mpf(13) / 1000, mp.mpf(1) / 7
    q = geometry["dipole"]
    before = [b * (i % 9) for i in range(18)]
    after = [value + amplitude * int(mask) for value, mask in zip(before, q)]
    for index, graph in enumerate(_graphs()):
        degrees = geometry["degrees_by_model"][index]

        def field(phases):
            return [
                gamma
                * sum(mp.sin(phases[j] - phases[i]) for j in graph[i])
                / degrees[i]
                for i in graph
            ]

        actual_slope = sum(int(mask) * value for mask, value in zip(q, field(after)))
        expected_slope = -gamma * (mp.sin(b + amplitude) - mp.sin(b - 2 * amplitude))
        assert abs(actual_slope - expected_slope) < mp.mpf("1e-65")
        assert abs(
            sum(int(mask) * value for mask, value in zip(q, field(before)))
        ) < mp.mpf("1e-65")
        work = sum(
            mp.cos(before[j] - before[i]) - mp.cos(after[j] - after[i])
            for i, j in graph.edges
        )
        expected_work = (
            3 * mp.cos(b) - 2 * mp.cos(b + amplitude) - mp.cos(b - 2 * amplitude)
        )
        assert abs(work - expected_work) < mp.mpf("1e-65")
    left, right = mp.mpf(7) / 5, mp.mpf(3) / 2
    direct = (
        -(mp.sin(right + amplitude) - mp.sin(right - 2 * amplitude))
        + mp.sin(left + amplitude)
        - mp.sin(left - 2 * amplitude)
    )
    correlated = (
        2
        * mp.sin(3 * amplitude / 2)
        * (mp.cos(left - amplitude / 2) - mp.cos(right - amplitude / 2))
    )
    assert abs(direct - correlated) < mp.mpf("1e-65")


def test_phase_blind_complete_rows_ignore_phase_jump_but_sine_rows_do_not(geometry):
    for index, graph in enumerate(_graphs()):
        d = np.array(geometry["degrees_by_model"][index], dtype=float)
        a = np.diag(1 / d) @ np.array(geometry["laplacians"][index], dtype=float)
        x = np.linspace(-0.01, 0.02, 18)
        theta = np.linspace(-0.1, 0.1, 18)
        jumped = theta + np.array(geometry["dipole"], dtype=float) / 50

        def rows(phases, feedback):
            sine = np.array(
                [
                    sum(np.sin(phases[j] - phases[i]) for j in graph[i]) / d[i]
                    for i in graph
                ]
            )
            return np.concatenate((-a @ x + feedback * sine / 7, a @ x / 7))

        np.testing.assert_array_equal(rows(theta, 0), rows(jumped, 0))
        assert np.linalg.norm(rows(theta, 1)[:18] - rows(jumped, 1)[:18]) > 1e-3
        np.testing.assert_array_equal(rows(theta, 1)[18:], rows(jumped, 1)[18:])


def test_static_warmup_coefficients_and_pi_bounds_are_exact():
    values = owner._warmup_envelope(Q(0))
    assert values["lyapunov_decay_rate"] == Q(1, 2233865700000)
    assert values["phase_stiffness_lower_bound"] == Q(1, 2227500000000)
    assert values["lyapunov_position_lower_coefficient"] == Q(34375001, 4455000000000)
    assert values["lyapunov_position_upper_coefficient"] == Q(225643, 81000000)
    assert values["warmup_decay_upper_bound"] == 1
    assert values["warmup_form_norm_squared_candidate"] > Q(1, 144)
    # Strict consequences of3<pi<22/7, without evaluating a target or probe.
    assert Q(1, 11000000) < Q(7**2, 1023**2 * 22**2)
    assert Q(1, 1023**2 * 3**2) < Q(1, 9000000)


def _arguments():
    return dict(
        warmup_duration=Q(0),
        form_radius=Q(1, 20),
        phase_radius=Q(1, 20),
        phase_increment=Q(1, 100),
        probe_duration=Q(1, 20),
        readout_error_bound=Q(0),
        contrast_threshold=Q(0),
        work_allowance=Q(1),
    )


@pytest.mark.parametrize("field", tuple(_arguments()))
@pytest.mark.parametrize("value", (True, np.bool_(False), float("nan"), float("inf")))
def test_invalid_scalar_rejects_before_geometry_or_target(field, value, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached geometry or a scientific target")

    monkeypatch.setattr(owner, "_dipole_geometry", forbidden)
    monkeypatch.setattr(owner, "assess_sine_two_port_compatibility", forbidden)
    arguments = _arguments()
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        owner.assess_sine_two_port_dipole(**arguments)


@pytest.mark.parametrize(
    "field,value",
    tuple((key, -Q(1, 100)) for key in _arguments())
    + (
        ("phase_increment", 0),
        ("probe_duration", 0),
        ("phase_increment", Q(1001, 1000)),
        ("probe_duration", Q(1001, 1000)),
        ("phase_increment", np.longdouble("1e-400")),
        ("warmup_duration", Q(4097 * 2233865700000)),
    ),
)
def test_invalid_domain_and_exponent_budget_reject_before_target(
    field, value, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid domain reached a scientific target")

    monkeypatch.setattr(owner, "_dipole_geometry", forbidden)
    monkeypatch.setattr(owner, "assess_sine_two_port_compatibility", forbidden)
    arguments = _arguments()
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        owner.assess_sine_two_port_dipole(**arguments)


@pytest.mark.parametrize(
    "field,value",
    (
        ("phase_increment", Q(1, 10**500)),
        ("form_radius", Q(10**500)),
        ("probe_duration", 1),
        ("warmup_duration", Q(8193, 2) * 2233865700000),
    ),
)
def test_valid_extreme_exact_admission_stops_before_scientific_assessment(
    field, value, monkeypatch
):
    class GeometryReached(Exception):
        pass

    def stop():
        raise GeometryReached

    monkeypatch.setattr(owner, "_dipole_geometry", stop)
    arguments = _arguments()
    arguments[field] = value
    with pytest.raises(GeometryReached):
        owner.assess_sine_two_port_dipole(**arguments)


def test_api_requires_all_eight_primitives_without_an_incoming_source_report():
    parameters = signature(owner.assess_sine_two_port_dipole).parameters
    assert tuple(parameters) == tuple(_arguments())
    assert all(
        value.default is value.empty and value.kind is value.KEYWORD_ONLY
        for value in parameters.values()
    )
    assert not any("source" in name or "report" in name for name in parameters)


@pytest.fixture
def mock_target(monkeypatch):
    """An interface fixture only; none of its target claims is acquired here."""
    target = SimpleNamespace(
        local_attraction_certified=True,
        acute_margin_turns_bounds=I(Q(1, 50)),
        bulk_arc_turn_bounds=(I(Q(23, 100)), I(Q(1, 9))),
    )
    calls = []

    def supplied(**kwargs):
        calls.append(kwargs)
        return target

    monkeypatch.setattr(owner, "assess_sine_two_port_compatibility", supplied)
    return target, calls


def _controlled_arguments():
    # Different from the reserved protocol; all targets below are mocked.
    return dict(
        warmup_duration=Q(144 * 2233865700000),
        form_radius=Q(1, 2**35),
        phase_radius=Q(1, 2**35),
        phase_increment=Q(1, 8192),
        probe_duration=Q(1, 2048),
        readout_error_bound=Q(0),
        contrast_threshold=Q(0),
        work_allowance=Q(1),
    )


def test_mocked_endpoint_pipeline_keeps_finite_warmup_and_complete_alternative(
    mock_target,
):
    target, calls = mock_target
    report = owner.assess_sine_two_port_dipole(**_controlled_arguments())
    assert calls == [dict(classes=(2, 1), outer_refinements=32, inner_refinements=64)]
    assert report.target is target
    assert report.status == "certified_dipole" and report.unavailable_reasons == ()
    assert (
        report.warmup_certified and report.target_admitted and report.geometry_certified
    )
    assert report.response_certified and report.heat_control_excluded
    assert (
        report.work_certified
        and report.identity_certified
        and report.recovery_certified
    )
    assert report.warmup_decay_power == 144
    assert report.warmup_decay_upper_bound == Q(1, 2**144)
    assert report.warmup_form_norm_squared_upper_bound < report.form_radius**2
    assert report.warmup_phase_norm_squared_upper_bound < report.phase_radius**2
    assert report.heat_warmup_form_norm_upper_bound == Q(7, 65536 * 2**128)
    assert report.heat_warmup_form_norm_upper_bound <= report.form_radius
    assert (
        report.phase_blind_law
        == "x'=-K*L*x; theta'=gamma*K*L*x; no phase-to-form feedback"
    )
    assert report.model_order == ("joined_two_port", "unjoined_two_cycles")
    assert report.degrees_by_model[0] != report.degrees_by_model[1]


@pytest.mark.parametrize("channel", ("form", "phase"))
def test_exact_warmup_budget_equality_cannot_deliver_endpoint_claims(
    mock_target, monkeypatch, channel
):
    arguments = _controlled_arguments()
    original = owner._warmup_envelope

    def boundary(duration):
        values = original(duration)
        # Controlled boundary at the private-kernel interface, not a claim
        # that these modified values came from an acquired source.
        values[f"warmup_{channel}_norm_squared_candidate"] = (
            arguments[f"{channel}_radius"] ** 2
        )
        return values

    monkeypatch.setattr(owner, "_warmup_envelope", boundary)
    report = owner.assess_sine_two_port_dipole(**arguments)
    assert getattr(report, f"warmup_{channel}_margin") == 0
    assert report.target_admitted and report.geometry_certified
    assert not report.warmup_certified
    assert not report.response_certified and not report.heat_control_excluded
    assert (
        not report.work_certified
        and not report.identity_certified
        and not report.recovery_certified
    )
    assert report.ideal_correlated_contrast_bounds is not None
    assert report.work_bounds_candidates is not None
    assert report.post_probe_excess_storage_candidates is not None
    for name in (
        "finite_remainder_upper_bound",
        "true_increment_bounds_by_model",
        "recorded_increment_bounds_by_model",
        "recorded_contrast_bounds",
        "response_margin",
        "phase_blind_exclusion_margin",
        "work_bounds_by_model",
        "work_allowance_margins",
        "post_probe_excess_storage_upper_bounds",
        "post_probe_radius_margin",
        "capture_storage_margins",
    ):
        assert getattr(report, name) is None


@pytest.mark.parametrize("failure", ("attraction", "margin", "bulk"))
def test_unavailable_target_leaves_target_dependent_bounds_unavailable(
    mock_target, failure
):
    target, _ = mock_target
    if failure == "attraction":
        target.local_attraction_certified = False
    elif failure == "margin":
        target.acute_margin_turns_bounds = None
    else:
        target.bulk_arc_turn_bounds = None
    report = owner.assess_sine_two_port_dipole(**_controlled_arguments())
    assert not report.target_admitted and not report.warmup_certified
    assert not report.geometry_certified and not report.response_certified
    assert not report.work_certified and not report.identity_certified
    for name in (
        "warmup_form_norm_squared_upper_bound",
        "warmup_phase_norm_squared_upper_bound",
        "warmup_form_margin",
        "warmup_phase_margin",
        "bulk_angle_bounds_by_model",
        "cosine_gap_bounds",
        "ideal_initial_slope_bounds_by_model",
        "ideal_increment_bounds_by_model",
        "ideal_correlated_contrast_bounds",
        "recorded_contrast_bounds",
        "work_bounds_candidates",
        "work_bounds_by_model",
        "post_probe_excess_storage_candidates",
        "capture_storage_margins",
    ):
        assert getattr(report, name) is None
    assert report.warmup_form_norm_squared_candidate > 0
    assert report.heat_warmup_certified
    assert report.phase_blind_recorded_contrast_upper_bound is not None


def test_exact_cosine_gap_boundary_is_not_a_geometry_certificate(
    mock_target, monkeypatch
):
    cosine = owner.cos
    calls = 0

    def boundary(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            return I(Q(1, 16))
        if calls == 2:
            return I(Q(1, 32))
        return cosine(value)

    monkeypatch.setattr(owner, "cos", boundary)
    report = owner.assess_sine_two_port_dipole(**_controlled_arguments())
    assert report.cosine_gap_bounds.lo == report.cosine_gap_threshold
    assert report.warmup_certified and report.response_margin > 0
    assert not report.geometry_certified
    assert not report.response_certified and not report.heat_control_excluded
    assert "strict_target_cosine_gap_not_certified" in report.unavailable_reasons


def test_recorded_contrast_threshold_equality_is_unavailable(mock_target):
    arguments = _controlled_arguments()
    baseline = owner.assess_sine_two_port_dipole(**arguments)
    arguments["contrast_threshold"] = baseline.recorded_contrast_bounds.lo
    report = owner.assess_sine_two_port_dipole(**arguments)
    assert report.response_margin == 0 and not report.response_certified
    assert report.warmup_certified and report.heat_control_excluded
    assert report.work_certified and report.recovery_certified
    assert report.unavailable_reasons == (
        "recorded_contrast_not_strictly_above_threshold",
    )


def test_work_allowance_equality_is_admitted_for_both_models(mock_target):
    arguments = _controlled_arguments()
    baseline = owner.assess_sine_two_port_dipole(**arguments)
    allowance = max(value.hi for value in baseline.work_bounds_by_model)
    arguments["work_allowance"] = allowance
    report = owner.assess_sine_two_port_dipole(**arguments)
    assert min(report.work_allowance_margins) == 0
    assert report.work_certified_by_model == (True, True)
    assert report.work_certified and report.status == "certified_dipole"
    arguments["work_allowance"] = 0
    failed = owner.assess_sine_two_port_dipole(**arguments)
    assert failed.positive_work_certified_by_model == (True, True)
    assert failed.work_certified_by_model == (False, False)
    assert failed.response_certified and failed.identity_certified


def test_heat_warmup_exponent_and_radius_equalities_are_admitted(mock_target):
    arguments = _controlled_arguments()
    arguments.update(
        warmup_duration=Q(128 * 90), form_radius=Q(10000), phase_radius=Q(100)
    )
    boundary = owner.assess_sine_two_port_dipole(**arguments)
    assert boundary.heat_warmup_exponent == 128
    assert boundary.heat_warmup_certified
    arguments["warmup_duration"] -= 1
    below = owner.assess_sine_two_port_dipole(**arguments)
    assert below.heat_warmup_exponent < 128
    assert not below.heat_warmup_certified and not below.heat_control_excluded
    assert below.heat_warmup_form_norm_upper_bound is None
    assert below.phase_blind_recorded_contrast_upper_bound is None
    arguments = _controlled_arguments()
    arguments.update(
        warmup_duration=Q(512 * 2233865700000), form_radius=Q(7, 65536 * 2**128)
    )
    equal_radius = owner.assess_sine_two_port_dipole(**arguments)
    assert equal_radius.warmup_certified
    assert equal_radius.heat_warmup_form_norm_upper_bound == equal_radius.form_radius
    assert equal_radius.heat_warmup_certified
    arguments["form_radius"] /= 2
    smaller_radius = owner.assess_sine_two_port_dipole(**arguments)
    assert smaller_radius.warmup_certified and not smaller_radius.heat_warmup_certified


def test_four_readout_errors_expand_contrast_and_phase_blind_control_in_opposite_directions(
    mock_target,
):
    arguments = _controlled_arguments()
    baseline = owner.assess_sine_two_port_dipole(**arguments)
    error = Q(1, 2**60)
    arguments["readout_error_bound"] = error
    report = owner.assess_sine_two_port_dipole(**arguments)
    assert (
        report.true_increment_bounds_by_model == baseline.true_increment_bounds_by_model
    )
    for clean, noisy in zip(
        baseline.recorded_increment_bounds_by_model,
        report.recorded_increment_bounds_by_model,
    ):
        assert noisy.lo == clean.lo - 2 * error
        assert noisy.hi == clean.hi + 2 * error
    assert (
        report.recorded_contrast_bounds.lo
        == baseline.recorded_contrast_bounds.lo - 4 * error
    )
    assert (
        report.recorded_contrast_bounds.hi
        == baseline.recorded_contrast_bounds.hi + 4 * error
    )
    assert (
        report.phase_blind_recorded_contrast_upper_bound
        == baseline.phase_blind_recorded_contrast_upper_bound + 4 * error
    )
    assert (
        report.phase_blind_exclusion_margin
        == baseline.phase_blind_exclusion_margin - 8 * error
    )
    # Recombining the previously rounded error enclosure can move the boundary
    # by one grid quantum. Cross it by a fixed four-quantum margin, rather than
    # treating a rounded baseline as an exact unrounded response coefficient.
    arguments["readout_error_bound"] = (
        baseline.phase_blind_exclusion_margin + Q(1, 2**126)
    ) / 8
    touching = owner.assess_sine_two_port_dipole(**arguments)
    assert touching.phase_blind_exclusion_margin <= 0
    assert not touching.heat_control_excluded


def test_large_admitted_phase_jump_cannot_inherit_source_identity(mock_target):
    arguments = _controlled_arguments()
    arguments["phase_increment"] = Q(1, 16)
    report = owner.assess_sine_two_port_dipole(**arguments)
    assert report.warmup_certified and report.target_admitted
    assert report.post_probe_radius_margin < 0
    assert all(value < 0 for value in report.capture_storage_margins)
    assert report.identity_certified_by_model == (False, False)
    assert not report.identity_certified and not report.recovery_certified
    assert report.recorded_contrast_bounds is not None
    assert (
        "strict_post_probe_capture_not_certified_for_both_models"
        in report.unavailable_reasons
    )


def test_exact_warmup_decay_below_the_interval_grid_is_never_materialized_as_zero(
    mock_target,
):
    arguments = _controlled_arguments()
    arguments["warmup_duration"] = Q(257 * 2233865700000)
    report = owner.assess_sine_two_port_dipole(**arguments)
    assert report.warmup_decay_power == 257
    assert report.warmup_decay_upper_bound == Q(1, 2**257)
    assert type(report.warmup_form_norm_squared_upper_bound) is Q
    assert 0 < report.warmup_form_norm_squared_upper_bound < Q(1, 2**128)
    assert 0 < report.warmup_phase_norm_squared_upper_bound < Q(1, 2**128)
    assert report.warmup_certified
