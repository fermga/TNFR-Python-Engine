"""Synthetic algebra and inverse controls; no target or trajectory acquisition.

Artificial readouts test necessary compatibility, not a measured response or
existence of a complete trajectory. Frozen forward producers are never run.
"""

import json
from fractions import Fraction as Q
from inspect import Parameter, signature

import mpmath as mp
import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I, cos
from tnfr.physics import relational_sine_two_port_inference as owner


def _rational(value):
    return Q(mp.nstr(value, 85))


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def _arguments():
    # A synthetic leading-order scalar, not an acquired full-flow observation.
    with mp.workdps(100):
        a, h, b = mp.mpf(1) / 1024, mp.mpf(1) / 65536, mp.mpf(23) / 16
        measured = _rational(
            -3 * h * mp.sin(3 * a / 2) * mp.cos(b - a / 2) / (1023 * mp.pi)
        )
    return dict(
        bulk_angle_bounds=(Q(11, 8), Q(3, 2)),
        receiver_short_angle_bounds=(Q(2, 3), Q(1)),
        form_radius=Q(1, 2**50),
        phase_radius=Q(1, 2**50),
        phase_increment=Q(1, 1024),
        probe_duration=Q(1, 65536),
        recorded_increment_bounds=(measured, measured),
        readout_error_bound=Q(1, 2**70),
        readout_gain_bounds=(Q(3, 2), Q(3, 2)),
        refinements=48,
    )


@pytest.fixture(scope="module")
def geometry():
    return owner._inference_geometry()


@pytest.fixture(scope="module")
def report():
    return owner.infer_sine_two_port_geometry(**_arguments())


def test_signature_requires_all_ten_primitive_arguments():
    parameters = signature(owner.infer_sine_two_port_geometry).parameters
    assert set(parameters) == set(_arguments())
    assert len(parameters) == 10
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )


def _independent_graph_and_lift(b, c):
    graph = nx.disjoint_union(nx.cycle_graph(9), nx.cycle_graph(9))
    graph.add_edges_from(((0, 9), (1, 10)))
    degrees = np.array([graph.degree[i] for i in range(18)], dtype=float)
    a = 4 * np.pi - 8 * b
    d = (2 * np.pi - c) / 8
    delta = (a - c) / 2
    phases = np.array(
        [
            0,
            *(a + (j - 1) * b for j in range(1, 9)),
            delta,
            *(delta + c + (j - 1) * d for j in range(1, 9)),
        ]
    )
    phases -= degrees @ phases / sum(degrees)
    laplacian = nx.laplacian_matrix(graph, nodelist=range(18)).toarray()
    return graph, degrees, phases, laplacian


def test_full_affine_lift_preserves_gauge_edges_and_three_periods(geometry):
    graph, degrees, phases, laplacian = _independent_graph_and_lift(23 / 16, 5 / 6)
    coordinates = np.array([np.pi, 23 / 16, 5 / 6])
    nodal = (
        np.array(geometry["nodal_angle_affine_coefficients"], dtype=float) @ coordinates
    )
    gaps = (
        np.array(geometry["edge_angle_affine_coefficients"], dtype=float) @ coordinates
    )
    assert np.allclose(nodal, phases, atol=3e-15, rtol=0)
    assert abs(degrees @ nodal) < 1e-13
    assert geometry["degrees"] == tuple(int(d) for d in degrees)
    assert np.array_equal(np.array(geometry["laplacian"], dtype=float), laplacian)
    assert (
        tuple(sorted(tuple(sorted(edge)) for edge in graph.edges))
        == geometry["geometry"].edges
    )
    for edge, gap, offset in zip(
        geometry["geometry"].edges, gaps, geometry["edge_integer_offsets"]
    ):
        assert gap == pytest.approx(
            phases[edge[1]] - phases[edge[0]] - 2 * np.pi * offset, abs=4e-15
        )
        assert abs(gap) < np.pi / 2
    # Compute periods directly from edge orientation, independently of cycle helpers.
    table = dict(zip(geometry["geometry"].edges, gaps))
    cycles = (tuple(range(9)), tuple(range(9, 18)), (0, 9, 10, 1))
    for cycle, winding in zip(cycles, (2, 1, 0)):
        total = sum(
            table[tuple(sorted((i, j)))] * (1 if i < j else -1)
            for i, j in zip(cycle, cycle[1:] + cycle[:1])
        )
        assert total == pytest.approx(2 * np.pi * winding, abs=5e-15)


def test_noncritical_port_forcing_has_three_cancellations_but_not_four(geometry):
    graph, degrees, phases, laplacian = _independent_graph_and_lift(23 / 16, 5 / 6)
    f = np.array(
        [
            sum(np.sin(phases[j] - phases[i]) for j in graph[i]) / degrees[i]
            for i in range(18)
        ]
    )
    assert np.linalg.norm(f) > 0.001  # This chart member is not an equilibrium.
    assert all(abs(f[i]) < 1e-14 for i in range(18) if i not in (0, 1, 9, 10))
    assert np.sqrt(degrees @ (f * f)) < 4
    normalized = np.diag(1 / degrees) @ laplacian
    q = np.zeros(18)
    q[4], q[5] = 1, -1
    row = q.copy()
    for k in range(4):
        value = row @ f
        assert abs(value) < 1e-13 if k < 3 else abs(value) > 0.000001
        row = row @ normalized
    assert all(not any(row) for row in geometry["readout_forcing_moments"])
    assert (
        np.linalg.eigvalsh(
            np.diag(1 / np.sqrt(degrees)) @ laplacian @ np.diag(1 / np.sqrt(degrees))
        )[-1]
        <= 2
    )


def test_local_full_row_slope_and_readout_dual_norm(geometry):
    graph, degrees, phases, laplacian = _independent_graph_and_lift(23 / 16, 5 / 6)
    q = np.array(geometry["dipole"], dtype=float)
    a = 1 / 256
    kicked = phases + a * q
    gamma = 1 / (1023 * np.pi)
    # Both complete rows are evaluated on this synthetic zero-form source.
    form = np.zeros(18)
    form_rate = -laplacian @ form / degrees + gamma * np.array(
        [
            sum(np.sin(kicked[j] - kicked[i]) for j in graph[i]) / degrees[i]
            for i in range(18)
        ]
    )
    phase_rate = gamma * (laplacian @ form) / degrees
    assert np.array_equal(phase_rate, np.zeros(18))
    assert q @ form_rate == pytest.approx(
        -gamma * (np.sin(23 / 16 + a) - np.sin(23 / 16 - 2 * a)), abs=1e-18
    )
    assert q @ np.diag(1 / degrees) @ q == 1
    assert q @ np.diag(degrees) @ q == 4


def test_noncritical_background_has_nonzero_fourth_order_heat_leakage():
    # Auxiliary linear heat convolution only: no complete sine trajectory or
    # reserved source is evolved. It independently detects the new background
    # contribution which the old critical-target dipole bound could omit.
    graph, degrees, _, laplacian = _independent_graph_and_lift(23 / 16, 5 / 6)
    with mp.workdps(70):
        b, c = mp.mpf(23) / 16, mp.mpf(5) / 6
        a, d = 4 * mp.pi - 8 * b, (2 * mp.pi - c) / 8
        delta = (a - c) / 2
        phases = [
            0,
            *(a + (j - 1) * b for j in range(1, 9)),
            delta,
            *(delta + c + (j - 1) * d for j in range(1, 9)),
        ]
        forcing = [
            sum(mp.sin(phases[j] - phases[i]) for j in graph[i]) / int(degrees[i])
            for i in range(18)
        ]
        generator = mp.zeros(19)
        for i in range(18):
            for j in range(18):
                generator[i, j] = -mp.mpf(int(laplacian[i, j])) / int(degrees[i])
            generator[i, 18] = forcing[i]
        increments = []
        for horizon in (mp.mpf(1) / 16, mp.mpf(1) / 32):
            propagated = mp.expm(horizon * generator)
            increment = propagated[4, 18] - propagated[5, 18]
            assert 0 < abs(increment) < 4 * horizon**4 / 3
            increments.append(increment)
        assert 14 < abs(increments[0] / increments[1]) < 17


def test_synthetic_inference_narrows_nominal_angle_without_existence_claim(report):
    assert report.status == "bounded_candidate"
    assert report.source_admitted and report.finite_response_certified
    assert report.whole_window_acute_certified and report.inverse_enclosure_available
    assert report.nominal_bulk_angle_outer_bounds.contains(Q(23, 16))
    assert report.nominal_bulk_angle_outer_bounds.width < Q(1, 1000)
    assert report.actual_long_arc_mean_outer_bounds.contains(
        report.nominal_bulk_angle_outer_bounds
    )
    assert report.receiver_short_angle_bounds == (Q(2, 3), Q(1))
    assert (
        "bounded_candidate_is_necessary_outer_compatibility_not_existence"
        in report.scope
    )


def test_gain_uncertainty_widens_necessary_geometry_region(report):
    args = _arguments()
    args["readout_gain_bounds"] = (Q(1), Q(2))
    wide = owner.infer_sine_two_port_geometry(**args)
    assert wide.nominal_bulk_angle_outer_bounds.contains(
        report.nominal_bulk_angle_outer_bounds
    )
    assert (
        wide.nominal_bulk_angle_outer_bounds.width
        > 100 * report.nominal_bulk_angle_outer_bounds.width
    )


def test_two_errors_are_in_sensor_units_before_gain_division(report):
    args = _arguments()
    args["readout_error_bound"] = Q(1, 2**35)
    noisy = owner.infer_sine_two_port_geometry(**args)
    lower, upper = args["recorded_increment_bounds"]
    delta = args["readout_error_bound"]
    assert noisy.calibrated_increment_bounds.contains((lower - 2 * delta) / Q(3, 2))
    assert noisy.calibrated_increment_bounds.contains((upper + 2 * delta) / Q(3, 2))
    assert noisy.nominal_bulk_angle_outer_bounds.contains(
        report.nominal_bulk_angle_outer_bounds
    )


def test_actual_mean_expansion_is_not_clipped_to_nominal_prior():
    args = _arguments()
    args.update(bulk_angle_bounds=(Q(23, 16), Q(23, 16)), phase_radius=Q(1, 10000))
    result = owner.infer_sine_two_port_geometry(**args)
    assert result.status == "bounded_candidate"
    assert result.actual_long_arc_mean_outer_bounds.contains(Q(23, 16) - Q(1, 80000))
    assert result.actual_long_arc_mean_outer_bounds.contains(Q(23, 16) + Q(1, 80000))
    assert result.actual_long_arc_mean_outer_bounds.lo < args["bulk_angle_bounds"][0]
    assert result.actual_long_arc_mean_outer_bounds.hi > args["bulk_angle_bounds"][1]
    # The mean of eight donor long gaps telescopes; the 0--1 edge has dual norm sqrt(2/3)<1.
    _, degrees, _, _ = _independent_graph_and_lift(23 / 16, 5 / 6)
    residual = np.zeros(18)
    residual[0], residual[1] = -1, 1
    residual *= float(args["phase_radius"]) / np.sqrt(degrees @ residual**2)
    changes = [residual[j + 1] - residual[j] for j in range(1, 8)] + [
        residual[0] - residual[8]
    ]
    assert sum(changes) == pytest.approx(residual[0] - residual[1])
    assert abs(sum(changes) / 8) < float(args["phase_radius"] / 8)


def test_unexpected_signed_readout_is_incompatible_not_invalid():
    args = _arguments()
    args["recorded_increment_bounds"] = (Q(1), Q(1))
    result = owner.infer_sine_two_port_geometry(**args)
    assert result.status == "incompatible"
    assert result.source_admitted and result.finite_response_certified
    assert not result.inverse_enclosure_available
    assert result.nominal_bulk_angle_outer_bounds is None
    assert result.actual_long_arc_mean_outer_bounds is None
    assert result.unavailable_reasons == ()


def test_source_acute_admission_is_strict_and_precedes_inversion(report):
    args = _arguments()
    args["phase_radius"] = report.nominal_acute_margin
    result = owner.infer_sine_two_port_geometry(**args)
    assert result.source_acute_margin == 0
    assert result.status == "unavailable"
    assert not result.source_admitted and not result.finite_response_certified
    assert result.true_increment_bounds is None
    assert result.finite_remainder_upper_bound is None
    assert result.nominal_bulk_angle_outer_bounds is None


def test_postevent_acuity_is_not_required_for_global_response_bound():
    args = _arguments()
    args.update(phase_increment=Q(1), recorded_increment_bounds=(Q(-1), Q(1)))
    result = owner.infer_sine_two_port_geometry(**args)
    assert result.source_admitted and result.finite_response_certified
    assert not result.whole_window_acute_certified
    assert result.status == "bounded_candidate"


@pytest.mark.parametrize(
    "name", ("phase_increment", "probe_duration", "readout_gain_bounds")
)
def test_tiny_exact_positive_divisor_is_retained_but_numerically_unavailable(name):
    args = _arguments()
    tiny = Q(1, 10**100)
    args[name] = (tiny, tiny) if name == "readout_gain_bounds" else tiny
    result = owner.infer_sine_two_port_geometry(**args)
    assert getattr(result, name) == args[name]
    assert result.status == "unavailable"
    assert result.source_admitted and result.finite_response_certified
    assert result.nominal_bulk_angle_outer_bounds is None


def test_huge_exact_finite_readout_is_admitted_and_excluded():
    args = _arguments()
    args["recorded_increment_bounds"] = (Q(10**400), Q(10**400))
    result = owner.infer_sine_two_port_geometry(**args)
    assert result.recorded_increment_bounds == args["recorded_increment_bounds"]
    assert result.status == "incompatible"


@pytest.mark.parametrize(
    "field,value",
    (
        ("form_radius", -1),
        ("phase_radius", -1),
        ("readout_error_bound", -1),
        ("phase_increment", 0),
        ("phase_increment", Q(1001, 1000)),
        ("probe_duration", 0),
        ("probe_duration", Q(1001, 1000)),
        ("bulk_angle_bounds", (Q(1), Q(3, 2))),
        ("bulk_angle_bounds", (Q(3, 2), Q(11, 8))),
        ("receiver_short_angle_bounds", (Q(2, 3), Q(1001, 1000))),
        ("readout_gain_bounds", (0, 1)),
        ("readout_gain_bounds", (-1, -1)),
        ("refinements", True),
        ("refinements", 0),
        ("refinements", 65),
        ("refinements", Q(2)),
        ("refinements", np.int64(2)),
    ),
)
def test_invalid_domains_are_rejected_before_geometry(monkeypatch, field, value):
    args = _arguments()
    args[field] = value
    monkeypatch.setattr(
        owner,
        "_inference_geometry",
        lambda: pytest.fail("invalid input reached geometry"),
    )
    with pytest.raises((ValueError, TypeError)):
        owner.infer_sine_two_port_geometry(**args)


@pytest.mark.parametrize(
    "value", (True, np.bool_(False), float("nan"), float("inf"), 1j)
)
@pytest.mark.parametrize("field", ("form_radius", "recorded_increment_bounds"))
def test_invalid_authoritative_scalars_do_not_reach_arithmetic(
    monkeypatch, field, value
):
    args = _arguments()
    args[field] = (value, value) if field == "recorded_increment_bounds" else value
    monkeypatch.setattr(
        owner,
        "_inference_geometry",
        lambda: pytest.fail("invalid scalar reached geometry"),
    )
    with pytest.raises((ValueError, TypeError)):
        owner.infer_sine_two_port_geometry(**args)


@pytest.mark.parametrize(
    "value", (I(1), (1,), (1, 2, 3), {1, 2}, {"lo": 1, "hi": 2}, "12")
)
def test_endpoint_pairs_cannot_bypass_primitive_admission(value):
    args = _arguments()
    args["readout_gain_bounds"] = value
    with pytest.raises((ValueError, TypeError)):
        owner.infer_sine_two_port_geometry(**args)


def test_represented_reals_are_admitted_before_materialization():
    args = _arguments()
    args.update(form_radius=np.float64(0.0), readout_gain_bounds=(1.5, 1.5))
    result = owner.infer_sine_two_port_geometry(**args)
    assert result.form_radius == 0
    assert result.readout_gain_bounds == (Q(3, 2), Q(3, 2))
    args["form_radius"] = mp.mpf("1e-4000")
    with pytest.raises((ValueError, TypeError)):
        owner.infer_sine_two_port_geometry(**args)


def test_json_projection_preserves_exact_primitives_and_interval_endpoints(report):
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["schema"] == "tnfr.sine-two-port-inference.v1"
    projected = payload["report"]
    assert projected["bulk_angle_bounds"][0] == {"numerator": 11, "denominator": 8}
    assert projected["phase_increment"] == {"numerator": 1, "denominator": 1024}
    assert projected["status"] == "bounded_candidate"
    endpoint = projected["nominal_bulk_angle_outer_bounds"]["lo"]
    assert (
        Q(endpoint["numerator"], endpoint["denominator"])
        == report.nominal_bulk_angle_outer_bounds.lo
    )
    assert projected["receiver_short_angle_bounds"] == [
        {"numerator": 2, "denominator": 3},
        {"numerator": 1, "denominator": 1},
    ]


def test_exact_branch_endpoints_are_retained_by_inverse_exclusion():
    prior, shift = (Q(11, 8), Q(3, 2)), Q(1, 512)
    for endpoint in prior:
        observed = cos(I(endpoint - shift))
        result, _, _, _ = owner._decreasing_cosine_outer(prior, shift, observed, 64)
        assert result is not None and result.contains(endpoint)


def test_unresolved_midpoint_never_discards_a_possible_preimage(monkeypatch):
    monkeypatch.setattr(owner, "cos", lambda _: I(0, 1))
    prior = (Q(1), Q(2))
    result, band, counts, limited = owner._decreasing_cosine_outer(
        prior, 0, I(Q(1, 2)), 64
    )
    assert result == I(*prior)
    assert band == I(Q(1, 2))
    assert counts == (1, 1) and limited


def test_decreasing_inverse_contains_high_precision_preimages():
    prior, shift = (Q(11, 8), Q(3, 2)), Q(1, 512)
    with mp.workdps(100):
        left, right = Q(7, 5), Q(29, 20)
        observed = I(
            _rational(mp.cos(_mp(right - shift))) - Q(1, 10**80),
            _rational(mp.cos(_mp(left - shift))) + Q(1, 10**80),
        )
    result, _, counts, _ = owner._decreasing_cosine_outer(prior, shift, observed, 48)
    assert result.contains(I(left, right))
    assert result.width < right - left + Q(1, 2**44)
    assert counts == (48, 48)


def test_no_equilibrium_or_frozen_producer_is_needed(monkeypatch):
    from tnfr.physics import relational_sine_two_port_capture as capture
    from tnfr.physics import relational_sine_two_port_compatibility as compatibility
    from tnfr.physics import relational_sine_two_port_dipole as dipole

    def forbidden(*args, **kwargs):
        pytest.fail("inference called a target or frozen forward producer")

    monkeypatch.setattr(compatibility, "assess_sine_two_port_compatibility", forbidden)
    monkeypatch.setattr(compatibility, "_root_enclosures", forbidden)
    monkeypatch.setattr(capture, "assess_sine_two_port_capture", forbidden)
    monkeypatch.setattr(dipole, "assess_sine_two_port_dipole", forbidden)
    assert (
        owner.infer_sine_two_port_geometry(**_arguments()).status == "bounded_candidate"
    )
