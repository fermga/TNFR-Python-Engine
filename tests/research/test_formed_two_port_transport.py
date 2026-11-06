"""Full-law transport controls without an actual-source continuation.

The retained prefix is read only. Manufactured rational phasors test the
two-row algebra; no original-source field, target sign or trajectory runs.
"""

from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import networkx as nx
import pytest

from tnfr.research import formed_two_port_reference as owner

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "formed-two-port-attempt-20261029"
)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    from tnfr.mathematics import _validated_metric, _validated_taylor

    def forbidden(*args, **kwargs):
        pytest.fail("transport bounds may not evaluate a field, target or trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner.FormedTwoPortReference, "field", forbidden)
        patch.setattr(owner, "admit_formed_two_port_target", forbidden)
        patch.setattr(owner, "certify_formed_two_port_fifth_step_acute_exit", forbidden)
        patch.setattr(owner, "formed_two_port_comparison_bounds", forbidden)
        patch.setattr(_validated_metric, "validated_metric_taylor_step", forbidden)
        patch.setattr(_validated_metric, "flow_jets", forbidden)
        patch.setattr(_validated_taylor, "flow_jets", forbidden)
        patch.setattr(_validated_taylor, "picard_tube", forbidden)
        yield


@pytest.fixture(scope="module")
def reference():
    return owner.build_formed_two_port_reference()


@pytest.fixture(scope="module")
def bounds(reference):
    return owner.bound_formed_two_port_transport(reference, EVIDENCE)


def _neighbors(node):
    start, local = 9 * (node // 9), node % 9
    result = [start + (local - 1) % 9, start + (local + 1) % 9]
    if node in (4, 5):
        result.append(node + 9)
    elif node in (13, 14):
        result.append(node - 9)
    return tuple(result)


def test_full_two_row_velocity_lag_identity_in_slow_time(reference):
    # This rational gamma tests the algebra, not the actual source constants.
    gamma = Q(1, 10)
    phasors = ((Q(1), Q(0)), (Q(3, 5), Q(4, 5)), (Q(0), Q(1))) * 6
    form = tuple(Q(i - 8, 19) for i in range(18))

    def current(i, j):
        ci, si = phasors[i]
        cj, sj = phasors[j]
        return sj * ci - cj * si

    def cosine(i, j):
        ci, si = phasors[i]
        cj, sj = phasors[j]
        return cj * ci + sj * si

    def diffusion(values):
        return tuple(
            sum((values[i] - values[j] for j in _neighbors(i)), Q(0))
            / len(_neighbors(i))
            for i in range(18)
        )

    def derivative(values):
        return tuple(
            sum((cosine(i, j) * (values[j] - values[i]) for j in _neighbors(i)), Q(0))
            / len(_neighbors(i))
            for i in range(18)
        )

    force = tuple(
        sum((current(i, j) for j in _neighbors(i)), Q(0)) / len(_neighbors(i))
        for i in range(18)
    )
    ax = diffusion(form)
    x_tau = tuple(-a + gamma * f for a, f in zip(ax, force, strict=True))
    theta_tau = tuple(gamma * a for a in ax)
    velocity = tuple(value / gamma**2 for value in theta_tau)
    lag = tuple(v - f for v, f in zip(velocity, force, strict=True))
    direct = tuple(
        a / gamma**3 - b
        for a, b in zip(diffusion(x_tau), derivative(velocity), strict=True)
    )
    predicted = tuple(
        -a / gamma**2 - b
        for a, b in zip(diffusion(lag), derivative(velocity), strict=True)
    )
    assert direct == predicted
    assert any(direct)
    for values in (force, velocity, lag, direct):
        assert (
            sum(d * value for d, value in zip(reference.degrees, values, strict=True))
            == 0
        )


def test_global_source_constants_and_boundary_layer_are_exact(reference, bounds):
    epsilon = Q(1, 10**32)
    assert bounds.initial_phase_velocity_upper_bound == 6 * 3216 * epsilon
    assert bounds.initial_velocity_lag_upper_bound == Q(3, 5) + 19302 * epsilon
    assert Q(10, 1023**2) < Q(1, 100000)
    assert bounds.lag_decay_lower_bound == 100000 - 2
    assert bounds.tracking_start_slow_time == Q(1, 128)
    assert bounds.lag_decay_lower_bound * bounds.tracking_start_slow_time > 512
    assert bounds.initial_transient_decay_upper_bound == Q(1, 2**512)
    assert bounds.velocity_lag_upper_bound == (
        bounds.initial_velocity_lag_upper_bound * Q(1, 2**512) + Q(14, 99998)
    )
    assert 0 < bounds.velocity_lag_upper_bound < Q(1, 7000)
    graph = nx.Graph()
    graph.add_edges_from(reference.oriented_edges)
    assert nx.diameter(graph) <= 9
    assert sum(dict(graph.degree()).values()) == 40
    assert Q(4, 40 * 9) == Q(1, 90)


def test_edge_tracking_and_finite_principal_retention_have_distinct_horizons(
    reference, bounds
):
    expected = tuple(
        Q(1, len(_neighbors(i))) + Q(1, len(_neighbors(j)))
        for i, j in reference.oriented_edges
    )
    assert bounds.edge_dual_norm_squares == expected
    assert bounds.edge_velocity_lag_squared_bounds == tuple(
        value * bounds.velocity_lag_upper_bound**2 for value in expected
    )
    post = bounds.post_acute_window
    assert bounds.retention_slow_interval == (Q(0), Q(7, 4))
    assert bounds.retained_periods == (2, 1, 0)
    assert bounds.reference_edge_displacement_upper_bound == 2 * Q(3, 4)
    assert bounds.principal_margin_lower_bound == (
        Q(1, 16) - reference.reference_margin - post.phase_error_upper_bound
    )
    assert bounds.principal_margin_lower_bound > Q(63, 1024) > Q(3, 50)
    assert min(post.prefix_corridor_slacks) > 0
    assert len(post.prefix_evidence_receipts) == 8


def test_existing_cut_geometry_and_outgoing_normal_use_actual_degrees(
    reference, bounds
):
    coefficients = bounds.cut_gap_pi_coefficients
    phases = [Q(0)] * 18
    for j in range(1, 9):
        phases[j] = phases[j - 1] + coefficients[j - 1]
    phases[9] = phases[4] + coefficients[18] - 4 * Q(2, 9)
    for j in range(10, 18):
        phases[j] = phases[j - 1] + coefficients[j - 1]
    for index, (left, right) in enumerate(reference.oriented_edges):
        lift = Q(4) if index == 8 else Q(2) if index == 17 else Q(0)
        assert phases[right] - phases[left] + lift == coefficients[index]
    assert coefficients[5] == 1
    assert all(abs(value) < 1 for i, value in enumerate(coefficients) if i != 5)

    # Formal current basis: donor bulk sine, receiver sine, positive contact sine.
    current_rows = (
        tuple((Q(0), Q(0), Q(0)) if i == 5 else (Q(1), Q(0), Q(0)) for i in range(9))
        + ((Q(0), Q(1), Q(0)),) * 9
        + ((Q(0), Q(0), Q(1)), (Q(0), Q(0), Q(-1)))
    )
    rates = [[Q(0)] * 3 for _ in range(18)]
    for (left, right), row in zip(reference.oriented_edges, current_rows, strict=True):
        for column, value in enumerate(row):
            rates[left][column] += value / len(_neighbors(left))
            rates[right][column] -= value / len(_neighbors(right))
    normal = tuple(b - a for a, b in zip(rates[5], rates[6], strict=True))
    assert normal == (Q(5, 6), Q(0), Q(1, 3))
    chord_bounds = (Q(3, 4), Q(4, 9), Q(11, 72))
    assert sum(a * b for a, b in zip(normal, chord_bounds, strict=True)) == Q(73, 108)
    assert bounds.conditional_cut_normal_lower_bound == (
        Q(73, 108) - 2 * bounds.cut_patch_radius - bounds.velocity_lag_upper_bound
    )
    assert bounds.conditional_cut_normal_lower_bound > Q(2, 3)
    assert bounds.local_outgoing_periods == (1, 1, 0)
    assert (
        "no_patch_arrival_target_entry_eventual_winding_or_basin_is_selected"
        in bounds.scope
    )


def test_transport_reuses_only_the_admitted_four_step_reader(reference, monkeypatch):
    from tnfr.research import formed_two_port_run

    original = formed_two_port_run.read_step_certificate
    indices = []

    def observed(directory, index):
        indices.append(index)
        return original(directory, index)

    monkeypatch.setattr(formed_two_port_run, "read_step_certificate", observed)
    owner.bound_formed_two_port_transport(reference, EVIDENCE)
    assert indices == [0, 1, 2, 3]


def test_actual_phase_target_ball_has_a_conditional_full_state_handoff(bounds):
    radius = bounds.conditional_capture_phase_radius
    form = bounds.conditional_capture_form_upper_bound
    assert radius == Q(1, 1024)
    assert form == Q(90, 3069) * (2 * radius + bounds.velocity_lag_upper_bound)
    assert form < Q(1, 8192)
    assert bounds.conditional_capture_excess_upper_bound == form**2 + radius**2
    assert bounds.conditional_capture_excess_upper_bound < Q(1, 648000)
    assert bounds.conditional_capture_storage_margin > 0
    assert (
        "capture_bounds_require_actual_phase_distance_to_the_exact_retained_target"
        in bounds.scope
    )


def test_transport_rejects_changed_source_or_corrupted_evidence(
    reference, monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="reference primitives"):
        owner.bound_formed_two_port_transport(
            replace(reference, initial_radius=False), EVIDENCE
        )
    with pytest.raises(OSError):
        owner.bound_formed_two_port_transport(reference, tmp_path)
    original = owner.read_bytes_bounded

    def corrupt(path, **kwargs):
        data = original(path, **kwargs)
        return data + b" " if Path(path).name == "0003.json" else data

    monkeypatch.setattr(owner, "read_bytes_bounded", corrupt)
    with pytest.raises(ValueError, match="prefix receipt differs"):
        owner.bound_formed_two_port_transport(reference, EVIDENCE)
