"""Read-only admission and record consistency for the frozen maintenance trial.

These checks execute neither a word nor a trajectory. Mathematical premises,
primitive event state, metric and observations are rebuilt; archived numerical
certificates retain the solver's explicitly recorded provenance.
"""

from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.mathematics._rational_interval import I, arg, cos, pi_interval, sin, sqrt
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_u6 import observe_structural_potential_confinement
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
)
from tnfr.research.collective_form_capacity_maintenance import (
    DOMAIN_RADIUS,
    RADIUS,
    STEP,
    admit_contact,
    identity_squared,
)
from tnfr.research.retained_graph_state import restore_recorded_graph_state
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/collective_form_capacity/maintenance_v1"
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")


def read(path):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(path, max_bytes=64 * 1024**2))
    )


def interval(value):
    assert set(value) == {"lo", "hi"}
    result = I(value["lo"], value["hi"])
    assert result.lo == value["lo"] and result.hi == value["hi"]
    return result


@pytest.fixture(scope="module")
def evidence():
    files = tuple(EVIDENCE.glob("*.json")) + (
        EVIDENCE / "source.zip",
        EVIDENCE / "evaluate.py",
    )
    before = tuple(file_receipt(path) for path in files)
    response = read(EVIDENCE / "response.json")
    yield response, read(EVIDENCE / "report.json")
    assert tuple(file_receipt(path) for path in files) == before


def rebuild(capture):
    graph = restore_recorded_graph_state(capture["state"]).graph
    return graph, bound_relational_sine_support_exchange(graph, reference_model=MODEL)


def test_frozen_inputs_source_archive_and_no_replacement_attempt(evidence):
    freeze = read(EVIDENCE / "freeze.json")
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert freeze["maximum_evaluations"] == 1
    for entry in freeze["artifacts"]:
        assert file_receipt(EVIDENCE / entry["path"], label=entry["path"]) == entry
    for entry in freeze["inputs"]:
        assert file_receipt(ROOT / entry["path"], label=entry["path"]) == entry
    manifest = read(EVIDENCE / "source-manifest.json")
    verify_archive_members(
        EVIDENCE / "source.zip",
        {entry["path"]: entry["sha256"] for entry in manifest["files"]},
        max_bytes=64 * 1024**2,
    )
    assert (
        read(EVIDENCE / "attempt.json")["freeze_sha256"]
        == file_receipt(EVIDENCE / "freeze.json")["sha256"]
    )
    assert not (EVIDENCE / "failure.json").exists()


def test_actual_word_retains_live_grammar_and_repays_the_inherited_debt(evidence):
    response, report = evidence
    assert response["whole_word_admission"]["names"] == (
        "contraction",
        "coherence",
        "silence",
    )
    initial, fields = rebuild(response["initial"])
    assert tuple(node_debt(initial.nodes[node]) for node in (0, 1)) == (1, 1)
    for glyph, event in zip(("NUL", "IL", "SHA"), response["events"], strict=True):
        before, _ = rebuild(event["before"])
        after, consumed = rebuild(event["completed"])
        assert event["glyph"] == event["stage"]["glyph"] == glyph
        assert event["stage"]["schedule"] == "two_phase_jacobi"
        assert event["stage"]["nodes_processed"] == 2
        assert before.graph["GRAMMAR_REJECTION_MODE"] == "raise"
        assert before.graph["VALIDATE_OPERATOR_PRECONDITIONS"] is True
        for node, candidate in zip((0, 1), event["live_candidates"], strict=True):
            assert (
                candidate["allowed"] and validate_candidate(before, node, glyph).allowed
            )
            assert tuple(after.nodes[node]["glyph_history"])[-1] == glyph
        assert event["u2_debt_before"] == tuple(
            node_debt(before.nodes[node]) for node in (0, 1)
        )
        assert event["u2_debt_after"] == tuple(
            node_debt(after.nodes[node]) for node in (0, 1)
        )
        assert (
            consumed.edges == fields.edges
            and consumed.conductances == fields.conductances
        )
    final, _ = rebuild(response["completed"])
    assert tuple(node_debt(final.nodes[node]) for node in (0, 1)) == (0, 0)
    assert report["complete_word_executed"] and report["stage_count"] == 3


def test_canonical_u6_snapshots_rebuild_from_stored_pressure_and_geometry(evidence):
    response, _ = evidence
    pairs = [(response["initial"], response["completed"], response["u6_complete_word"])]
    pairs += [
        (event["before"], event["completed"], event["u6_stage"])
        for event in response["events"]
    ]
    pairs += [
        (item["before"], item["after"], item["u6_raw_to_refreshed"])
        for item in response["refreshes"]
    ]
    for before, after, declared in pairs:
        old, _ = rebuild(before)
        new, _ = rebuild(after)
        old_field, new_field = compute_structural_potential(
            old
        ), compute_structural_potential(new)
        assert before["structural_potential"] == tuple(
            (node, Q(old_field[node])) for node in old
        )
        assert after["structural_potential"] == tuple(
            (node, Q(new_field[node])) for node in new
        )
        actual = observe_structural_potential_confinement(
            new,
            old_field,
            new_field,
            threshold=declared["threshold"],
            reference=declared["reference"],
            time_coverage=declared["time_coverage"],
        )
        assert actual.confined is declared["confined"]
        assert actual.mean_absolute_drift == declared["mean_absolute_drift"]


def test_primitive_endpoint_rebuilds_the_exact_reduction_and_metric(evidence):
    response, _ = evidence
    _, source = rebuild(response["completed"])
    reduction, _, metric, references = admit_contact(source)
    forecast = response["forecast"]
    assert metric == forecast["metric"]
    assert references == forecast["reference_phase_pi_coefficients"]
    assert reduction.multiplicities == forecast["multiplicities"]
    assert reduction.orbits == forecast["orbits"]
    initial = identity_squared(
        reduction.source_form_coordinates + reduction.source_phase_coordinates,
        reduction.multiplicities,
        references,
    )
    assert initial == interval(forecast["initial_identity_squared"])
    assert initial.hi < RADIUS**2


def test_every_retained_step_keeps_metric_radius_and_actual_identity_domain(evidence):
    response, report = evidence
    forecast = response["forecast"]
    _, source = rebuild(response["completed"])
    reduction, _, metric, references = admit_contact(source)
    inverse = exact_matrix_inverse(metric)
    factors = tuple(sqrt(I(inverse[i][i])).hi for i in range(len(metric)))
    center = reduction.source_form_coordinates + reduction.source_phase_coordinates
    radius = Q(0)
    records = forecast["records"]
    for index, record in enumerate(records):
        assert record["time"] == index * STEP and record["duration"] == STEP
        assert (
            record["initial_center"] == center
            and record["initial_metric_radius"] == radius
        )
        tube = tuple(map(interval, record["tube"]))
        margin = (
            DOMAIN_RADIUS**2
            - identity_squared(tube, reduction.multiplicities, references).hi
        )
        assert margin > 0 and record["domain_lower_bounds"] == (margin,)
        assert record["picard_interior_margin"] > 0
        center, radius = record["endpoint_center"], record["endpoint_metric_radius"]
        assert radius >= record["local_metric_error_upper_bound"] >= 0
        endpoint = tuple(
            I(x - factor * radius, x + factor * radius)
            for x, factor in zip(center, factors, strict=True)
        )
        assert identity_squared(
            endpoint, reduction.multiplicities, references
        ) == interval(record["endpoint_identity_squared"])
        assert len(record["whole_step_identity_squared_bins"]) == 8
    assert (
        forecast["validated_duration"]
        == len(records) * STEP
        == report["validated_duration"]
    )
    assert forecast["all_time_maintenance_claimed"] is False
    assert report["all_time_maintenance_claimed"] is False


def test_finite_verdict_uses_the_frozen_radius_and_whole_time_bins(evidence):
    response, report = evidence
    forecast = response["forecast"]
    assert forecast["status"] == report["status"]
    records = forecast["records"]
    if report["status"] == "finite_horizon_maintained":
        assert len(records) == 160 and forecast["validated_duration"] == 40
        assert all(
            interval(value).hi < RADIUS**2
            for row in records
            for value in row["whole_step_identity_squared_bins"]
        )
    elif report["status"] == "finite_departure_certified":
        assert interval(records[-1]["endpoint_identity_squared"]).lo > RADIUS**2
        assert forecast["validated_duration"] <= 40
    else:
        assert report["status"] in (
            "identity_verdict_unavailable",
            "numerical_certificate_unavailable",
        )


def test_recorded_first_departure_has_a_strict_existing_observation_bracket(evidence):
    response, report = evidence
    assert report["status"] == "finite_departure_certified"
    records = response["forecast"]["records"]
    assert records[-1]["time"] == 14 and records[-1]["duration"] == Q(1, 4)
    assert all(
        interval(value).hi < RADIUS**2
        for row in records[:-1]
        for value in row["whole_step_identity_squared_bins"]
    )
    final_bins = records[-1]["whole_step_identity_squared_bins"]
    assert all(interval(value).hi < RADIUS**2 for value in final_bins[:4])
    assert records[-1]["time"] + 4 * STEP / 8 == Q(113, 8)
    final = interval(records[-1]["endpoint_identity_squared"])
    assert Q("0.003923728663717655") < final.lo <= final.hi < Q("0.003923728663717657")
    assert final.lo > RADIUS**2
    # Continuous exact flow therefore first reaches the same radius after
    # 113/8 and before 57/4; this observation neither changes nor enlarges it.


def test_retained_tubes_protect_winding_while_endpoint_decomposes_beyond_joint_radius(
    evidence,
):
    response, _ = evidence
    _, source = rebuild(response["completed"])
    reduction, _, metric, references = admit_contact(source)
    records = response["forecast"]["records"]
    pi = pi_interval()
    maximum = max(
        identity_squared(
            tuple(map(interval, record["tube"])), reduction.multiplicities, references
        ).hi
        for record in records
    )
    edge_error = sqrt(I(2 * maximum)).hi
    assert pi.lo / 2 - (2 * pi.hi / 5 + edge_error) > 0
    assert pi.lo / 2 - edge_error > 0
    # All original edges remain acute, so none can cross the principal-phase
    # cut. The two A0 choices cover its two unequal retained target members;
    # other member choices coincide exactly in the admitted fixed family.
    for cycle in ((0, 2, 4, 6, 8), (1, 2, 4, 6, 8), (10, 12, 14, 16, 18)):
        increments = []
        for left, right in zip(cycle, cycle[1:] + cycle[:1], strict=True):
            gap = I(source.phase[right] - source.phase[left])
            increments.append(arg(cos(gap), sin(gap)))
        winding = sum(increments, I(0)) / (2 * pi)
        assert Q(1, 2) < winding.lo <= 1 <= winding.hi < Q(3, 2)

    inverse = exact_matrix_inverse(metric)
    factors = tuple(sqrt(I(inverse[i][i])).hi for i in range(len(metric)))
    last = records[-1]
    endpoint = tuple(
        I(
            center - factor * last["endpoint_metric_radius"],
            center + factor * last["endpoint_metric_radius"],
        )
        for center, factor in zip(last["endpoint_center"], factors, strict=True)
    )
    size = len(reduction.orbits)
    forms = tuple(endpoint[cell] for cell in reduction.node_orbits)
    errors = tuple(
        endpoint[size + cell] - references[cell] * pi for cell in reduction.node_orbits
    )

    def component(channels):
        means = tuple(sum(channel, I(0)) / len(channel) for channel in channels)
        within = sum(
            (
                sum(((value - mean) ** 2 for value in channel), I(0))
                for channel, mean in zip(channels, means, strict=True)
            ),
            I(0),
        )
        return means, within

    a_means, a_within = component((forms[:10], errors[:10]))
    b_means, b_within = component((forms[10:], errors[10:]))
    registration = 5 * sum(
        ((left - right) ** 2 for left, right in zip(a_means, b_means, strict=True)),
        I(0),
    )
    joint = identity_squared(endpoint, reduction.multiplicities, references)
    recombined = a_within + b_within + registration
    assert max(joint.lo, recombined.lo) <= min(joint.hi, recombined.hi)
    assert a_within.hi < RADIUS**2 and b_within.hi < RADIUS**2 < joint.lo

    # Independently verify the decomposition as exact arithmetic, rather than
    # treating overlapping interval evaluations as an algebraic identity.
    point_channels = (
        tuple(last["endpoint_center"][cell] for cell in reduction.node_orbits),
        tuple(
            last["endpoint_center"][size + cell] - references[cell] * pi.midpoint
            for cell in reduction.node_orbits
        ),
    )
    for channel in point_channels:
        mean = sum(channel) / 20
        a_mean, b_mean = sum(channel[:10]) / 10, sum(channel[10:]) / 10
        assert (
            sum((value - mean) ** 2 for value in channel)
            == sum((value - a_mean) ** 2 for value in channel[:10])
            + sum((value - b_mean) ** 2 for value in channel[10:])
            + 5 * (a_mean - b_mean) ** 2
        )
