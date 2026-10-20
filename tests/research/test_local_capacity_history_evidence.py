"""Read retained temporal evidence without rerunning its numerical producer.

The independent interval certificate is rebuilt once from admitted primitives.
Numerical integration, preparation and operator execution are forbidden here;
records and cached verdicts cannot supply their own mathematical premises.
"""

from datetime import datetime
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA
from tnfr.dynamics import integrators, relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.operators._thol_config import resolve_thol_bifurcation_threshold
from tnfr.operators.definitions import Expansion, SelfOrganization, Silence
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import validate_expansion
from tnfr.operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.research import local_capacity_history as producer
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
)
from tnfr.research.local_capacity_history_certificate import (
    ENDPOINT_BUDGET,
    READER_BUDGET,
    SAMPLE_STEP,
    TIMES,
    admit_source,
    certify_history,
)
from tnfr.research.retained_graph_state import restore_recorded_graph_state
from tnfr.sdk._state import copy_graph_state
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/local_capacity_history"
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=2**25))
    )


def _interval(record):
    assert set(record) == {"lo", "hi"}
    interval = I(record["lo"], record["hi"])
    assert interval.lo == record["lo"] and interval.hi == record["hi"]
    return interval


def _rebuild(record):
    graph = restore_recorded_graph_state(record).graph
    source = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    return graph, source


def _persistent_equal(left, right):
    left, right = copy_graph_state(left), copy_graph_state(right)
    assert left.graph == right.graph
    assert dict(left.nodes) == dict(right.nodes)
    assert dict(left.edges) == dict(right.edges)
    assert tuple((node, tuple(left[node])) for node in left) == tuple(
        (node, tuple(right[node])) for node in right
    )


@pytest.fixture(scope="module", autouse=True)
def no_numerical_response_or_event():
    def forbidden(*args, **kwargs):
        pytest.fail("retained evidence consumers cannot execute a response or event")

    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "declared_preparation",
            "evaluate_local_history",
            "rk4_coupled_step",
            "execute_network_operator_stage",
        ):
            patch.setattr(producer, name, forbidden)
        patch.setattr(integrators, "rk4_coupled_step", forbidden)
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(SelfOrganization, "_prepare_execution", forbidden)
        yield


@pytest.fixture(scope="module")
def evidence():
    files = tuple(sorted(EVIDENCE.iterdir()))
    before = tuple(file_receipt(path) for path in files if path.is_file())
    yield _read("response.json"), _read("report.json")
    assert tuple(file_receipt(path) for path in files if path.is_file()) == before


@pytest.fixture(scope="module")
def rebuilt_certificate(evidence):
    response, _ = evidence
    _, source = _rebuild(response["after_val"])
    return certify_history(source, response["times"], response["states"])


def test_frozen_artifacts_archive_and_response_keep_their_byte_associations(evidence):
    response, report = evidence
    freeze, attempt, protocol = map(
        _read, ("freeze.json", "attempt.json", "protocol.json")
    )
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert (
        protocol["maximum_reserved_evaluations"] == attempt["maximum_evaluations"] == 1
    )
    for entry in freeze["artifacts"]:
        assert file_receipt(EVIDENCE / entry["path"], label=entry["path"]) == entry
    for entry in freeze["prior_artifacts"]:
        assert file_receipt(ROOT / entry["path"], label=entry["path"]) == entry
    entries = _read("source-manifest.json")["files"]
    verify_archive_members(
        EVIDENCE / "source.zip",
        {entry["path"]: entry["sha256"] for entry in entries},
        max_bytes=32 * 1024**2,
    )
    # Every archived source member remains byte-verified above. The two direct
    # study owners must also remain exact: they specify this fixed producer
    # and its independent certificate. Changing either requires review of this
    # study contract, not another dependency exemption.
    # Compatibility of current shared dependencies is checked by the exact
    # reconstruction, independent complete rows and read-only gates below,
    # under the no-response/no-event fixture. It is not identity of all live
    # engine files with an old snapshot or permission to replay that producer.
    entries_by_path = {entry["path"]: entry for entry in entries}
    for path in (
        "src/tnfr/research/local_capacity_history.py",
        "src/tnfr/research/local_capacity_history_certificate.py",
    ):
        assert file_receipt(ROOT / path, label=path) == entries_by_path[path]
    assert attempt["freeze_sha256"] == file_receipt(EVIDENCE / "freeze.json")["sha256"]
    assert report["response"] == file_receipt(
        EVIDENCE / "response.json", label="response.json"
    )
    assert report["checks"] == response["checks"]
    assert (
        datetime.fromisoformat(freeze["created_utc"])
        <= datetime.fromisoformat(attempt["created_utc"])
        <= datetime.fromisoformat(report["completed_utc"])
    )
    assert not (EVIDENCE / "failure.json").exists()


def test_preparation_actual_capacity_event_and_source_admission_rebuild(evidence):
    response, _ = evidence
    frozen, _ = _rebuild(_read("source.json")["graph"])
    before, prior = _rebuild(response["preparation"])
    after, source = _rebuild(response["after_val"])
    _persistent_equal(frozen, before)
    admitted = admit_source(source)
    assert admitted.radius_squared.hi < Q(1, 256)
    assert admitted.storage_excess.hi < Q(1, 8192)
    assert admitted.parent_pressure.lo > Q(1, 2**14)
    assert producer.encode_result(admitted) == response["source_admission"]
    assert (source.epi, source.phase, source.edges, source.conductances) == (
        prior.epi,
        prior.phase,
        prior.edges,
        prior.conductances,
    )
    assert source.capacity == tuple(
        128 * value if index == 2 else value
        for index, value in enumerate(prior.capacity)
    )
    assert tuple(before.nodes[2]["glyph_history"]) == ()
    assert tuple(after.nodes[2]["glyph_history"]) == ("VAL",)
    assert node_debt(before.nodes[2]) == 0 and node_debt(after.nodes[2]) == 1
    assert "epi_time_history" not in after.nodes[2]
    assert len(before) == len(after) == 20 and before.number_of_edges() == 45
    assert response["stage"]["glyph"] == "VAL"
    assert response["stage"]["schedule"] == "two_phase_jacobi"
    assert response["stage"]["nodes_processed"] == 1


def test_live_prefix_and_endpoint_gates_are_read_only_and_keep_their_history(evidence):
    response, _ = evidence
    before, _ = _rebuild(response["preparation"])
    endpoint, source = _rebuild(response["endpoint"])
    prior_copy, endpoint_copy = copy_graph_state(before), copy_graph_state(endpoint)
    producer._admit_execution_settings(before)
    producer._admit_execution_settings(endpoint)
    sequence = ValidatedSequence(
        (Expansion(), SelfOrganization(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    validate_expansion(before, 2)
    assert validate_candidate(
        before, 2, "VAL", sequence_context=sequence.step(0)
    ).allowed
    validate_self_organization_strict(endpoint, 2, emit_warnings=False)
    assert validate_candidate(endpoint, 2, "THOL").allowed
    observation = observe_structural_acceleration(endpoint, 2)
    assert response["threshold"] == Q(
        resolve_thol_bifurcation_threshold(endpoint.graph)
    )
    assert observation.available and observation.current_endpoint_matches_state
    assert observation.value > response["threshold"]
    assert producer.encode_result(observation) == response["observation"]
    assert Q(endpoint.graph["_t"]) == TIMES[-1]
    assert (
        tuple((Q(t), Q(x)) for t, x in endpoint.nodes[2]["epi_time_history"])
        == response["parent_history"]
    )
    assert tuple(endpoint.nodes[2]["glyph_history"]) == ("VAL",)
    assert node_debt(endpoint.nodes[2]) == 1
    assert source.epi + source.phase == response["states"][-1]
    assert Q(get_attr(endpoint.nodes[2], ALIAS_DNFR)) > 0
    _persistent_equal(before, prior_copy)
    _persistent_equal(endpoint, endpoint_copy)


def test_independent_stencil_error_budget_and_report_verdict_follow_records(
    evidence, rebuilt_certificate
):
    response, report = evidence
    certificate = rebuilt_certificate
    assert producer.encode_result(certificate) == response["certificate"]
    assert response["times"] == TIMES
    assert response["parent_history"] == tuple(
        (time, row[2]) for time, row in zip(TIMES, response["states"], strict=True)
    )
    x0, x1, x2 = (row[2] for row in response["states"])
    exact_stencil = (x2 - 2 * x1 + x0) / SAMPLE_STEP**2
    observed = Q(response["observation"]["value"])
    assert abs(observed - exact_stencil) <= READER_BUDGET
    errors = certificate.parent_error_bounds
    independent_sample_error = (errors[0] + 2 * errors[1] + errors[2]) / SAMPLE_STEP**2
    assert certificate.sample_error_bound == independent_sample_error
    assert (
        certificate.trigger_lower_bound
        == Q(1, 8) - independent_sample_error - READER_BUDGET
    )
    assert max(errors) <= ENDPOINT_BUDGET
    assert certificate.trigger_lower_bound > response["threshold"]
    assert observed > response["threshold"]
    assert report["observed_acceleration"] == response["observation"]["value"]
    assert report["certified_acceleration_lower"] == certificate.trigger_lower_bound
    assert report["parent_endpoint_error_bounds"] == errors
    assert report["all_checks_passed"] and all(response["checks"].values())


def test_enclosures_use_full_rows_same_source_and_actual_point_errors(
    evidence, rebuilt_certificate
):
    response, _ = evidence
    certificate = rebuilt_certificate
    assert len(certificate.steps) == 2
    initial = tuple(I(value) for value in response["states"][0])
    for step, end in zip(certificate.steps, TIMES[1:], strict=True):
        assert step.time == TIMES[0] and step.duration == end - TIMES[0]
        assert step.initial_box == initial and len(step.endpoint) == 40
        assert step.picard_interior_margin > 0
        assert all(value > 0 for value in step.domain_lower_bounds)
    for record, enclosure, errors in zip(
        response["states"],
        certificate.endpoint_bounds,
        certificate.state_error_bounds,
        strict=True,
    ):
        assert errors == tuple(
            max(abs(value - bound.lo), abs(value - bound.hi))
            for value, bound in zip(record, enclosure, strict=True)
        )
    assert all(bound.lo > 0 for bound in certificate.true_parent_pressure_bounds)
    assert response["states"][1][30] < 0 and response["states"][2][30] < 0


def test_retained_rk_stage_associations_and_quadrature_residuals_recompute(evidence):
    response, _ = evidence
    for index, step in enumerate(response["numerical_steps"]):
        start, end = TIMES[index : index + 2]
        assert (step["start"], step["end"]) == (start, end)
        h = end - start
        stages = step["stages"]
        assert tuple(stage["time"] for stage in stages) == (
            start,
            start + h / 2,
            start + h / 2,
            end,
        )
        assert stages[0]["state"] == response["states"][index]
        for stage, previous, factor in zip(
            stages[1:], stages[:-1], (h / 2, h / 2, h), strict=True
        ):
            # Dyadic h scales every finite recorded rate exactly here. Only
            # the final addition rounds; this checks supplied stage inputs.
            expected = tuple(
                Q(float(x + factor * rate))
                for x, rate in zip(stages[0]["state"], previous["rates"], strict=True)
            )
            assert stage["state"] == expected
        residual = tuple(
            final
            - initial
            - h
            * sum(
                weight * stage["rates"][j]
                for weight, stage in zip((1, 2, 2, 1), stages, strict=True)
            )
            / 6
            for j, (initial, final) in enumerate(
                zip(
                    response["states"][index],
                    response["states"][index + 1],
                    strict=True,
                )
            )
        )
        assert residual == step["represented_quadrature_residuals"]


def test_recorded_rates_match_independent_high_precision_complete_rows(evidence):
    response, _ = evidence
    _, source = _rebuild(response["after_val"])
    mp = mpmath.mp.clone()
    mp.dps = 90

    def exact(value):
        return mp.mpf(value.numerator) / value.denominator

    adjacent = [[] for _ in source.nodes]
    for (left, right), weight in zip(source.edges, source.conductances, strict=True):
        adjacent[left].append((right, exact(weight)))
        adjacent[right].append((left, exact(weight)))
    for step in response["numerical_steps"]:
        for stage in step["stages"]:
            forms = tuple(map(exact, stage["state"][:20]))
            phases = tuple(map(exact, stage["state"][20:]))
            form_rates, phase_rates = [], []
            for i, neighbors in enumerate(adjacent):
                mobility = exact(source.capacity[i]) / (mp.pi * len(neighbors))
                form_rates.append(
                    mobility
                    * mp.fsum(mp.sin(phases[j] - phases[i]) for j, _ in neighbors)
                )
                phase_rates.append(
                    mobility
                    * mp.fsum(weight * (forms[i] - forms[j]) for j, weight in neighbors)
                )
            # An independent coarse row-consistency control. Actual endpoint
            # accuracy comes from the interval certificate, not this tolerance.
            assert (
                max(
                    abs(exact(value) - expected)
                    for value, expected in zip(
                        stage["rates"], form_rates + phase_rates, strict=True
                    )
                )
                < mp.mpf(2) ** -42
            )


def test_retained_endpoint_pressure_work_and_mean_defects_rebuild(evidence):
    response, _ = evidence
    graph, source = _rebuild(response["after_val"])
    rho = tuple(
        Q(d) / nu for d, nu in zip(source.degrees, source.capacity, strict=True)
    )
    for time, state, record in zip(
        TIMES, response["states"], response["residual_evidence"], strict=True
    ):
        for node in graph:
            set_attr(graph.nodes[node], ALIAS_EPI, float(state[node]))
            set_attr(graph.nodes[node], ALIAS_THETA, float(state[node + 20]))
        current = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
        assert record["time"] == time
        pressure = tuple(
            sum(
                (
                    sin(I(current.phase[other] - current.phase[node]))
                    for other in graph.neighbors(node)
                ),
                I(0),
            )
            / (pi_interval() * graph.degree(node))
            for node in graph
        )
        assert tuple(map(_interval, record["pressure_defects"])) == tuple(
            I(value) - bound
            for value, bound in zip(record["stored_pressures"], pressure, strict=True)
        )
        assert max(
            _interval(value).abs_max for value in record["pressure_defects"]
        ) <= Q(1, 2**46)
        numerical_rates = producer.numerical_field(source)(
            float(time), tuple(map(float, state))
        )
        assert tuple(map(_interval, record["rate_defects"])) == tuple(
            I(Q(value)) - bound
            for value, bound in zip(
                numerical_rates, current.form_rates + current.phase_rates, strict=True
            )
        )
        assert _interval(record["storage_change"]) == current.storage - source.storage
        for key, values, original in (
            ("form_mean_change", current.epi, source.epi),
            ("phase_lift_mean_change", current.phase, source.phase),
        ):
            assert record[key] == sum(
                weight * (value - old)
                for weight, value, old in zip(rho, values, original, strict=True)
            ) / sum(rho)
