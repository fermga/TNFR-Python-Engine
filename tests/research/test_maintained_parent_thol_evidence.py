"""Independent retained-event checks without replaying THOL or any flow.

The prebirth history certificate is rebuilt once. Primitive state, ordinary
operator configuration and detached fields supply premises; stored verdicts do
not. All archived evidence remains immutable throughout these consumers.
"""

from datetime import datetime
from fractions import Fraction as Q
from math import pi, tau
from pathlib import Path

import mpmath
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.constants.canonical import COUPLING_GENTLE, COUPLING_MODERATE
from tnfr.dynamics import integrators, relational
from tnfr.operators import network_stage, word_execution
from tnfr.operators._thol_constants import THOL_CHILD_VF_DAMPING, THOL_SUB_EPI_SCALING
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_u6 import observe_structural_potential_confinement
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.multiscale_coherence import assess_u5_parent_child_coherence
from tnfr.physics.relational_sine_support import (
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
)
from tnfr.research.local_capacity_history import MODEL, encode_result
from tnfr.research.maintained_parent_thol import _signature, load_retained_parent
from tnfr.research.maintained_parent_thol_prediction import (
    NUMERICAL_ALLOWANCE,
    predict_thol_reset,
)
from tnfr.research.retained_graph_state import (
    encode_recorded_container,
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_thol"
PRIOR = ROOT / "docs/assets/local_capacity_history"
PARENT, CHILD = 2, "2_sub_0"


def _read(name, directory=EVIDENCE):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(directory / name, max_bytes=2**25))
    )


def _physical(graph, nodes):
    return tuple(
        tuple(
            Q(get_attr(graph.nodes[node], alias))
            for alias in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF)
        )
        for node in nodes
    )


@pytest.fixture(scope="module", autouse=True)
def no_response_or_proposal():
    def forbidden(*args, **kwargs):
        pytest.fail(
            "retained event consumers cannot execute a producer, proposal or flow"
        )

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (maintained_parent_thol_execution, "execute_network_operator_stage"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "rk4_coupled_step"),
            (integrators, "rk4_coupled_step"),
            (relational, "step_relational_exchange"),
            (word_execution, "execute_network_operator_stage"),
            (network_stage, "execute_self_organization_stage"),
            (SelfOrganization, "_prepare_execution"),
            (SelfOrganization, "_prepare_bifurcation"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


@pytest.fixture(scope="module")
def evidence():
    paths = tuple(
        sorted(
            path
            for directory in (EVIDENCE, PRIOR)
            for path in directory.iterdir()
            if path.is_file()
        )
    )
    before = tuple(file_receipt(path) for path in paths)
    yield _read("response.json"), _read("report.json")
    assert paths == tuple(
        sorted(
            path
            for directory in (EVIDENCE, PRIOR)
            for path in directory.iterdir()
            if path.is_file()
        )
    )
    assert before == tuple(file_receipt(path) for path in paths)


@pytest.fixture(scope="module")
def graphs(evidence):
    response, _ = evidence
    return tuple(
        restore_recorded_graph_state(response[key]).graph
        for key in ("before", "raw_after", "endpoint")
    )


@pytest.fixture(scope="module")
def admitted():
    return load_retained_parent(PRIOR, repository_root=ROOT)


def test_frozen_protocol_sources_and_retained_response_keep_byte_associations(evidence):
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
        {row["path"]: row["sha256"] for row in entries},
        max_bytes=32 * 1024**2,
    )
    # Preserve the archived producer's exact bytes, not an indefinite lock on
    # every current engine source. Current readers below independently rebuild
    # the retained source, history and birth predictions. Their compatibility
    # does not relabel them as the historical implementation or authorize its
    # replay; the frozen archive and response remain unchanged.
    assert attempt["freeze_sha256"] == file_receipt(EVIDENCE / "freeze.json")["sha256"]
    assert report["response"] == file_receipt(
        EVIDENCE / "response.json", label="response.json"
    )
    assert report["checks"] == response["checks"]
    assert report["all_checks_passed"] == all(response["checks"].values())
    assert datetime.fromisoformat(freeze["created_utc"]) <= datetime.fromisoformat(
        attempt["created_utc"]
    )
    assert datetime.fromisoformat(attempt["created_utc"]) <= datetime.fromisoformat(
        report["completed_utc"]
    )
    assert not (EVIDENCE / "failure.json").exists()


def test_genuine_retained_source_certificate_and_independent_prediction_rebuild(
    evidence, graphs, admitted
):
    response, _ = evidence
    before, _, _ = graphs
    original = _signature(before)
    assert original == _signature(admitted.graph)
    assert encode_result(admitted.certificate) == response["source_certificate"]
    assert admitted.receipts == response["retained_source_receipts"]
    assert admitted.certificate.trigger_lower_bound > Q(0.1)
    validate_self_organization_strict(before, PARENT, emit_warnings=False)
    assert validate_candidate(
        before, PARENT, "THOL", sequence_context=admitted.sequence.step(1)
    ).allowed
    assert encode_result(predict_thol_reset(before)) == response["prediction"]
    assert _read("prediction.json")["prediction"] == response["prediction"]
    assert datetime.fromisoformat(
        _read("prediction.json")["created_utc"]
    ) <= datetime.fromisoformat(_read("freeze.json")["created_utc"])
    assert _signature(before) == original


def test_actual_event_changes_only_declared_old_pressure_and_adds_one_isolated_child(
    evidence, graphs
):
    response, _ = evidence
    before, raw, endpoint = graphs
    old_nodes = tuple(before)
    assert old_nodes == tuple(range(20))
    assert tuple(raw) == tuple(endpoint) == old_nodes + (CHILD,)
    assert (
        tuple(before.edges(data=True))
        == tuple(raw.edges(data=True))
        == tuple(endpoint.edges(data=True))
    )
    assert (
        _physical(before, old_nodes)
        == _physical(raw, old_nodes)
        == _physical(endpoint, old_nodes)
    )
    assert raw.degree(CHILD) == endpoint.degree(CHILD) == 0
    observation = observe_structural_acceleration(before, PARENT)
    expected = Q(get_attr(before.nodes[PARENT], ALIAS_DNFR)) + Q(COUPLING_GENTLE) * Q(
        observation.value
    )
    actual = Q(get_attr(raw.nodes[PARENT], ALIAS_DNFR))
    assert abs(actual - expected) <= NUMERICAL_ALLOWANCE
    assert actual == response["parent_raw_pressure"]
    assert response["parent_pressure_increment"] == actual - Q(
        get_attr(before.nodes[PARENT], ALIAS_DNFR)
    )
    assert response["parent_pressure_increment"] > 0
    for node in old_nodes:
        if node != PARENT:
            assert get_attr(raw.nodes[node], ALIAS_DNFR) == get_attr(
                before.nodes[node], ALIAS_DNFR
            )
            assert encode_recorded_container(
                raw.nodes[node]
            ) == encode_recorded_container(before.nodes[node])
            assert encode_recorded_container(
                endpoint.nodes[node]
            ) == encode_recorded_container(before.nodes[node])
        assert get_attr(endpoint.nodes[node], ALIAS_DNFR) == get_attr(
            before.nodes[node], ALIAS_DNFR
        )
    assert (
        get_attr(raw.nodes[CHILD], ALIAS_DNFR)
        == get_attr(endpoint.nodes[CHILD], ALIAS_DNFR)
        == 0
    )
    assert response["stage"]["glyph"] == "THOL"
    assert response["stage"]["nodes_processed"] == 1
    assert response["stage"]["schedule"] == "two_phase_jacobi"
    restored = restore_recorded_graph_state(record_graph_state(endpoint)).graph
    assert _signature(restored) == _signature(endpoint)
    assert tuple(restored) == tuple(range(20)) + (CHILD,)
    assert type(next(reversed(tuple(restored)))) is str


def test_metabolic_construction_matches_independent_high_precision_neighbor_geometry(
    graphs,
):
    before, raw, endpoint = graphs
    mp = mpmath.mp.clone()
    mp.dps = 90

    def exact(value):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator

    neighbors = tuple(before[PARENT])
    forms = tuple(exact(get_attr(before.nodes[n], ALIAS_EPI)) for n in neighbors)
    phases = tuple(exact(get_attr(before.nodes[n], ALIAS_THETA)) for n in neighbors)
    parent_form = exact(get_attr(before.nodes[PARENT], ALIAS_EPI))
    parent_phase = exact(get_attr(before.nodes[PARENT], ALIAS_THETA))
    mean = mp.fsum(forms) / len(forms)
    gradient = mean - parent_form
    mean_cos = mp.fsum(mp.cos(value) for value in phases) / len(phases)
    mean_sin = mp.fsum(mp.sin(value) for value in phases) / len(phases)
    variance = max(mp.mpf(0), min(mp.mpf(1), 1 - mp.sqrt(mean_cos**2 + mean_sin**2)))
    assert all(abs(value - parent_phase) < exact(pi) for value in phases)
    affinity = 1 - mp.fsum(abs(value - parent_phase) for value in phases) / (
        len(phases) * exact(pi)
    )
    ideal = (
        parent_form * exact(THOL_SUB_EPI_SCALING)
        + gradient * exact(COUPLING_MODERATE)
        + variance * exact(COUPLING_GENTLE)
    )
    assert 0 < ideal < 1  # No clipping is active in this retained construction.
    for graph in (raw, endpoint):
        record = graph.nodes[PARENT]["sub_epis"][0]
        signals = record["network_signals"]
        assert signals["neighbor_count"] == len(neighbors)
        for key, expected in (
            ("mean_neighbor_epi", mean),
            ("epi_gradient", gradient),
            ("phase_variance", variance),
            ("coupling_strength_mean", affinity),
        ):
            assert abs(exact(signals[key]) - expected) <= exact(NUMERICAL_ALLOWANCE)
        assert abs(exact(get_attr(graph.nodes[CHILD], ALIAS_EPI)) - ideal) <= exact(
            NUMERICAL_ALLOWANCE
        )
        assert record["epi"] == get_attr(graph.nodes[CHILD], ALIAS_EPI)
        assert record["vf"] == get_attr(graph.nodes[CHILD], ALIAS_VF)
        assert (
            get_attr(graph.nodes[CHILD], ALIAS_VF)
            == get_attr(before.nodes[PARENT], ALIAS_VF) * THOL_CHILD_VF_DAMPING
        )
        assert (
            get_attr(graph.nodes[CHILD], ALIAS_THETA)
            == get_attr(before.nodes[PARENT], ALIAS_THETA) % tau
        )


def test_child_hierarchy_lifecycle_and_parent_history_are_separate_observations(graphs):
    before, raw, endpoint = graphs
    observation = observe_structural_acceleration(before, PARENT)
    for graph in (raw, endpoint):
        parent, child = graph.nodes[PARENT], graph.nodes[CHILD]
        record = parent["sub_epis"][0]
        assert graph.graph["hierarchy"] == {PARENT: [CHILD]}
        assert parent["sub_nodes"] == [CHILD]
        assert child["parent_node"] == PARENT
        assert (
            child["hierarchy_level"]
            == child["_bifurcation_level"]
            == record["bifurcation_level"]
            == 1
        )
        assert child["_hierarchy_path"] == record["hierarchy_path"] == [PARENT]
        assert record["timestamp"] == before.nodes[PARENT]["_operator_step"] + 1 == 2
        assert record["d2_epi"] == observation.value and record["tau"] == 0.1
        assert record["metabolized"] is True and record["node_id"] == CHILD
        assert tuple(parent["glyph_history"]) == ("VAL", "THOL")
        assert node_debt(before.nodes[PARENT]) == 1 and node_debt(parent) == 0
        assert parent["epi_time_history"] == before.nodes[PARENT]["epi_time_history"]
        assert (
            graph.graph["_t"] == before.graph["_t"] == parent["epi_time_history"][-1][0]
        )
        assert child["glyph_history"] == []
        assert child["epi_history"] == [get_attr(child, ALIAS_EPI)]
        assert "epi_time_history" not in child
        assert not observe_structural_acceleration(graph, CHILD).available


def test_complete_successor_fields_and_old_component_means_rebuild_without_new_support(
    evidence, graphs
):
    response, _ = evidence
    before, _, endpoint = graphs
    source = bound_relational_sine_support_exchange(before, reference_model=MODEL)
    successor = bound_relational_sine_frozen_isolates(endpoint, reference_model=MODEL)
    assert encode_result(successor) == response["successor_fields"]
    assert successor.isolate_indices == (20,)
    assert successor.inverse_phase_metric[20] is None
    for row in (successor.pressure, successor.form_rates, successor.phase_rates):
        assert row[20].lo == row[20].hi == 0
    old = bound_relational_sine_support_exchange(
        endpoint.subgraph(tuple(before)).copy(), reference_model=MODEL
    )
    assert source.epi == old.epi and source.phase == old.phase
    assert source.edges == old.edges and source.conductances == old.conductances
    assert source.form_rates == old.form_rates and source.phase_rates == old.phase_rates
    # Identical edge arguments establish an exact zero jump. Subtracting two
    # nondegenerate storage enclosures would only produce an interval around it.
    assert source.form_storage == successor.form_storage
    assert source.phase_storage == successor.phase_storage
    assert response["edge_storage_jump"] == 0
    rho = tuple(
        Q(d) / nu for d, nu in zip(source.degrees, source.capacity, strict=True)
    )
    for before_row, after_row in ((source.epi, old.epi), (source.phase, old.phase)):
        assert (
            sum(w * (x - y) for w, x, y in zip(rho, after_row, before_row, strict=True))
            / sum(rho)
            == 0
        )
    assert response["old_component_mean_changes"] == (0, 0)
    assert response["global_positive_mean_metric_available"] is False


def test_connected_law_u5_observation_and_cross_birth_u6_remain_unsupported(
    evidence, graphs
):
    response, _ = evidence
    before, _, endpoint = graphs
    with pytest.raises(ValueError, match="connected|isolated"):
        bound_relational_sine_support_exchange(endpoint, reference_model=MODEL)
    assert response["connected_reader_rejection"]["type"] == "ValueError"
    assert all(key not in endpoint.nodes[CHILD] for key in ALIAS_DEPI)
    with pytest.raises(ValueError, match="missing required stored dEPI"):
        assess_u5_parent_child_coherence(endpoint, PARENT, alpha=1.0)
    with pytest.raises(ValueError, match="exactly the graph nodes"):
        observe_structural_potential_confinement(
            endpoint,
            compute_structural_potential(before),
            compute_structural_potential(endpoint),
            reference="unsupported_retained_birth_baseline",
        )
    assert response["numerical_u5_observation_available"] is False
    assert response["cross_birth_u6_observation_available"] is False
