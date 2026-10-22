"""Retained pre-event admission controls; no THOL proposal or flow is executed."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.alias import get_attr, set_dnfr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics import integrators, relational
from tnfr.gamma import GAMMA_REGISTRY, GammaEntry
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.grammar_debt import U2_DEBT_KEY
from tnfr.research import local_capacity_history as producer
from tnfr.research import maintained_parent_thol as study
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
)
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/local_capacity_history"


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=2**25))
    )


@pytest.fixture(scope="module", autouse=True)
def no_response_or_event():
    def forbidden(*args, **kwargs):
        pytest.fail(
            "admission cannot evaluate a numerical response or operator proposal"
        )

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
        # Reuse certificates only for identical admitted inputs. Mutated
        # sample/state controls invoke the real independent certificate.
        original = study.certify_history
        cache = {}

        def certified_once(source, times, states):
            key = repr((source, times, states))
            if key not in cache:
                cache[key] = original(source, times, states)
            return cache[key]

        patch.setattr(study, "certify_history", certified_once)
        yield


@pytest.fixture(scope="module")
def retained():
    paths = tuple(sorted(EVIDENCE.iterdir()))
    before = tuple(file_receipt(path) for path in paths)
    yield _read("response.json"), _read("source.json")["graph"]
    assert tuple(file_receipt(path) for path in paths) == before


def _admit(retained, response=None, preparation=None):
    return study.admit_retained_parent(
        retained[0] if response is None else response,
        frozen_preparation=retained[1] if preparation is None else preparation,
    )


def _edit_graph(response, key, edit):
    graph = restore_recorded_graph_state(response[key]).graph
    edit(graph)
    response[key] = record_graph_state(graph)


def test_retained_loader_rebuilds_law_storage_trigger_and_associations(retained):
    admitted = study.load_retained_parent(EVIDENCE, repository_root=ROOT)
    assert len(admitted.receipts) == 11
    assert tuple(admitted.graph) == tuple(range(20))
    assert admitted.source.capacity[2] == admitted.endpoint.capacity[2] == 128
    assert admitted.source.edges == admitted.endpoint.edges
    assert admitted.source.conductances == admitted.endpoint.conductances
    assert admitted.endpoint == study.capture(admitted.graph)
    assert (
        admitted.endpoint.storage
        == admitted.endpoint.form_storage + admitted.endpoint.phase_storage
    )
    assert admitted.endpoint.continuous_loss == 0
    assert admitted.certificate.endpoint_budget_certified
    assert admitted.certificate.trigger_lower_bound > admitted.threshold
    assert Q(admitted.observation.value) > admitted.threshold
    assert all(
        bound.lo > 0 for bound in admitted.certificate.true_parent_pressure_bounds
    )
    assert (
        max(defect.abs_max for defect in admitted.pressure_defects)
        <= study.PRESSURE_ERROR_BUDGET
    )
    assert admitted.sequence.step(1).index == 1
    assert Q(admitted.graph.graph["_t"]) == study.TIMES[-1]
    assert (
        admitted.graph.nodes[2]["epi_time_history"][-1][0] == admitted.graph.graph["_t"]
    )
    assert len(admitted.graph) == 20  # No child or event was introduced.


def test_cached_certificate_verdict_and_derived_fields_are_not_premises(retained):
    altered = deepcopy(retained[0])
    altered["certificate"] = {"trigger_lower_bound": -999}
    altered["source_admission"] = {"storage": 999}
    altered["checks"] = {"all": False}
    altered["all_checks_passed"] = False
    altered["observation"] = {"value": -999}
    altered["threshold"] = -999
    admission = _admit(retained, altered)
    assert admission.certificate.trigger_lower_bound > Q(1, 10)
    assert admission.observation.value > admission.threshold > 0


@pytest.mark.parametrize("record", ("preparation", "after_val", "endpoint"))
def test_live_gamma_override_rejects_every_consumed_graph(retained, record):
    response = deepcopy(retained[0])
    _edit_graph(
        response, record, lambda graph: graph.graph.update(GAMMA={"type": "harmonic"})
    )
    with pytest.raises(ValueError, match="Gamma"):
        _admit(retained, response)


def test_live_zero_gamma_registry_replacement_cannot_reuse_old_certificate(
    retained, monkeypatch
):
    monkeypatch.setitem(GAMMA_REGISTRY, "none", GammaEntry(lambda *args: 1.0, False))
    with pytest.raises(ValueError, match="live Gamma"):
        _admit(retained)


@pytest.mark.parametrize(
    "edit, message",
    (
        (lambda graph: graph.graph.update(_t=7.0), "clock"),
        (lambda graph: graph.graph.update(_t=True), "structural clock"),
        (lambda graph: graph.graph.update(THOL_BIFURCATION_THRESHOLD=0.0), "defaults"),
        (lambda graph: graph.graph.update(VALIDATE_OPERATOR_PRECONDITIONS=1), "strict"),
        (lambda graph: graph.graph.update(GRAMMAR_REJECTION_MODE="warn"), "raising"),
        (lambda graph: graph.graph.update(EDGE_AWARE_ENABLED=True), "capacity-only"),
        (
            lambda graph: graph.nodes[2].update(glyph_history=["AL", "VAL"]),
            "VAL prefix",
        ),
        (lambda graph: graph.nodes[2].update({U2_DEBT_KEY: True}), "VAL debt"),
        (lambda graph: graph.nodes[2].update({U2_DEBT_KEY: 0}), "VAL debt"),
        (lambda graph: graph.nodes[2].update(nu_f=127.0), "held support"),
        (lambda graph: graph.edges[0, 1].update(weight=0.75), "held support"),
        (lambda graph: graph.nodes[2].update(EPI=0.3125), "final full state"),
        (lambda graph: graph.nodes[3].update(Si=0.75), "undeclared persistent"),
        (lambda graph: set_dnfr(graph, 2, 0.0), "stored pressure"),
        (lambda graph: set_dnfr(graph, 3, True), "stored pressure"),
    ),
)
def test_changed_endpoint_rejects_without_retuning(retained, edit, message):
    response = deepcopy(retained[0])
    _edit_graph(response, "endpoint", edit)
    with pytest.raises((TypeError, ValueError), match=message):
        _admit(retained, response)


@pytest.mark.parametrize("location", ("response", "live"))
def test_history_clock_and_samples_are_associated_independently(retained, location):
    response = deepcopy(retained[0])
    if location == "response":
        rows = list(response["parent_history"])
        rows[1] = rows[0]
        response["parent_history"] = tuple(rows)
    else:

        def edit(graph):
            rows = list(graph.nodes[2]["epi_time_history"])
            rows[1] = rows[0]
            graph.nodes[2]["epi_time_history"] = tuple(rows)

        _edit_graph(response, "endpoint", edit)
    with pytest.raises(ValueError, match="differs from certified samples"):
        _admit(retained, response)


def test_boolean_sample_cannot_equal_an_exact_physical_coordinate(retained):
    response = deepcopy(retained[0])
    rows = [list(row) for row in response["states"]]
    rows[0][20] = False  # Same Python equality as the true zero phase.
    response["states"] = tuple(tuple(row) for row in rows)
    with pytest.raises((TypeError, ValueError), match="recorded_states"):
        _admit(retained, response)


def test_frozen_preparation_lineage_is_checked_without_producer_replay(retained):
    graph = restore_recorded_graph_state(retained[1]).graph
    graph.nodes[3]["Si"] = 0.75
    with pytest.raises(ValueError, match="differs from frozen source"):
        _admit(retained, preparation=record_graph_state(graph))


def test_mutated_source_cannot_reuse_endpoint_certificate(retained):
    response = deepcopy(retained[0])
    _edit_graph(response, "after_val", lambda graph: graph.nodes[2].update(nu_f=64.0))
    _edit_graph(response, "endpoint", lambda graph: graph.nodes[2].update(nu_f=64.0))
    with pytest.raises(ValueError, match="fixed local VAL endpoint"):
        _admit(retained, response)


def test_independent_grammar_rejection_is_not_hidden_by_true_trigger(
    retained, monkeypatch
):
    class Rejected:
        allowed = False

    monkeypatch.setattr(study, "validate_candidate", lambda *args, **kwargs: Rejected())
    with pytest.raises(ValueError, match="live THOL grammar"):
        _admit(retained)


def test_loader_rejects_changed_bytes_before_scientific_admission(monkeypatch):
    original = study.read_bytes_bounded

    def altered(path, **kwargs):
        raw = original(path, **kwargs)
        return raw + b" " if Path(path).name == "response.json" else raw

    monkeypatch.setattr(study, "read_bytes_bounded", altered)
    monkeypatch.setattr(
        study,
        "admit_retained_parent",
        lambda *args, **kwargs: pytest.fail(
            "changed bytes must reject before state admission"
        ),
    )
    with pytest.raises(ValueError, match="artifact bytes differ"):
        study.load_retained_parent(EVIDENCE, repository_root=ROOT)


def test_artifact_path_containment_and_receipt_boolean_rejection(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        study._contained(tmp_path.resolve(), "../outside.json")
    declared = {"path": "x", "bytes": True, "sha256": "a"}
    actual = {"path": "x", "bytes": 1, "sha256": "a"}
    with pytest.raises(ValueError, match="artifact bytes differ"):
        study._check_receipt(actual, declared)


def test_graph_admission_is_read_only_and_retains_stored_pressure(retained):
    response = deepcopy(retained[0])
    before = deepcopy(response)
    admitted = _admit(retained, response)
    assert response == before
    previous = restore_recorded_graph_state(response["endpoint"]).graph
    assert study._signature(admitted.graph) == study._signature(previous)
    assert get_attr(admitted.graph.nodes[2], ALIAS_DNFR) == get_attr(
        previous.nodes[2], ALIAS_DNFR
    )


def test_immediate_revalidation_rebuilds_derived_reports_and_certificate(retained):
    admitted = _admit(retained)
    forged = replace(
        admitted,
        endpoint=replace(admitted.endpoint, storage=study.I(999)),
        certificate=replace(admitted.certificate, trigger_lower_bound=Q(999)),
        threshold=Q(-1),
    )
    fresh = study.validate_live_parent(forged)
    assert fresh.endpoint == admitted.endpoint
    assert fresh.certificate == admitted.certificate
    assert fresh.threshold == admitted.threshold
    assert fresh.graph is admitted.graph


def test_immediate_revalidation_rejects_mutated_live_state(retained):
    admitted = _admit(retained)
    admitted.graph.nodes[2]["Si"] = 0.75
    with pytest.raises(ValueError, match="changed after admission"):
        study.validate_live_parent(admitted)


def test_immediate_revalidation_rechecks_live_gamma_registry(retained, monkeypatch):
    admitted = _admit(retained)
    monkeypatch.setitem(GAMMA_REGISTRY, "none", GammaEntry(lambda *args: 1.0, False))
    with pytest.raises(ValueError, match="live Gamma"):
        study.validate_live_parent(admitted)
