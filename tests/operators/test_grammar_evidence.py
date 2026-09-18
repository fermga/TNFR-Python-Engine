"""Grammar reads the existing finite execution proof without inventing control."""

from __future__ import annotations

from fractions import Fraction
import json

import networkx as nx
import pytest

from tnfr.operators.event_runtime import (
    OperatorEventExecutionResult,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.grammar_evidence import (
    StructuralGrammarEvidence,
    assess_structural_grammar_evidence,
)


def _refresh_pure_epi_pressure(graph: nx.Graph) -> None:
    left = float(graph.nodes[0]["EPI"])
    right = float(graph.nodes[1]["EPI"])
    graph.nodes[0]["delta_nfr"] = right - left
    graph.nodes[1]["delta_nfr"] = left - right


def _execution(
    durations: tuple[float, ...] = (0.125, 0.125),
    *,
    sequence: tuple[str, ...] = ("transition",),
    include_stage_certificates: bool = True,
) -> OperatorEventExecutionResult:
    # Reuse the P2 NAV/flow controls of test_operator_event_stage_certificates.
    # These finite owner-conformance checks introduce no new evolution law.
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        NAV_RANDOM=False,
        RANDOM_SEED=7,
        GLYPH_FACTORS={"NAV_eta": 0.25, "NAV_jitter": 0.0},
    )
    if sequence == ("transition",):
        graph.graph["compute_delta_nfr"] = _refresh_pure_epi_pressure
    for node, epi, pressure in zip(
        graph, (0.25, -0.25), (-0.5, 0.5), strict=True
    ):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=pressure,
            latent=False,
            glyph_history=[],
        )
    schedule = build_operator_event_schedule(
        sequence, start_time=0.0, flow_durations=durations
    )
    return execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=include_stage_certificates,
        suppress_birth_warnings=True,
    )


def test_public_observer_keeps_current_snapshot_separate_from_executed_evidence():
    from tnfr.operators import assess_structural_grammar_evidence as public_assess
    from tnfr.operators import observe_grammar

    assert public_assess is assess_structural_grammar_evidence
    execution = _execution()
    current = nx.path_graph(2)
    for node in current:
        current.nodes[node].update(EPI=7.0, nu_f=2.0, theta=0.0)
    report = observe_grammar(
        current, 0, ["IL", "SHA"], execution_evidence=execution,
        contract_satisfied=True, u6_checked=True,
    )
    assert report.sequence_valid
    assert report.structural_evidence.execution is execution
    record = report.as_dict()
    evidence = record["structural_evidence"]
    assert evidence["exact_energy_gain_upper_bound"] == "81/256"
    assert evidence["represented_contraction_certified"]
    assert not evidence["prospective_admission_certified"]
    assert not evidence["tetrad_trajectory_certified"]
    assert not record["verified_contract_postconditions"]
    assert not record["verified_u6"]


@pytest.fixture(scope="module")
def contracting_execution() -> OperatorEventExecutionResult:
    return _execution()


@pytest.mark.parametrize(
    ("durations", "expected_gain", "nonincrease", "contraction"),
    [
        ((0.125, 0.125), Fraction(81, 256), True, True),
        ((0.0, 0.0), Fraction(1), True, False),
        ((1.5, 0.0), Fraction(4), False, False),
    ],
)
def test_same_word_has_state_map_evidence_not_a_label_gain(
    durations: tuple[float, ...],
    expected_gain: Fraction,
    nonincrease: bool,
    contraction: bool,
) -> None:
    evidence = assess_structural_grammar_evidence(_execution(durations))

    assert evidence._proof_fields_are_intact()
    assert evidence.available
    assert evidence.exact_energy_gain_upper_bound == expected_gain
    assert evidence.represented_nonincrease_certified is nonincrease
    assert evidence.represented_contraction_certified is contraction
    assert evidence.unavailable_reasons == ()
    assert "does not prove expansion" in evidence.scope


def test_finite_evidence_never_claims_prospective_or_full_stability(
    contracting_execution: OperatorEventExecutionResult,
) -> None:
    evidence = assess_structural_grammar_evidence(contracting_execution)

    assert evidence.execution is contracting_execution
    assert not evidence.prospective_admission_certified
    assert not evidence.runtime_schedule_global_gain_certified
    assert not evidence.full_multichannel_stability_certified
    assert not evidence.tetrad_trajectory_certified
    assert not evidence.future_or_repeated_schedule_stability_certified
    assert not evidence.solver_accuracy_certified
    record = json.loads(json.dumps(evidence.as_dict()))
    assert record["exact_energy_gain_upper_bound"] == "81/256"
    assert record["tetrad_trajectory_certified"] is False
    assert record == evidence.as_record()
    with pytest.raises(TypeError, match="canonical OperatorEventExecutionResult"):
        assess_structural_grammar_evidence(record)  # type: ignore[arg-type]


def test_missing_execution_and_missing_opt_in_remain_unavailable() -> None:
    for source, reason in (
        (None, "execution_evidence_not_supplied"),
        (
            _execution((0.0, 0.0), include_stage_certificates=False),
            "represented_composition_not_recorded",
        ),
    ):
        evidence = assess_structural_grammar_evidence(source)
        assert evidence._proof_fields_are_intact()
        assert not evidence.available
        assert evidence.exact_energy_gain_upper_bound is None
        assert evidence.represented_nonincrease_certified is None
        assert evidence.represented_contraction_certified is None
        assert evidence.unavailable_reasons == (reason,)


def test_admitted_word_with_stabilizer_does_not_invent_numerical_evidence() -> None:
    execution = _execution(
        (0.0,) * 5,
        sequence=("emission", "coupling", "coherence", "silence"),
    )
    composition = execution.represented_epi_schedule_composition
    assert composition is not None
    evidence = assess_structural_grammar_evidence(execution)

    assert execution._proof_fields_are_intact()
    assert not evidence.available
    assert evidence.exact_energy_gain_upper_bound is None
    assert evidence.represented_nonincrease_certified is None
    assert evidence.unavailable_reasons == composition.failed_conditions
    assert evidence.unavailable_reasons


@pytest.mark.parametrize("source", [True, False, {}, {"certified": True}, "derived"])
def test_untyped_claims_are_not_evidence(source: object) -> None:
    with pytest.raises(TypeError, match="canonical OperatorEventExecutionResult"):
        assess_structural_grammar_evidence(source)  # type: ignore[arg-type]


def test_detached_composition_cannot_replace_complete_execution(
    contracting_execution: OperatorEventExecutionResult,
) -> None:
    with pytest.raises(TypeError, match="canonical OperatorEventExecutionResult"):
        assess_structural_grammar_evidence(
            contracting_execution.represented_epi_schedule_composition  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        StructuralGrammarEvidence(
            contracting_execution, available=True  # type: ignore[call-arg]
        )


def test_nested_proof_mutation_revokes_previously_constructed_readout() -> None:
    execution = _execution((0.0, 0.0))
    evidence = assess_structural_grammar_evidence(execution)
    composition = execution.represented_epi_schedule_composition
    assert composition is not None
    assert evidence.available

    object.__setattr__(composition, "exact_energy_gain_upper_bound", Fraction(0))

    assert not execution._proof_fields_are_intact()
    assert not evidence._proof_fields_are_intact()
    assert not evidence.available
    assert evidence.exact_energy_gain_upper_bound is None
    assert evidence.represented_contraction_certified is None
    assert evidence.unavailable_reasons == ("execution_evidence_integrity_failed",)
    assert evidence.as_dict()["represented_nonincrease_certified"] is None
    with pytest.raises(ValueError, match="proof fields are not intact"):
        assess_structural_grammar_evidence(execution)


def test_report_cannot_be_repointed_to_another_intact_execution(
    contracting_execution: OperatorEventExecutionResult,
) -> None:
    evidence = assess_structural_grammar_evidence(None)
    object.__setattr__(evidence, "execution", contracting_execution)

    assert contracting_execution._proof_fields_are_intact()
    assert not evidence._proof_fields_are_intact()
    assert not evidence.available
    with pytest.raises(ValueError, match="source link is not intact"):
        evidence.__post_init__()
