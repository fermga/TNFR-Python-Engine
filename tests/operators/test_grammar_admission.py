"""Canonical word admission rejects malformed identifiers and false permissions."""

import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators.definitions import (
    Coherence,
    Emission,
    Expansion,
    Mutation,
    Operator,
    Reception,
    Recursivity,
    Silence,
)
from tnfr.operators.grammar_core import GrammarValidator
from tnfr.operators.grammar_error_factory import (
    collect_grammar_errors,
    make_grammar_error,
)
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_memoization import validate_sequence_optimized
from tnfr.operators.grammar_patterns import parse_sequence, validate_sequence
from tnfr.operators.grammar_types import SequenceSyntaxError
from tnfr.operators.grammar_validate import validate_grammar
from tnfr.operators.word_execution import run_network_sequence
from tnfr.structural import run_sequence
from tnfr.validation.aggregator import run_structural_validation


@pytest.mark.parametrize("flag", [False, "false", "0", "off", " NO "])
def test_false_initialization_flags_cannot_grant_word_permission(flag):
    context = {"initial_epi_nonzero": flag}
    assert not validate_sequence(["coherence", "silence"], context=context).passed
    with pytest.raises(SequenceSyntaxError, match="start"):
        parse_sequence(["coherence", "silence"], context=context)
    with pytest.raises(TNFRValueError, match="U1a"):
        ValidatedSequence([Coherence(), Silence()], context=context)
    assert context == {"initial_epi_nonzero": flag}


@pytest.mark.parametrize("flag", [True, "true", "1", "enabled"])
def test_true_initialization_flags_share_the_existing_context_contract(flag):
    context = {"initial_epi_nonzero": flag, "compatibility_profile": "core"}
    assert parse_sequence(["coherence", "silence"], context=context).passed
    assert ValidatedSequence([Coherence(), Silence()], context=context).names == (
        "coherence",
        "silence",
    )
    assert context["initial_epi_nonzero"] == flag


@pytest.mark.parametrize("flag", [False, "false", "0", "off"])
def test_false_diagnostic_flags_cannot_waive_canonical_word_checks(flag):
    result = validate_sequence(
        ["dissonance", "mutation"],
        context={"initial_epi_nonzero": True, "diagnostic": flag},
        compatibility_profile="core",
    )
    assert not result.passed


def test_explicit_true_diagnostic_flag_retains_only_the_exact_probe_waiver():
    context = {"initial_epi_nonzero": "true", "diagnostic": "true"}
    result = validate_sequence(["dissonance", "mutation"], context=context)
    assert result.passed
    assert result.metadata["diagnostic_probe"]
    assert not result.metadata["canonical_word_checked"]
    assert not validate_sequence(
        ["dissonance", "mutation", "silence"], context=context
    ).passed


@pytest.mark.parametrize("flag_name", ["initial_epi_nonzero", "diagnostic"])
@pytest.mark.parametrize("value", ["", "maybe", "notfalse"])
def test_unrecognized_context_flags_reject_before_word_admission(flag_name, value):
    context = {flag_name: value}
    with pytest.raises(ValueError, match="true/false"):
        validate_sequence(["emission", "coherence", "silence"], context=context)
    with pytest.raises(ValueError, match="true/false"):
        ValidatedSequence([Emission(), Coherence(), Silence()], context=context)


@pytest.mark.parametrize("network_word", [False, True])
def test_false_initialization_rejects_execution_without_graph_mutation(network_word):
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.2)
    before = deepcopy((graph.graph, dict(graph.nodes(data=True)), list(graph.edges)))
    context = {"initial_epi_nonzero": "false", "compatibility_profile": "core"}
    with pytest.raises(TNFRValueError):
        if network_word:
            run_network_sequence(graph, ["coherence", "silence"], context=context)
        else:
            run_sequence(graph, 0, [Coherence(), Silence()], context=context)
    assert (graph.graph, dict(graph.nodes(data=True)), list(graph.edges)) == before


@pytest.mark.parametrize(
    "invalid",
    [
        Operator(),
        SimpleNamespace(name="unknown"),
        SimpleNamespace(name=None),
        SimpleNamespace(name=["coherence"]),
        SimpleNamespace(name="coherence", canonical_name="unknown"),
    ],
)
def test_unknown_or_malformed_identifiers_cannot_be_neutral_operations(invalid):
    sequence = [Emission(), invalid, Silence()]
    valid, messages = GrammarValidator().validate(sequence)
    assert not valid
    assert "SYNTAX" in messages[0] and "position 1" in messages[0]
    assert not validate_grammar(sequence)
    assert not validate_sequence_optimized(sequence)[0]
    with pytest.raises(TNFRValueError, match="SYNTAX"):
        ValidatedSequence(sequence)


def test_syntax_preflight_retains_recursivity_depth_metadata():
    recursive = Recursivity(depth=2)
    word = [Emission(), recursive, Silence()]
    valid, messages = GrammarValidator().validate(word)
    assert not valid
    assert any("U5 violated" in message for message in messages)
    with pytest.raises(TNFRValueError, match="U5"):
        ValidatedSequence(word)
    assert GrammarValidator().validate([Emission(), recursive, Coherence(), Silence()])[
        0
    ]
    assert recursive.depth == 2


def _spacing_word():
    return [Expansion(), Reception(), Reception(), Expansion()]


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, float("inf"), float("nan"), "1", 1j, Fraction(1, 10**1000)],
)
def test_temporal_diagnostic_rejects_invalid_capacity(value):
    with pytest.raises((TypeError, ValueError)):
        GrammarValidator.validate_temporal_ordering(_spacing_word(), vf=value)


@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan"), "1", 1j])
def test_temporal_diagnostic_rejects_invalid_multiplier(value):
    with pytest.raises((TypeError, ValueError)):
        GrammarValidator.validate_temporal_ordering(_spacing_word(), k_top=value)


@pytest.mark.parametrize("vf, k_top", [(5e-324, 1.0), (1e308, 5e-324), (1e-308, 1.0)])
def test_temporal_diagnostic_rejects_unrepresentable_spacing(vf, k_top):
    with pytest.raises(ValueError, match="spacing"):
        GrammarValidator.validate_temporal_ordering(_spacing_word(), vf=vf, k_top=k_top)


def test_temporal_diagnostic_retains_the_configured_position_threshold():
    word = _spacing_word()  # Consecutive destabilizers are three positions apart.
    assert not GrammarValidator.validate_temporal_ordering(word)[0]
    assert GrammarValidator.validate_temporal_ordering(word, vf=2.0)[0]
    assert GrammarValidator.validate_temporal_ordering(word, k_top=0.0)[0]


def test_optional_temporal_diagnostic_does_not_change_canonical_word_verdict():
    word = [Emission(), Coherence(), Expansion(), Expansion(), Coherence(), Silence()]
    valid, messages = GrammarValidator(experimental_u6="true").validate(word)
    assert valid
    assert any("U6-EXP WARNING" in message for message in messages)
    valid, messages = GrammarValidator(experimental_u6="false").validate(word, vf=0)
    assert valid
    assert not any("U6-EXP" in message for message in messages)


def test_word_validation_does_not_create_unobserved_telemetry(monkeypatch, capsys):
    from tnfr.physics import fields

    def unexpected_observation(*args, **kwargs):
        pytest.fail("word validation has no observed graph")

    monkeypatch.setattr(fields, "compute_unified_telemetry", unexpected_observation)
    assert validate_grammar([Emission(), Silence()])
    assert not validate_grammar([Coherence(), Silence()])
    assert capsys.readouterr().out == ""


def test_retired_demo_telemetry_argument_rejects_instead_of_inventing_a_graph():
    with pytest.raises(TypeError, match="collect_unified_telemetry"):
        validate_grammar([Emission(), Silence()], collect_unified_telemetry=True)


@pytest.mark.parametrize(
    "middle, expected_rule",
    [(Operator(), "SYNTAX"), (Recursivity(depth=2), "U5"), (Mutation(), "U4b")],
)
def test_structured_errors_and_aggregator_retain_complete_word_rejection(
    middle, expected_rule
):
    sequence = [Emission(), middle, Silence()]
    assert not validate_grammar(sequence)
    errors = collect_grammar_errors(sequence)
    target = next(error for error in errors if error.rule == expected_rule)
    assert target.index == 1
    assert target.candidate == middle.name
    assert target.order == tuple(operator.name for operator in sequence)
    assert target.invariants == ((3, 4) if expected_rule == "U5" else (4,))
    report = run_structural_validation(nx.Graph(), sequence=sequence)
    assert report.status == "invalid"
    assert report.to_dict()["grammar_errors"] == [
        error.to_payload() for error in errors
    ]
    assert sequence[1] is middle
    if expected_rule == "U5":
        assert middle.depth == 2
        assert target.operator_meta["mnemonic"] == "REMESH"


def test_structured_errors_include_recursive_coverage_and_keep_legacy_rule_order():
    sequence = [Emission(), Recursivity(depth=2), Expansion(), Silence()]
    errors = collect_grammar_errors(sequence)
    assert [error.rule for error in errors] == ["U2", "U2-REMESH", "U5"]
    assert errors[1].invariants == errors[0].invariants == (1, 4)
    assert errors[-1].to_structural_error().context["index"] == 1


@pytest.mark.parametrize(
    "sequence",
    [
        [Emission(), Coherence(), Silence()],
        [Emission(), Recursivity(depth=2), Coherence(), Silence()],
        ["AL", "IL", "SHA"],
        ["emission", "coherence", "silence"],
        ["Emission", "Coherence", "Silence"],
        ["al", "il", "sha"],
    ],
)
def test_structured_valid_words_do_not_invent_graph_or_trajectory_evidence(sequence):
    assert collect_grammar_errors(sequence) == []
    report = run_structural_validation(nx.Graph(), sequence=sequence)
    assert report.status == "valid"
    assert report.grammar_errors == []
    assert report.field_metrics["u6_status"] == "not_requested"
    assert not any(report.field_metrics["field_availability"].values())


@pytest.mark.parametrize("invalid", ["unknown", None, [], SimpleNamespace(name=None)])
def test_structured_syntax_rejection_keeps_invalid_identifiers_out_of_metadata(invalid):
    errors = collect_grammar_errors(["AL", invalid, "SHA"])
    assert len(errors) == 1
    assert errors[0].rule == "SYNTAX"
    assert errors[0].index == 1
    assert errors[0].operator_meta is None


@pytest.mark.parametrize("identifier", ["coherence", "Coherence", "IL"])
def test_error_metadata_resolves_all_public_contract_identifiers(identifier):
    error = make_grammar_error(
        rule="U1a",
        candidate=identifier,
        message="word rejection",
        sequence=[identifier],
    )
    assert error.operator_meta["name"] == "Coherence"
    assert error.operator_meta["mnemonic"] == "IL"
    assert error.invariants == (1, 4)
    assert error.candidate == identifier


@pytest.mark.parametrize(
    "passed, message", [(False, "Unfamiliar failure"), (True, "violated")]
)
def test_structured_verdict_uses_shared_booleans_not_message_keywords(
    monkeypatch, passed, message
):
    monkeypatch.setattr(
        GrammarValidator,
        "validate_multiscale_coherence",
        staticmethod(lambda sequence: (passed, message)),
    )
    sequence = [Emission(), Silence()]
    assert validate_grammar(sequence) is passed
    errors = collect_grammar_errors(sequence)
    assert bool(errors) is not passed
    if errors:
        assert errors[0].rule == "U5"
        assert errors[0].message == message


def test_detailed_outcomes_preserve_legacy_order_early_exit_and_immutability():
    validator = GrammarValidator()
    sequence = [Coherence()]
    checks = validator.validate_checks(sequence)
    assert tuple(check.rule for check in checks) == (
        "U1a",
        "U1b",
        "U2",
        "U3",
        "U4a",
        "U4b",
        "U2-REMESH",
        "U5",
    )
    assert not checks[0].passed and not checks[1].passed
    short = validator.validate_checks(sequence, stop_on_first_error=True)
    assert short == checks[:1]
    assert validator.validate(sequence, stop_on_first_error=True) == (
        False,
        ["U1a: " + checks[0].message],
    )
    with pytest.raises(FrozenInstanceError):
        checks[0].passed = True


def test_detailed_optional_spacing_failure_remains_nonblocking():
    sequence = [
        Emission(),
        Coherence(),
        Expansion(),
        Expansion(),
        Coherence(),
        Silence(),
    ]
    validator = GrammarValidator(experimental_u6=True)
    check = validator.validate_checks(sequence)[-1]
    assert check.rule == "U6-EXP" and not check.passed and not check.blocking
    valid, messages = validator.validate(sequence)
    assert valid
    assert messages[-1].startswith(
        "U6-EXP (temporal ordering, experimental): U6-EXP WARNING"
    )


@pytest.mark.parametrize(
    "first_module",
    [
        "tnfr.operators.grammar_core",
        "tnfr.operators.grammar_error_factory",
        "tnfr.validation.aggregator",
    ],
)
def test_structured_grammar_cold_import_orders(first_module, source_tree_environment):
    code = """
import importlib, sys
importlib.import_module(sys.argv[1])
from tnfr.operators.definitions import Emission, Operator, Silence, Recursivity
from tnfr.operators.grammar_error_factory import collect_grammar_errors
from tnfr.operators.grammar_core import GrammarValidator
for middle, rule in ((Operator(), 'SYNTAX'), (Recursivity(depth=2), 'U5')):
    sequence = [Emission(), middle, Silence()]
    if GrammarValidator().validate(sequence)[0]:
        raise RuntimeError('invalid word admitted')
    if rule not in {error.rule for error in collect_grammar_errors(sequence)}:
        raise RuntimeError('structured error lost')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, first_module],
        env=source_tree_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
