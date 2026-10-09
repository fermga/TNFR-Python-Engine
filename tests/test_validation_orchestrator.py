"""Public validation dispatch retains error, context and mutation boundaries."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.config.operator_names import CANONICAL_OPERATOR_NAMES
from tnfr.errors import TNFRValueError
from tnfr.operators import preconditions
from tnfr.validation.invariants import InvariantSeverity, InvariantViolation
from tnfr.validation.validator import TNFRValidationError, TNFRValidator


def _graph():
    graph = nx.Graph()
    graph.add_node(0, EPI=0.2, nu_f=0.5, theta=7.0, source_glyph="AL")
    return graph


def _state(graph):
    return deepcopy(
        (graph.graph, dict(graph.nodes(data=True)), list(graph.edges(data=True)))
    )


@pytest.mark.parametrize("name", sorted(CANONICAL_OPERATOR_NAMES))
def test_all_public_names_dispatch_to_existing_precondition_owner(monkeypatch, name):
    graph = _graph()
    calls = []

    def owner(given_graph, given_node):
        calls.append((given_graph, given_node))

    monkeypatch.setattr(preconditions, f"validate_{name}", owner)
    assert TNFRValidator().validate_operator_preconditions(graph, 0, name.upper())
    assert calls == [(graph, 0)]


@pytest.mark.parametrize("operator", [True, 3, None, "unknown"])
def test_direct_operator_dispatch_handles_invalid_names_in_both_modes(operator):
    validator = TNFRValidator()
    graph = _graph()
    before = _state(graph)
    assert not validator.validate_operator_preconditions(graph, 0, operator, False)
    with pytest.raises(TNFRValueError, match="Unknown operator"):
        validator.validate_operator_preconditions(graph, 0, operator)
    assert _state(graph) == before


def test_real_operator_precondition_failure_is_preserved():
    graph = _graph()
    graph.nodes[0]["EPI"] = 0.9
    validator = TNFRValidator()
    assert not validator.validate_operator_preconditions(graph, 0, "emission", False)
    with pytest.raises(preconditions.OperatorPreconditionError, match="already active"):
        validator.validate_operator_preconditions(graph, 0, "emission")


@pytest.mark.parametrize(
    "arguments",
    [{"operator": "emission"}, {"operator": "emission", "node_id": 0}],
)
def test_operator_request_cannot_succeed_without_its_graph_or_target(arguments):
    validator = TNFRValidator()
    assert not validator.validate(**arguments)["passed"]
    with pytest.raises(TNFRValueError, match="requires graph and node_id"):
        validator.validate(**arguments, raise_on_error=True)


@pytest.mark.parametrize(
    "arguments",
    [
        {"operator": "emission"},
        {"operator": "emission", "node_id": 1},
        {"operator": "emission", "node_id": []},
        {"operator": False, "node_id": 0},
    ],
)
def test_invalid_operator_context_rejects_before_requested_runtime_effects(arguments):
    graph = _graph()
    before = _state(graph)
    result = TNFRValidator().validate(graph, include_runtime=True, **arguments)
    assert not result["passed"]
    assert result["errors"]
    assert result["runtime"] is None
    assert _state(graph) == before


@pytest.mark.parametrize(
    "flag", ["include_invariants", "include_graph_structure", "include_runtime"]
)
def test_nonboolean_check_flag_cannot_select_a_mutating_pass(flag):
    validator = TNFRValidator()
    graph = _graph()
    before = _state(graph)
    arguments = {"include_runtime": True, flag: "false"}
    result = validator.validate(graph, **arguments)
    assert not result["passed"]
    assert result["runtime"] is None
    with pytest.raises(TNFRValueError, match="must be a boolean"):
        validator.validate(graph, **arguments, raise_on_error=True)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "flag",
    ["enable_input_validation", "enable_graph_validation", "enable_runtime_validation"],
)
def test_constructor_rejects_truthy_switch_substitutes(flag):
    with pytest.raises(TNFRValueError, match="must be a boolean"):
        TNFRValidator(**{flag: "false"})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("raise_on_error", ["false", 0, None])
def test_direct_input_error_flag_is_boolean_even_when_validation_is_disabled(
    enabled, raise_on_error
):
    validator = TNFRValidator(enable_input_validation=enabled)
    with pytest.raises(TNFRValueError, match="raise_on_error must be a boolean"):
        validator.validate_inputs(epi=0.5, raise_on_error=raise_on_error)


def test_invalid_graph_is_reported_without_attribute_error():
    validator = TNFRValidator()
    result = validator.validate(object(), raise_on_error=False)
    assert not result["passed"]
    assert "required graph attributes" in result["errors"][0]
    assert not validator.validate_graph_structure(object(), False)["passed"]
    assert not validator.validate_runtime_canonical(object(), False)["passed"]


def test_graph_structure_delegation_preserves_failure_and_does_not_clamp():
    graph = _graph()
    graph.nodes[0]["source_glyph"] = "invalid"
    before = _state(graph)
    validator = TNFRValidator()
    assert not validator.validate_graph_structure(graph, False)["passed"]
    with pytest.raises(TNFRValueError, match="Invalid glyph"):
        validator.validate_graph_structure(graph)
    assert _state(graph) == before


def test_failed_runtime_outcome_raises_or_reports_with_retained_details():
    validator = TNFRValidator()
    graph = _graph()
    graph.nodes[0]["source_glyph"] = "invalid"
    result = validator.validate_runtime_canonical(graph, False)
    assert not result["passed"]
    assert "Invalid glyph" in result["error"]
    assert result["summary"]["errors"]
    assert result["artifacts"]["clamped_nodes"]
    # Runtime validation intentionally retains its existing mutating contract.
    assert graph.nodes[0]["theta"] != 7.0
    with pytest.raises(TNFRValueError, match="Invalid glyph"):
        validator.validate_runtime_canonical(graph, True)
    with pytest.raises(TNFRValueError, match="Invalid glyph"):
        validator.validate(
            graph,
            include_graph_structure=False,
            include_runtime=True,
            include_invariants=False,
            raise_on_error=True,
        )


def test_enabled_runtime_still_applies_the_shared_clamp_pass():
    graph = _graph()
    result = TNFRValidator().validate(
        graph, include_runtime=True, include_invariants=False
    )
    assert result["passed"]
    assert result["runtime"]["passed"]
    assert graph.nodes[0]["theta"] != 7.0


def test_disabled_checks_are_identified_and_preserve_the_graph():
    graph = _graph()
    before = _state(graph)
    validator = TNFRValidator(
        enable_graph_validation=False, enable_runtime_validation=False
    )
    result = validator.validate(graph, include_runtime=True, include_invariants=False)
    assert result["passed"]
    assert result["runtime"]["skipped"]
    assert result["graph_structure"]["skipped"]
    assert _state(graph) == before


def test_raise_on_error_covers_reported_invariant_failures(monkeypatch):
    violation = InvariantViolation(
        invariant_id=4, severity=InvariantSeverity.ERROR, description="invariant failed"
    )
    validator = TNFRValidator()
    monkeypatch.setattr(
        validator, "validate_graph", lambda *args, **kwargs: [violation]
    )
    result = validator.validate(_graph(), include_graph_structure=False)
    assert not result["passed"]
    assert result["invariants"] == [violation]
    with pytest.raises(TNFRValidationError):
        validator.validate(_graph(), include_graph_structure=False, raise_on_error=True)


def _isolated_graph_validator():
    validator = TNFRValidator()
    # Isolate graph dispatch from the independent historical invariant policies.
    validator._invariant_validators = []
    validator.enable_cache(True)
    return validator


def test_live_graph_changes_cannot_reuse_a_previous_success():
    validator = _isolated_graph_validator()
    graph = _graph()
    first = validator.validate_graph(graph)
    assert first == []
    graph.nodes[0]["source_glyph"] = "invalid"
    second = validator.validate_graph(graph)
    assert any("Invalid glyph" in violation.description for violation in second)
    assert first == []
    graph.nodes[0]["source_glyph"] = "AL"
    assert validator.validate_graph(graph) == []
    validator.enable_cache(False)
    validator.clear_cache()


def test_changed_graph_check_selection_always_runs_the_requested_owners():
    validator = _isolated_graph_validator()
    graph = _graph()
    graph.nodes[0]["source_glyph"] = "invalid"
    assert validator.validate_graph(graph, include_graph_validation=False) == []
    assert validator.validate_graph(graph, include_graph_validation=True)
    graph.nodes[0]["source_glyph"] = "AL"
    assert (
        validator.validate_graph(
            graph, include_graph_validation=False, include_runtime_validation=True
        )
        == []
    )
    assert graph.nodes[0]["theta"] != 7.0


def test_custom_validator_changes_and_caller_mutation_cannot_reuse_old_evidence():
    validator = _isolated_graph_validator()
    graph = nx.Graph()
    node = object()
    graph.add_node(node)
    assert validator.validate_graph(graph, include_graph_validation=False) == []
    source = InvariantViolation(
        invariant_id=4,
        severity=InvariantSeverity.ERROR,
        description="custom diagnostic",
        node_id=node,
        actual_value={"samples": [1.0]},
    )

    class CustomCheck:
        invariant_id = 4
        active = True
        calls = 0

        def validate(self, graph):
            self.calls += 1
            return [source] if self.active else []

    check = CustomCheck()
    validator.add_custom_validator(check)
    first = validator.validate_graph(graph, include_graph_validation=False)
    assert first[0].node_id is node
    first[0].description = "caller mutation"
    first[0].actual_value["samples"].append(9.0)
    first.clear()
    second = validator.validate_graph(graph, include_graph_validation=False)
    assert second[0].description == "custom diagnostic"
    assert second[0].actual_value == {"samples": [1.0]}
    assert second[0].node_id is node
    check.active = False
    assert validator.validate_graph(graph, include_graph_validation=False) == []
    assert check.calls == 3
