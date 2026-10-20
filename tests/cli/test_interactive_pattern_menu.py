"""The interactive pattern menu forwards registered choices without executing words."""

from types import SimpleNamespace

import pytest

from tnfr.cli.interactive_validator import TNFRInteractiveValidator
from tnfr.operators.grammar_types import StructuralPattern

PATTERNS = tuple(
    item for item in StructuralPattern if item != StructuralPattern.UNKNOWN
)


@pytest.mark.parametrize("selected", PATTERNS)
def test_registered_pattern_selection_reaches_generator(selected, monkeypatch, capsys):
    calls = []
    result = object()
    displayed = []

    def generate(**kwargs):
        calls.append(kwargs)
        return result

    validator = TNFRInteractiveValidator.__new__(TNFRInteractiveValidator)
    validator.generator = SimpleNamespace(generate_for_pattern=generate)
    validator._display_generated_sequence = displayed.append
    monkeypatch.setattr("builtins.input", lambda _: str(PATTERNS.index(selected) + 1))

    validator._generate_by_pattern()

    assert calls == [{"target_pattern": selected.name, "min_health": 0.70}]
    assert displayed == [result]
    output = capsys.readouterr().out
    for index, pattern in enumerate(PATTERNS, 1):
        assert f"{index}. {pattern.name}" in output
    assert "UNKNOWN" not in output
    assert "heuristic, not dynamical guarantees" in output


@pytest.mark.parametrize("choice", ["0", "-1", str(len(PATTERNS) + 1), "invalid", None])
def test_invalid_selection_never_requests_generation(choice, monkeypatch, capsys):
    validator = TNFRInteractiveValidator.__new__(TNFRInteractiveValidator)

    def input_value(_):
        if choice is None:
            raise EOFError
        return choice

    def unexpected_generation(**kwargs):
        pytest.fail("Invalid menu input must not request generation")

    validator.generator = SimpleNamespace(generate_for_pattern=unexpected_generation)
    monkeypatch.setattr("builtins.input", input_value)
    validator._generate_by_pattern()
    assert "Invalid" in capsys.readouterr().out


def test_pattern_explanations_report_actual_labels_and_scope(capsys):
    validator = TNFRInteractiveValidator.__new__(TNFRInteractiveValidator)
    validator._explain_patterns()
    output = capsys.readouterr().out
    for pattern in PATTERNS:
        assert f"{pattern.name}: {pattern.value}" in output
    assert "UNKNOWN" not in output
    assert "not formation or therapeutic effects" in output
