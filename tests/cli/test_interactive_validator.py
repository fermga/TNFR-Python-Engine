"""The terminal adapter reads registered contracts and current grammar roles."""

from types import SimpleNamespace

from tnfr.cli import interactive_validator
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer


def test_help_uses_contract_owner_instead_of_a_second_operator_catalog(
    monkeypatch, capsys
):
    contract = SimpleNamespace(
        name="supplied_token", glyph="TEST", purpose="Supplied contract scope."
    )
    monkeypatch.setattr(interactive_validator, "iter_contracts", lambda: (contract,))
    validator = interactive_validator.TNFRInteractiveValidator.__new__(
        interactive_validator.TNFRInteractiveValidator
    )
    validator._show_help()
    output = capsys.readouterr().out
    assert "supplied_token" in output
    assert "(TEST) - Supplied contract scope." in output
    assert "emission" not in output
    assert "not stability or formation proofs" in output


def test_u1_advice_distinguishes_initiators_and_closures(capsys):
    validator = interactive_validator.TNFRInteractiveValidator.__new__(
        interactive_validator.TNFRInteractiveValidator
    )
    validator._suggest_fixes(["reception", "coherence"], None)
    output = capsys.readouterr().out
    assert "Standalone U1 initiators: emission, recursivity, transition" in output
    assert (
        "Standalone U1 closures: dissonance, recursivity, silence, transition" in output
    )
    assert "These roles alone do not satisfy every grammar or live-state rule" in output
    assert "starts with emission or reception" not in output


def test_suggestions_follow_shared_roles_when_they_change(monkeypatch, capsys):
    monkeypatch.setattr(
        interactive_validator, "VALID_START_OPERATORS", {"supplied_start"}
    )
    monkeypatch.setattr(interactive_validator, "VALID_END_OPERATORS", {"supplied_end"})
    validator = interactive_validator.TNFRInteractiveValidator.__new__(
        interactive_validator.TNFRInteractiveValidator
    )
    validator._suggest_fixes([], None)
    output = capsys.readouterr().out
    assert "Standalone U1 initiators: supplied_start" in output
    assert "Standalone U1 closures: supplied_end" in output


def test_health_ending_advice_preserves_rubric_but_respects_standalone_u1():
    # This word triggers ending advice while its 0.1+0.3+0.3 score stays fixed.
    metrics = SequenceHealthAnalyzer().analyze_health(["emission", "transition"])
    assert metrics.sustainability_index == 0.7
    advice = next(item for item in metrics.recommendations if "U1 closure:" in item)
    assert "U1 closure: silence;" in advice
    assert "other grammar and live-state conditions still apply" in advice
    assert "coherence," not in advice
