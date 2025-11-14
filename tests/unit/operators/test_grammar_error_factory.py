from __future__ import annotations

from tnfr.operators.grammar_error_factory import collect_grammar_errors


def test_collect_errors_missing_generator():
    # Sequence starting with stabilizer should trigger U1a initiation error
    seq = ["IL", "SHA"]
    errors = collect_grammar_errors(seq)
    assert errors, "Expected grammar errors for missing generator"
    rules = {e.rule for e in errors}
    assert "U1a" in rules, f"Rules present: {rules}"
    # Invariants for U1a should include 1 (EPI form) and 4 (closure)
    u1a = [e for e in errors if e.rule == "U1a"][0]
    assert 1 in u1a.invariants and 4 in u1a.invariants
