"""Validation reports retain zero-valued structural evidence and plain text."""

import json

import pytest

from tnfr.validation.invariants import InvariantSeverity, InvariantViolation
from tnfr.validation.validator import TNFRValidationError, TNFRValidator


@pytest.fixture
def violation():
    return InvariantViolation(
        invariant_id=2,
        severity=InvariantSeverity.ERROR,
        description="Frequency < minimum",
        node_id=0,
        expected_value=0.0,
        actual_value=False,
        suggestion="Keep <nu_f> in Hz_str",
    )


@pytest.mark.parametrize("as_error", [False, True])
def test_reports_preserve_zero_and_false(violation, as_error):
    reporter = TNFRValidationError([violation]) if as_error else TNFRValidator()
    record = json.loads(reporter.export_to_json([violation]))["violations"][0]
    assert record["node_id"] == 0
    assert record["expected_value"] == "0.0"
    assert record["actual_value"] == "False"
    html = reporter.export_to_html([violation])
    assert "<strong>Node:</strong> 0" in html
    assert "<strong>Expected:</strong> 0.0" in html
    assert "<strong>Actual:</strong> False" in html


def test_report_treats_structural_text_as_text(violation):
    reporter = TNFRValidator()
    html = reporter.export_to_html([violation])
    assert "Frequency &lt; minimum" in html
    assert "Keep &lt;nu_f&gt; in Hz_str" in html
    plain = reporter.generate_report([violation])
    assert "Node: 0" in plain
    assert "Expected: 0.0" in plain
    assert "Actual: False" in plain
