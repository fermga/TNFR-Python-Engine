"""Validation reports retain zero-valued structural evidence and plain text."""

import json
import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics import BEPIElement
from tnfr.types import Glyph
from tnfr.validation import unified_validation_system as unified
from tnfr.validation.input_validation import ValidationError
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


@pytest.fixture
def input_validator(monkeypatch):
    # Keep this public adapter's configured policy independent of other callers.
    monkeypatch.setattr(
        unified, "_unified_validation_system", unified.TNFRUnifiedValidationSystem()
    )
    return TNFRValidator()


def test_input_bridge_returns_normalized_values_and_checks_every_supplied_kind(
    input_validator,
):
    graph = nx.Graph()
    node = object()
    epi = BEPIElement((-2.0, -2.0), (-2.0, -2.0), (0.0, 1.0))
    result = input_validator.validate_inputs(
        epi=epi,
        vf=Fraction(1, 2),
        theta=-math.pi,
        dnfr=Fraction(-1, 4),
        node_id=node,
        glyph="AL",
        graph=graph,
    )
    assert result == {
        "epi": -2.0,
        "vf": 0.5,
        "theta": math.pi,
        "dnfr": -0.25,
        "node_id": node,
        "glyph": Glyph.AL,
        "graph": graph,
    }
    assert result["node_id"] is node
    assert result["graph"] is graph
    assert isinstance(result["glyph"], Glyph)


@pytest.mark.parametrize("field", ["epi", "vf", "theta", "dnfr"])
@pytest.mark.parametrize(
    "value",
    [True, "0.5", np.array(0.5), Fraction(1, 2**1075), float("nan"), 1j],
)
def test_input_bridge_rejects_coercions_in_both_error_modes(
    input_validator, field, value
):
    with pytest.raises(ValidationError):
        input_validator.validate_inputs(**{field: value})
    result = input_validator.validate_inputs(**{field: value}, raise_on_error=False)
    assert result.keys() == {"error"}
    assert result["error"]


@pytest.mark.parametrize(
    "invalid",
    [{"node_id": []}, {"glyph": "unknown"}, {"graph": object()}],
)
def test_input_bridge_does_not_silently_skip_nonscalar_inputs(input_validator, invalid):
    with pytest.raises(ValidationError):
        input_validator.validate_inputs(**invalid)
    result = input_validator.validate_inputs(epi=-2.0, **invalid, raise_on_error=False)
    assert result["epi"] == -2.0
    assert result["error"]
    assert not (result.keys() & invalid.keys())


def test_input_bridge_rich_epi_is_not_substituted_by_magnitude(input_validator):
    rich = BEPIElement((-2.0, -1.0), (-2.0, -2.0), (0.0, 1.0))
    with pytest.raises(ValidationError):
        input_validator.validate_inputs(epi=rich)


def test_input_bridge_config_is_compatibility_only_and_disabled_mode_stays_empty(
    input_validator,
):
    # A graph's arbitrary keys do not configure this separate input-report owner.
    result = input_validator.validate_inputs(
        vf=1.0, config={"max_structural_frequency": 0.1}
    )
    assert result == {"vf": 1.0}
    assert "error" in input_validator.validate_inputs(
        vf=1001.0, config={"max_structural_frequency": 2000.0}, raise_on_error=False
    )
    disabled = TNFRValidator(enable_input_validation=False)
    assert disabled.validate_inputs(vf=True, glyph="unknown") == {}


def test_comprehensive_validation_consumes_normalized_values_and_input_failures(
    input_validator,
):
    good = input_validator.validate(epi=-0.5, vf=0.5, theta=-math.pi)
    assert good["passed"]
    assert good["inputs"] == {"epi": -0.5, "vf": 0.5, "theta": math.pi}
    for inputs in ({"dnfr": True}, {"node_id": []}):
        bad = input_validator.validate(**inputs)
        assert not bad["passed"]
        assert bad["inputs"]["error"]
        assert bad["errors"]
        with pytest.raises(ValidationError):
            input_validator.validate(**inputs, raise_on_error=True)
