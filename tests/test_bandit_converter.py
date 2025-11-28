"""Unit tests for Bandit JSON to SARIF converter.

This test module validates the conversion of Bandit security scan results
from JSON format to SARIF format, ensuring proper mapping of severity levels,
confidence ratings, and result structures.
"""

import pytest
from pathlib import Path
from typing import Any, Dict

# Import converter functions
import sys

tools_path = Path(__file__).parent.parent / "tools"
sys.path.insert(0, str(tools_path))

from bandit_to_sarif import (
    bandit_to_sarif,
    convert_severity,
    convert_confidence,
)


@pytest.fixture
def minimal_bandit_json() -> Dict[str, Any]:
    """Provide a minimal valid Bandit JSON output for testing.

    Returns:
        Dictionary representing minimal Bandit JSON structure
    """
    return {
        "errors": [],
        "generated_at": "2024-01-01T00:00:00Z",
        "metrics": {
            "_totals": {
                "CONFIDENCE.HIGH": 1,
                "SEVERITY.HIGH": 1,
                "loc": 100,
                "nosec": 0,
            }
        },
        "results": [
            {
                "code": "import pickle\n",
                "col_offset": 0,
                "end_col_offset": 13,
                "filename": "src/example.py",
                "issue_confidence": "HIGH",
                "issue_severity": "HIGH",
                "issue_text": "Consider possible security implications associated with pickle module.",
                "line_number": 5,
                "line_range": [5],
                "more_info": "https://bandit.readthedocs.io/en/latest/blacklists/blacklist_imports.html#b403-import-pickle",
                "test_id": "B403",
                "test_name": "blacklist",
            }
        ],
    }


@pytest.fixture
def empty_bandit_json() -> Dict[str, Any]:
    """Provide Bandit JSON with no results.

    Returns:
        Dictionary representing Bandit JSON with empty results
    """
    return {
        "errors": [],
        "generated_at": "2024-01-01T00:00:00Z",
        "metrics": {"_totals": {"loc": 100, "nosec": 0}},
        "results": [],
    }


@pytest.fixture
def multi_severity_bandit_json() -> Dict[str, Any]:
    """Provide Bandit JSON with multiple severity levels.

    Returns:
        Dictionary with LOW, MEDIUM, and HIGH severity results
    """
    return {
        "errors": [],
        "generated_at": "2024-01-01T00:00:00Z",
        "metrics": {},
        "results": [
            {
                "filename": "src/test1.py",
                "issue_confidence": "HIGH",
                "issue_severity": "LOW",
                "issue_text": "Low severity issue",
                "line_number": 10,
                "test_id": "B101",
                "test_name": "assert_used",
            },
            {
                "filename": "src/test2.py",
                "issue_confidence": "MEDIUM",
                "issue_severity": "MEDIUM",
                "issue_text": "Medium severity issue",
                "line_number": 20,
                "test_id": "B201",
                "test_name": "flask_debug_true",
            },
            {
                "filename": "src/test3.py",
                "issue_confidence": "LOW",
                "issue_severity": "HIGH",
                "issue_text": "High severity issue",
                "line_number": 30,
                "test_id": "B301",
                "test_name": "pickle",
            },
        ],
    }


class TestSeverityConversion:
    """Test severity level conversion from Bandit to SARIF."""

    def test_convert_low_severity(self):
        """Test LOW severity maps to 'note'."""
        assert convert_severity("LOW") == "note"

    def test_convert_medium_severity(self):
        """Test MEDIUM severity maps to 'warning'."""
        assert convert_severity("MEDIUM") == "warning"

    def test_convert_high_severity(self):
        """Test HIGH severity maps to 'error'."""
        assert convert_severity("HIGH") == "error"

    def test_convert_unknown_severity(self):
        """Test unknown severity defaults to 'warning'."""
        assert convert_severity("UNKNOWN") == "warning"

    def test_convert_case_insensitive(self):
        """Test severity conversion is case-insensitive."""
        assert convert_severity("low") == "note"
        assert convert_severity("MeDiUm") == "warning"
        assert convert_severity("high") == "error"


class TestConfidenceConversion:
    """Test confidence level conversion from Bandit to descriptive text."""

    def test_convert_low_confidence(self):
        """Test LOW confidence conversion."""
        assert convert_confidence("LOW") == "Low confidence"

    def test_convert_medium_confidence(self):
        """Test MEDIUM confidence conversion."""
        assert convert_confidence("MEDIUM") == "Medium confidence"

    def test_convert_high_confidence(self):
        """Test HIGH confidence conversion."""
        assert convert_confidence("HIGH") == "High confidence"

    def test_convert_unknown_confidence(self):
        """Test unknown confidence handling."""
        assert convert_confidence("UNKNOWN") == "Unknown confidence"


class TestBanditToSARIF:
    """Test complete Bandit JSON to SARIF conversion."""

    def test_minimal_conversion(self, minimal_bandit_json):
        """Test conversion of minimal valid Bandit JSON."""
        sarif = bandit_to_sarif(minimal_bandit_json)

        # Validate SARIF structure
        assert "$schema" in sarif
        assert sarif["version"] == "2.1.0"
        assert "runs" in sarif
        assert len(sarif["runs"]) == 1

        run = sarif["runs"][0]
        assert "tool" in run
        assert "results" in run

        # Validate tool information
        tool = run["tool"]["driver"]
        assert tool["name"] == "Bandit"
        assert "informationUri" in tool
        assert "rules" in tool
        assert len(tool["rules"]) == 1

        # Validate results
        assert len(run["results"]) == 1
        result = run["results"][0]
        assert result["ruleId"] == "B403"
        assert result["level"] == "error"
        assert "message" in result
        assert "locations" in result

    def test_empty_results(self, empty_bandit_json):
        """Test conversion with no security issues found."""
        sarif = bandit_to_sarif(empty_bandit_json)

        assert sarif["version"] == "2.1.0"
        run = sarif["runs"][0]
        assert len(run["results"]) == 0
        assert len(run["tool"]["driver"]["rules"]) == 0

    def test_multi_severity_conversion(self, multi_severity_bandit_json):
        """Test conversion with multiple severity levels."""
        sarif = bandit_to_sarif(multi_severity_bandit_json)

        run = sarif["runs"][0]
        results = run["results"]

        assert len(results) == 3

        # Check severity mapping
        assert results[0]["level"] == "note"  # LOW
        assert results[1]["level"] == "warning"  # MEDIUM
        assert results[2]["level"] == "error"  # HIGH

        # Check confidence mapping
        assert results[0]["properties"]["confidence"] == "High confidence"
        assert results[1]["properties"]["confidence"] == "Medium confidence"
        assert results[2]["properties"]["confidence"] == "Low confidence"

    def test_location_mapping(self, minimal_bandit_json):
        """Test that file locations are properly mapped."""
        sarif = bandit_to_sarif(minimal_bandit_json)

        result = sarif["runs"][0]["results"][0]
        location = result["locations"][0]["physicalLocation"]

        assert location["artifactLocation"]["uri"] == "src/example.py"
        assert location["artifactLocation"]["uriBaseId"] == "%SRCROOT%"
        assert location["region"]["startLine"] == 5
        assert location["region"]["startColumn"] == 1

    def test_code_snippet_inclusion(self, minimal_bandit_json):
        """Test that code snippets are included when available."""
        sarif = bandit_to_sarif(minimal_bandit_json)

        result = sarif["runs"][0]["results"][0]
        region = result["locations"][0]["physicalLocation"]["region"]

        assert "snippet" in region
        assert region["snippet"]["text"] == "import pickle\n"

    def test_rule_deduplication(self, multi_severity_bandit_json):
        """Test that rules are deduplicated across multiple results."""
        sarif = bandit_to_sarif(multi_severity_bandit_json)

        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = [rule["id"] for rule in rules]

        # Each test_id should appear only once in rules
        assert len(rule_ids) == len(set(rule_ids))
        assert len(rules) == 3  # B101, B201, B301

    def test_sarif_schema_reference(self, minimal_bandit_json):
        """Test that SARIF schema reference is correct."""
        sarif = bandit_to_sarif(minimal_bandit_json)

        expected_schema = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
        assert sarif["$schema"] == expected_schema

    def test_column_kind_specified(self, minimal_bandit_json):
        """Test that columnKind is specified in SARIF output."""
        sarif = bandit_to_sarif(minimal_bandit_json)

        run = sarif["runs"][0]
        assert run["columnKind"] == "utf16CodeUnits"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_optional_fields(self):
        """Test handling of missing optional fields in Bandit JSON."""
        minimal_result = {
            "results": [
                {
                    "test_id": "B000",
                    "line_number": 1,
                }
            ]
        }

        sarif = bandit_to_sarif(minimal_result)

        # Should not raise and should provide defaults
        assert len(sarif["runs"][0]["results"]) == 1
        result = sarif["runs"][0]["results"][0]
        assert result["ruleId"] == "B000"
        assert result["level"] == "warning"  # Default

    def test_handles_results_key_missing(self):
        """Test handling when 'results' key is missing."""
        bandit_data = {"errors": [], "metrics": {}}

        sarif = bandit_to_sarif(bandit_data)

        # Should return valid SARIF with no results
        assert len(sarif["runs"][0]["results"]) == 0
