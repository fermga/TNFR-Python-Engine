#!/usr/bin/env python3
"""Convert Bandit JSON output to SARIF format.

This script converts Bandit security scan results from JSON to SARIF
(Static Analysis Results Interchange Format) for GitHub Code Scanning integration.

Usage:
    python tools/bandit_to_sarif.py <input.json> <output.sarif>

Example:
    python tools/bandit_to_sarif.py bandit.json bandit.sarif
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def convert_severity(bandit_severity: str) -> str:
    """Convert Bandit severity to SARIF level.

    Args:
        bandit_severity: Bandit severity string (LOW, MEDIUM, HIGH)

    Returns:
        SARIF level string (note, warning, error)
    """
    severity_map = {
        "LOW": "note",
        "MEDIUM": "warning",
        "HIGH": "error",
    }
    return severity_map.get(bandit_severity.upper(), "warning")


def convert_confidence(bandit_confidence: str) -> str:
    """Convert Bandit confidence to descriptive text.

    Args:
        bandit_confidence: Bandit confidence string (LOW, MEDIUM, HIGH)

    Returns:
        Descriptive confidence text
    """
    confidence_map = {
        "LOW": "Low confidence",
        "MEDIUM": "Medium confidence",
        "HIGH": "High confidence",
    }
    return confidence_map.get(bandit_confidence.upper(), "Unknown confidence")


def bandit_to_sarif(bandit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Bandit JSON output to SARIF format.

    Args:
        bandit_data: Parsed Bandit JSON output

    Returns:
        SARIF-formatted dictionary
    """
    results: List[Dict[str, Any]] = []
    rules: Dict[str, Dict[str, Any]] = {}

    # Process each result from Bandit
    for result in bandit_data.get("results", []):
        test_id = result.get("test_id", "UNKNOWN")
        test_name = result.get("test_name", "unknown_test")

        # Create rule if not already defined
        if test_id not in rules:
            rules[test_id] = {
                "id": test_id,
                "name": test_name,
                "shortDescription": {
                    "text": result.get("issue_text", "Security issue detected")
                },
                "fullDescription": {
                    "text": result.get("issue_text", "Security issue detected")
                },
                "help": {
                    "text": f"More info: https://bandit.readthedocs.io/en/latest/plugins/{test_id.lower()}.html"
                },
                "properties": {
                    "tags": ["security", "bandit"],
                },
            }

        # Create SARIF result
        sarif_result = {
            "ruleId": test_id,
            "level": convert_severity(result.get("issue_severity", "MEDIUM")),
            "message": {"text": result.get("issue_text", "Security issue detected")},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": result.get("filename", "unknown"),
                            "uriBaseId": "%SRCROOT%",
                        },
                        "region": {
                            "startLine": result.get("line_number", 1),
                            "startColumn": 1,
                        },
                    }
                }
            ],
            "properties": {
                "confidence": convert_confidence(
                    result.get("issue_confidence", "MEDIUM")
                ),
            },
        }

        # Add code snippet if available
        code = result.get("code")
        if code:
            sarif_result["locations"][0]["physicalLocation"]["region"]["snippet"] = {
                "text": code
            }

        results.append(sarif_result)

    # Build SARIF structure
    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Bandit",
                        "informationUri": "https://bandit.readthedocs.io/",
                        "version": bandit_data.get("generated_at", "unknown"),
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
                "columnKind": "utf16CodeUnits",
            }
        ],
    }

    return sarif


def main() -> int:
    """Main entry point for the converter."""
    parser = argparse.ArgumentParser(
        description="Convert Bandit JSON output to SARIF format"
    )
    parser.add_argument("input_file", type=Path, help="Input Bandit JSON file")
    parser.add_argument("output_file", type=Path, help="Output SARIF file")

    args = parser.parse_args()

    # Read Bandit JSON
    try:
        with args.input_file.open("r", encoding="utf-8") as f:
            bandit_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        return 1

    # Convert to SARIF
    try:
        sarif_data = bandit_to_sarif(bandit_data)
    except Exception as e:
        print(f"Error: Conversion failed: {e}", file=sys.stderr)
        return 1

    # Write SARIF output
    try:
        with args.output_file.open("w", encoding="utf-8") as f:
            json.dump(sarif_data, f, indent=2)
    except Exception as e:
        print(f"Error: Failed to write output file: {e}", file=sys.stderr)
        return 1

    print(f"Successfully converted {args.input_file} to {args.output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
