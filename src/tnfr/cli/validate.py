#!/usr/bin/env python
"""CLI tool for TNFR validation.

This tool provides two modes:
1. Graph validation: Validate TNFR graphs from files (original behavior)
2. Interactive sequence validation: User-friendly sequence validator (new)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import networkx as nx

from ..validation import (
    InvariantSeverity,
    TNFRValidator,
    configure_validation,
)


def load_graph(filepath: str) -> nx.Graph:
    """Load a graph from file.

    Parameters
    ----------
    filepath : str
        Path to graph file (supports .graphml, .gml, .json).

    Returns
    -------
    nx.Graph
        Loaded graph.
    """
    filepath_obj = Path(filepath)

    if not filepath_obj.exists():
        raise FileNotFoundError(f"Graph file not found: {filepath}")

    # Determine format from extension
    ext = filepath_obj.suffix.lower()

    if ext == ".graphml":
        return nx.read_graphml(filepath)
    elif ext == ".gml":
        return nx.read_gml(filepath)
    elif ext == ".json":
        with open(filepath, "r") as f:
            data = json.load(f)
        return nx.node_link_graph(data)
    else:
        raise ValueError(
            f"Unsupported graph format: {ext}. Use .graphml, .gml, or .json"
        )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TNFR Validation Tool - Validate graphs or sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive sequence validator (NEW!)
  tnfr-validate --interactive
  tnfr-validate -i
  
  # Validate a graph file
  tnfr-validate graph.graphml
  
  # Export to JSON
  tnfr-validate graph.graphml --format json --output report.json
  
  # Export to HTML
  tnfr-validate graph.graphml --format html --output report.html
  
  # Set minimum severity to WARNING
  tnfr-validate graph.graphml --min-severity warning
  
  # Enable caching for multiple validations
  tnfr-validate graph.graphml --cache
        """,
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive sequence validator",
    )

    parser.add_argument(
        "graph_file",
        nargs="?",
        help="Path to graph file (.graphml, .gml, or .json). Required unless using --interactive mode.",
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout for text, or auto-generated for json/html)",
    )

    parser.add_argument(
        "--min-severity",
        choices=["info", "warning", "error", "critical"],
        default="error",
        help="Minimum severity to report (default: error)",
    )

    parser.add_argument(
        "--cache", action="store_true", help="Enable validation result caching"
    )

    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic sequence validation",
    )

    parser.add_argument(
        "--phase-threshold",
        type=float,
        help="Phase coupling threshold in radians (default: π/2)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for interactive validator (deterministic generation)",
    )

    args = parser.parse_args()

    # Handle interactive mode
    if args.interactive:
        from .interactive_validator import run_interactive_validator
        return run_interactive_validator(seed=args.seed)

    # Require graph_file for non-interactive mode
    if not args.graph_file:
        parser.error("graph_file is required when not using --interactive mode")

    # Configure validation
    severity_map = {
        "info": InvariantSeverity.INFO,
        "warning": InvariantSeverity.WARNING,
        "error": InvariantSeverity.ERROR,
        "critical": InvariantSeverity.CRITICAL,
    }

    configure_validation(
        validate_invariants=True,
        enable_semantic_validation=not args.no_semantic,
        min_severity=severity_map[args.min_severity],
    )

    if args.verbose:
        print(f"Loading graph from: {args.graph_file}")

    try:
        # Load graph
        graph = load_graph(args.graph_file)

        if args.verbose:
            print(
                f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )

        # Create validator
        if args.phase_threshold:
            validator = TNFRValidator(phase_coupling_threshold=args.phase_threshold)
        else:
            validator = TNFRValidator()

        if args.cache:
            validator.enable_cache(True)

        # Validate
        if args.verbose:
            print("Running TNFR validation...")

        violations = validator.validate_graph(graph)

        # Generate output
        if args.format == "text":
            output = validator.generate_report(violations)
        elif args.format == "json":
            output = validator.export_to_json(violations)
        elif args.format == "html":
            output = validator.export_to_html(violations)

        # Write output
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                f.write(output)
            print(f"Report written to: {output_path}")
        else:
            # Print to stdout
            print(output)

        # Exit code based on violations
        if violations:
            # Check if there are any ERROR or CRITICAL violations
            has_errors = any(
                v.severity in [InvariantSeverity.ERROR, InvariantSeverity.CRITICAL]
                for v in violations
            )
            sys.exit(1 if has_errors else 0)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
