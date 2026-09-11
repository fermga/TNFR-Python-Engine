#!/usr/bin/env python3
"""Example 151 — grammar decisions versus geometric telemetry.

The flat operator grammar, U3 phase admissibility, U6 potential drift, the
five-term structural energy and the auxiliary symplectic substrate answer
different questions. This example evaluates them side by side on a reproducible
graph and records the glyphs that actually execute.

The experiment is deliberately negative: neither a grammar verdict nor one
geometric scalar determines the others. In particular:

* U2 is an operator-history policy, not a theorem that the displayed energies
  decrease for every valid word.
* U6 compares structural potential before and after a declared interval; a
  single-state maximum is reported separately.
* verify_substrate_geometry checks algebraic properties of the auxiliary
  harmonic construction. It is not a grammar validator or a boundedness test.

This finite comparison closes no general grammar/geometry equivalence.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults
from tnfr.constants.canonical import U6_STRUCTURAL_POTENTIAL_LIMIT
from tnfr.operators.definitions import (
    Coherence,
    Coupling,
    Dissonance,
    Emission,
    Reception,
    Silence,
)
from tnfr.operators.grammar_validate import validate_grammar
from tnfr.physics.conservation import compute_energy_functional
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.symplectic_substrate import (
    extract_phase_space_point,
    substrate_hamiltonian,
    verify_substrate_geometry,
)

SEED = 42
OPERATORS = {
    "AL": Emission,
    "EN": Reception,
    "IL": Coherence,
    "OZ": Dissonance,
    "UM": Coupling,
    "SHA": Silence,
}


def build_graph(n: int = 24, p: float = 0.25) -> nx.Graph:
    """Build one connected graph with reproducible nonuniform nodal state."""
    graph = nx.erdos_renyi_graph(n, p, seed=SEED)
    components = list(nx.connected_components(graph))
    for index in range(1, len(components)):
        graph.add_edge(
            next(iter(components[index - 1])),
            next(iter(components[index])),
        )
    inject_defaults(graph)
    rng = np.random.default_rng(SEED)
    for node in graph:
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        graph.nodes[node]["phase"] = phase
        graph.nodes[node]["theta"] = phase
        graph.nodes[node]["delta_nfr"] = float(rng.uniform(-0.3, 0.3))
        graph.nodes[node]["nu_f"] = float(rng.uniform(0.8, 1.2))
    return graph


def _history_codes(graph: nx.Graph, node: Any) -> tuple[str, ...]:
    """Read normalized runtime glyph history without changing it."""
    history = graph.nodes[node].get("glyph_history") or ()
    return tuple(
        str(getattr(item, "value", item)).rsplit(".", 1)[-1].upper()
        for item in history
    )


def _apply_request(graph: nx.Graph, node: Any, requested: str) -> dict[str, str]:
    """Apply one request and expose success, substitution, or failure."""
    before = _history_codes(graph, node)
    try:
        OPERATORS[requested]()(graph, node)
    except Exception as exc:
        after = _history_codes(graph, node)
        return {
            "requested": requested,
            "actual": "NONE",
            "error": (
                f"{type(exc).__name__}: {exc}; history {before} -> {after}"
            ),
        }
    after = _history_codes(graph, node)
    if not after or after == before:
        return {
            "requested": requested,
            "actual": "NONE",
            "error": f"no auditable history append: {before} -> {after}",
        }
    return {"requested": requested, "actual": after[-1], "error": ""}


def _mean_potential_drift(
    before: dict[Any, float],
    after: dict[Any, float],
) -> float:
    nodes = tuple(before)
    return float(np.mean([abs(after[node] - before[node]) for node in nodes]))


def run_case(word: tuple[str, ...], *, sweep: bool) -> dict[str, Any]:
    """Execute a word while retaining every runtime outcome."""
    graph = build_graph()
    phi_before = compute_structural_potential(graph)
    records: list[dict[str, str]] = []
    for requested in word:
        targets = tuple(graph) if sweep else (0,)
        for node in targets:
            record = _apply_request(graph, node, requested)
            records.append(record)
            if record["error"]:
                break

    phi_after = compute_structural_potential(graph)
    point = extract_phase_space_point(graph)
    geometry = verify_substrate_geometry(graph)
    actual = Counter(
        record["actual"] for record in records if record["actual"] != "NONE"
    )
    errors = tuple(record["error"] for record in records if record["error"])
    phi_drift = _mean_potential_drift(phi_before, phi_after)
    return {
        "word": word,
        "flat_valid": validate_grammar(
            [OPERATORS[glyph]() for glyph in word],
            epi_initial=0.0,
        ),
        "actual": ",".join(
            f"{key}:{value}" for key, value in sorted(actual.items())
        ),
        "errors": errors,
        "h_sub": substrate_hamiltonian(point),
        "energy": compute_energy_functional(graph),
        "phi_max": max(abs(value) for value in phi_after.values()),
        "phi_drift": phi_drift,
        "u6_pass": phi_drift < U6_STRUCTURAL_POTENTIAL_LIMIT,
        "substrate_algebra": geometry.all_structures_valid,
    }


def main() -> None:
    print()
    print("=" * 92)
    print("TNFR Example 151: grammar decisions versus geometric telemetry")
    print("=" * 92)
    cases = (
        (("AL", "IL", "SHA"), False),
        (("EN", "IL", "SHA"), False),
        (("AL", "OZ", "SHA"), True),
        (("AL", "OZ", "IL", "SHA"), True),
        (("AL", "OZ", "OZ", "SHA"), True),
        (("AL", "UM", "IL", "SHA"), False),
    )

    print(
        f"{'word':32s} {'flat':>5s} {'H_sub':>10s} {'E5':>10s} "
        f"{'max|Phi|':>10s} {'mean|dPhi|':>11s} {'U6':>5s} {'alg':>5s}"
    )
    print("-" * 92)
    results = []
    for word, sweep in cases:
        result = run_case(word, sweep=sweep)
        results.append(result)
        print(
            f"{str(list(word)):32s} {str(result['flat_valid']):>5s} "
            f"{result['h_sub']:10.3g} {result['energy']:10.3g} "
            f"{result['phi_max']:10.3g} {result['phi_drift']:11.3g} "
            f"{str(result['u6_pass']):>5s} "
            f"{str(result['substrate_algebra']):>5s}"
        )
        print(f"  actual={result['actual'] or 'none'}")
        for error in result["errors"]:
            print(f"  execution failure: {error}")

    valid_energies = {
        round(float(result["energy"]), 12)
        for result in results
        if result["flat_valid"]
    }
    invalid_energies = {
        round(float(result["energy"]), 12)
        for result in results
        if not result["flat_valid"]
    }

    print()
    print("Findings")
    print("--------")
    print(
        "The table keeps four decisions separate: flat symbol/history grammar, "
        "observed U6 drift, finite energy values and auxiliary substrate algebra."
    )
    print(
        "Distinct energy values among flat-valid cases: "
        f"{len(valid_energies)}; among flat-invalid cases: {len(invalid_energies)}."
    )
    print(
        "A runtime failure or fallback is printed explicitly and is never "
        "attributed to the requested glyph."
    )
    print(
        "No equality between grammar validity, energy descent, U6 policy or "
        "substrate-algebra validity is inferred from this finite sample."
    )


if __name__ == "__main__":
    main()
