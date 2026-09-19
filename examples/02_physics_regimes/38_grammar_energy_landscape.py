#!/usr/bin/env python3
"""
Example 38 — Grammar Energy Landscape
======================================

Maps the candidate energy functional E through requested operator sequences
while recording the glyph that the runtime actually applies. Incremental
grammar enforcement can replace a rejected standalone request with a fallback.

Physics
-------
The candidate energy E = 0.5 * sum_i [Phi_s^2 + |grad_phi|^2 + K_phi^2
+ J_phi^2 + J_DNFR^2] is measured along grammar-labelled trajectories.
Grammar compliance alone does not prove dE/dt <= 0.

This experiment shows that:
  - A grammar-valid requested word can be checked for energy descent or growth
  - Requested fragments can differ from their runtime-applied glyph traces
  - The policy multipliers from lyapunov.py can be compared with measurements

The energy landscape is the "potential surface" on which operator sequences
trace trajectories. U2, U4 and U1 constrain operator composition; additional
dynamical hypotheses are required for boundedness, controlled excursions and
attractor convergence.

References
----------
- theory/STRUCTURAL_CONSERVATION_THEOREM.md (candidate and proof boundary)
- theory/UNIFIED_GRAMMAR_RULES.md (U1-U6 sequence policies)
- theory/STRUCTURAL_OPERATORS.md (per-operator energy classification)
- src/tnfr/physics/lyapunov.py (operator Lyapunov bounds)
"""

import math
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults
from tnfr.constants.canonical import U6_STRUCTURAL_POTENTIAL_LIMIT
from tnfr.operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from tnfr.operators.grammar import validate_grammar
from tnfr.operators.grammar_types import glyph_function_name
from tnfr.operators.registry import get_operator_class
from tnfr.physics.conservation import (
    capture_conservation_snapshot,
    compute_energy_functional,
    compute_lyapunov_derivative,
    compute_noether_charge,
)

# Optional: legacy nominal multipliers if available
try:
    from tnfr.physics.lyapunov import OPERATOR_LYAPUNOV_BOUNDS, prove_sequence_lyapunov

    _HAS_LYAPUNOV = True
except ImportError:
    _HAS_LYAPUNOV = False

# ── reproducibility (Invariant #6) ──────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ── helpers ──────────────────────────────────────────────────────────────


def _build_graph(n: int = 20, p: float = 0.25) -> nx.Graph:
    """Build a connected random graph with TNFR defaults and non-trivial state."""
    G = nx.erdos_renyi_graph(n, p, seed=SEED)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            u = next(iter(components[i - 1]))
            v = next(iter(components[i]))
            G.add_edge(u, v)
    inject_defaults(G)
    rng = np.random.default_rng(SEED)
    for n_id in G.nodes():
        G.nodes[n_id]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[n_id]["theta"] = G.nodes[n_id]["phase"]
        G.nodes[n_id]["delta_nfr"] = rng.uniform(-0.3, 0.3)
        G.nodes[n_id]["nu_f"] = rng.uniform(0.8, 1.2)
    return G


def _history_codes(G: nx.Graph, node: int) -> tuple[str, ...]:
    """Return the recorded runtime glyph trace without modifying it."""
    history = G.nodes[node].get("glyph_history") or ()
    return tuple(
        str(getattr(item, "value", item)).removeprefix("Glyph.") for item in history
    )


def _apply_and_read_actual_glyph(G: nx.Graph, node: int, op) -> str:
    """Apply one public operator and return the glyph appended by the runtime.

    Operator exceptions deliberately propagate. A successful public call must
    leave an auditable glyph-history change; otherwise the example fails rather
    than inventing an applied label.
    """
    before = _history_codes(G, node)
    op(G, node)
    after = _history_codes(G, node)
    if not after or after == before:
        raise RuntimeError(
            "operator call completed without an auditable glyph-history change"
        )
    return after[-1]


def _operator_name_for_glyph(glyph: str) -> str:
    """Resolve an applied glyph to the public operator class name."""
    function_name = glyph_function_name(glyph)
    return get_operator_class(function_name).__name__


def _apply_op_and_record(G, node, op, requested, history):
    """Apply an operator and record its requested and actual runtime glyphs."""
    snap_before = capture_conservation_snapshot(G)
    actual = _apply_and_read_actual_glyph(G, node, op)
    snap_after = capture_conservation_snapshot(G)
    E = compute_energy_functional(G)
    Q = compute_noether_charge(G)
    lyap = compute_lyapunov_derivative(snap_before, snap_after)
    history.append(
        {
            "step": len(history),
            "requested": requested,
            "actual": actual,
            "E": E,
            "Q": Q,
            "dE_dt": lyap.energy_derivative,
            "lyapunov_stable": lyap.is_stable,
            "fallback": actual != requested,
        }
    )


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Grammar-valid requested trajectory
# ═══════════════════════════════════════════════════════════════════════


def experiment_compliant_trajectory():
    """Execute one grammar-valid requested word and track candidate energy.

    Sequences: Bootstrap [AL, UM, IL] -> Explore [OZ, IL] -> Stabilise [IL, SHA]
    The whole requested word passes the static validator. The runtime trace is
    still recorded separately because standalone calls undergo incremental
    selection one step at a time.
    """
    print("=" * 72)
    print("  EXPERIMENT 1: Grammar-Valid Requested Trajectory")
    print("  Sequence: Bootstrap -> Explore -> Stabilise")
    print("=" * 72)

    G = _build_graph()
    target = 0

    # Full compliant sequence
    ops = [
        ("AL", Emission()),
        ("UM", Coupling()),
        ("IL", Coherence()),  # Bootstrap complete
        ("OZ", Dissonance()),  # Explore start
        ("IL", Coherence()),  # U2: balance destabiliser
        ("IL", Coherence()),  # Extra stabilisation
        ("SHA", Silence()),  # Closure
    ]

    seq_glyphs = [g for g, _ in ops]

    # Validate grammar (pass operator instances, returns bool)
    seq_ops = [op for _, op in ops]
    is_valid = validate_grammar(seq_ops, epi_initial=0.0)
    print(f'\n  Sequence: {" -> ".join(seq_glyphs)}')
    print(f"  Grammar valid: {is_valid}")

    # Legacy-named nominal multiplier diagnostic (if available)
    if _HAS_LYAPUNOV:
        op_names = [
            "Emission",
            "Coupling",
            "Coherence",
            "Dissonance",
            "Coherence",
            "Coherence",
            "Silence",
        ]
        proof = prove_sequence_lyapunov(op_names)
        print(f"  Nominal product <= 1:      {proof.is_net_contractive}")
        print(f"  Nominal multiplier product:{proof.cumulative_product:10.6f}")

    # Run and record
    history = []
    E0 = compute_energy_functional(G)
    Q0 = compute_noether_charge(G)
    history.append(
        {
            "step": 0,
            "requested": "INIT",
            "actual": "INIT",
            "E": E0,
            "Q": Q0,
            "dE_dt": 0.0,
            "lyapunov_stable": True,
            "fallback": False,
        }
    )

    for name, op in ops:
        _apply_op_and_record(G, target, op, name, history)

    # Print trajectory
    print("\n  Actual values below are runtime-recorded glyphs.")
    print(
        f'  {"Step":>5s} {"Request":>8s} {"Actual":>8s} {"E":>12s}'
        f' {"dE":>12s} {"Sign":>9s} {"Q":>12s}'
    )
    print("  " + "-" * 64)
    for h in history:
        sign = "NON-POS" if h["lyapunov_stable"] else "POSITIVE"
        print(
            f"  {h['step']:5d} {h['requested']:>8s} {h['actual']:>8s}"
            f" {h['E']:12.6f} {h['dE_dt']:+12.6f}"
            f" {sign:>9s} {h['Q']:12.6f}"
        )

    # Report the finite energy trend and any runtime replacements.
    energies = [h["E"] for h in history]
    if len(energies) > 2:
        net_change = energies[-1] - energies[0]
        print(f"\n  Net energy change: {net_change:+.6f}")
        print(
            f'  Measured trend: {"NON-INCREASING" if net_change <= 0 else "INCREASING"}'
        )
    replacements = sum(h["fallback"] for h in history)
    print(f"  Runtime replacements: {replacements}")
    print("  The measured sign is not a grammar-wide Lyapunov conclusion.")

    return history


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Multiple Canonical Patterns Compared
# ═══════════════════════════════════════════════════════════════════════


def experiment_pattern_comparison():
    """Compare four requested fragments and their actual runtime traces.

    From STRUCTURAL_OPERATORS.md:
      Bootstrap  = [AL, UM, IL]       (generator -> coupling -> stabilise)
      Stabilise  = [IL, SHA]          (coherence -> silence)
      Explore    = [OZ, ZHIR, IL]     (destabilise -> mutate -> stabilise)
      Propagate  = [RA, UM]           (resonance -> coupling)
    """
    print("\n" + "=" * 72)
    print("  EXPERIMENT 2: Requested Fragment Energy Comparison")
    print("  (runtime fallbacks are shown explicitly)")
    print("=" * 72)

    patterns = {
        "Bootstrap": [("AL", Emission()), ("UM", Coupling()), ("IL", Coherence())],
        "Stabilise": [("IL", Coherence()), ("SHA", Silence())],
        "Explore": [("OZ", Dissonance()), ("IL", Coherence()), ("IL", Coherence())],
        "Propagate": [("RA", Resonance()), ("UM", Coupling())],
    }

    for pname, ops in patterns.items():
        G = _build_graph()
        target = 0
        E_init = compute_energy_functional(G)
        energies = [E_init]
        actual_glyphs: list[str] = []

        for glyph, op in ops:
            actual_glyphs.append(_apply_and_read_actual_glyph(G, target, op))
            energies.append(compute_energy_functional(G))

        requested = " -> ".join(g for g, _ in ops)
        applied = " -> ".join(actual_glyphs)
        dE = energies[-1] - energies[0]
        trend = "DESCENT" if dE <= 0 else "ASCENT"
        print(f"\n  {pname:12s}")
        print(f"    Requested: {requested}")
        print(f"    Applied:   {applied}")
        print(f'    E: {" -> ".join(f"{e:.4f}" for e in energies)}')
        print(f"    Net dE = {dE:+.6f}  ({trend})")

    print("\n  Scope:")
    print("    These named objects are fragments, applied on fresh graphs without")
    print("    surrounding word context. When Applied differs from Requested, the")
    print("    energy change belongs to the fallback trace and says nothing about")
    print("    the rejected requested glyph.")


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Nominal policy multipliers vs measurement
# ═══════════════════════════════════════════════════════════════════════


def experiment_lyapunov_bounds():
    """Compare legacy policy multipliers with measured energy changes.

    The values in lyapunov.py are nominal role multipliers. They do not have
    the units or hypotheses needed to bound the candidate-energy difference
    measured here.
    """
    print("\n" + "=" * 72)
    print("  EXPERIMENT 3: Nominal Policy vs Measured Energy Change")
    print("  (fresh graph per request; actual fallback glyph shown)")
    print("=" * 72)

    if not _HAS_LYAPUNOV:
        print("\n  [SKIPPED: lyapunov module not available]")
        return

    operators = [
        ("Emission", Emission),
        ("Reception", Reception),
        ("Coherence", Coherence),
        ("Dissonance", Dissonance),
        ("Coupling", Coupling),
        ("Resonance", Resonance),
        ("Silence", Silence),
        ("Expansion", Expansion),
        ("Contraction", Contraction),
        ("SelfOrganization", SelfOrganization),
        ("Mutation", Mutation),
        ("Transition", Transition),
        ("Recursivity", Recursivity),
    ]

    print(
        f'\n  {"Requested":20s} {"Applied":22s} {"Policy":14s}'
        f' {"Nominal":>10s} {"Measured dE":>14s} {"Sign":>9s}'
    )
    print("  " + "-" * 96)

    for op_name, cls in operators:
        G = _build_graph()
        target = 0
        E_before = compute_energy_functional(G)

        actual_glyph = _apply_and_read_actual_glyph(G, target, cls())
        actual_name = _operator_name_for_glyph(actual_glyph)
        E_after = compute_energy_functional(G)
        dE = E_after - E_before

        # Compare against the policy attached to what actually executed.
        bound = OPERATOR_LYAPUNOV_BOUNDS.get(actual_name)
        if bound:
            nominal = bound.contraction_rate
            eclass = bound.energy_class.name
        else:
            nominal = float("nan")
            eclass = "UNKNOWN"

        sign = "DESCENT" if dE < -1e-12 else ("ASCENT" if dE > 1e-12 else "FLAT")
        applied_label = f"{actual_glyph}/{actual_name}"
        print(
            f"  {op_name:20s} {applied_label:22s} {eclass:14s}"
            f" {nominal:10.6f} {dE:+14.6f} {sign:>9s}"
        )

    print("\n  Interpretation:")
    print("  - Policy and measured columns are distinct quantities.")
    print("  - A fallback row characterizes the applied operator only.")
    print("  - No quantitative bound is accepted or rejected by this table.")


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Grammar Rule -> Energy Constraint Mapping
# ═══════════════════════════════════════════════════════════════════════


def experiment_grammar_energy_mapping():
    """Annotate requested operators with grammar policies and measured energy.

    Grammar rules constrain sequence form, debt, phase gates, bifurcation
    context, nesting and potential drift. They do not assign a universal
    candidate-energy change to an operator step.
    """
    print("\n" + "=" * 72)
    print("  EXPERIMENT 4: Grammar Policy Annotations and Energy")
    print("  (requested and actually applied glyphs are kept distinct)")
    print("=" * 72)

    G = _build_graph()
    target = 0

    # Track energy through a sequence that exercises multiple rules
    sequence_plan = [
        # Step, Op,         Grammar rules exercised
        ("AL", Emission(), "U1a (generator initiation)"),
        ("EN", Reception(), "continuation; EN has no U3 phase gate"),
        ("UM", Coupling(), "U3 (phase-gated coupling)"),
        ("OZ", Dissonance(), "U2, U4a (destabiliser needs handler)"),
        ("IL", Coherence(), "U2 stabilizer, U4a handler role"),
        ("THOL", SelfOrganization(), "U2 (stabiliser), U4b (transformer)"),
        ("IL", Coherence(), "U2 stabilizer role"),
        ("SHA", Silence(), "U1b (closure)"),
    ]

    print(
        f'\n  {"Step":>5s} {"Req":>5s} {"Actual":>8s} {"E":>12s}'
        f' {"dE":>10s} {"Requested-policy annotation":42s}'
    )
    print("  " + "-" * 91)

    E_prev = compute_energy_functional(G)
    print(
        f"  {'INIT':>5s} {'---':>5s} {'---':>8s} {E_prev:12.6f}"
        f" {'---':>10s} {'Baseline state':42s}"
    )

    replacements: list[tuple[str, str]] = []
    for step, (glyph, op, rule_desc) in enumerate(sequence_plan, start=1):
        actual = _apply_and_read_actual_glyph(G, target, op)
        if actual != glyph:
            replacements.append((glyph, actual))
        E = compute_energy_functional(G)
        dE = E - E_prev
        print(
            f"  {step:5d} {glyph:>5s} {actual:>8s} {E:12.6f}"
            f" {dE:+10.6f} {rule_desc:42s}"
        )
        E_prev = E

    print(f"\n  Runtime replacements: {len(replacements)}")
    for requested, actual in replacements:
        actual_name = _operator_name_for_glyph(actual)
        print(f"    requested {requested} -> applied {actual}/{actual_name}")

    print("\n  Grammar-policy scope:")
    print("  U1: start/closure syntax; no candidate-energy sign follows")
    print("  U2: destabilizer debt and stabilizer compensation policy")
    print("  U3: phase-compatibility gate for UM/RA")
    print("  U4: trigger/handler and transformer-context policy")
    print("  U5: nested-coherence policy; not exercised by this flat sequence")
    print(
        f"  U6: selected max |Delta Phi_s| < {U6_STRUCTURAL_POTENTIAL_LIMIT:.3f}; "
        "not evaluated here"
    )
    print("  The dE column is measured output, not a consequence of these labels.")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    print()
    print("  TNFR Example 38: Grammar Energy Landscape")
    print("  Operator Sequences as Energy Trajectories")
    print("  " + "=" * 50)
    print(f"  Seed: {SEED}  |  Scope: candidate energy and U1-U6 policies")
    print()

    experiment_compliant_trajectory()
    experiment_pattern_comparison()
    experiment_lyapunov_bounds()
    experiment_grammar_energy_mapping()

    print("\n" + "=" * 72)
    print("  SUMMARY: Grammar Energy Landscape Findings")
    print("=" * 72)
    print(
        """
  1. Grammar-Valid Requested Trajectory:
     The first finite run records requested and runtime-applied glyphs for
     one statically valid requested word.
     Its observed sign does not establish a grammar-wide Lyapunov theorem.

  2. Requested Fragment Signatures:
     Each table reports both the requested fragment and the actual runtime
     trace. A fallback trace cannot characterize the rejected request.

  3. Nominal Policy Multipliers:
     Legacy per-operator multipliers and measured candidate-energy changes
     remain separate quantities; this example does not call one a bound.

  4. Grammar Policy Scope:
     The rules constrain the following structural conditions:
       U1 -> trajectory existence and termination
       U2 -> destabilizer/stabilizer composition policy
       U3 -> phase-compatible coupling gate
       U4 -> trigger/handler and recency policy
       U5 -> nested-coherence constraint
       U6 -> selected structural-potential drift monitor

     Candidate-energy changes are measured alongside these conditions;
     no universal implication between them is asserted.

  Operator exceptions propagate, and every successful row is labelled from
  the actual glyph history written by the runtime.
"""
    )


if __name__ == "__main__":
    main()
