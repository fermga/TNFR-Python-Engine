#!/usr/bin/env python3
"""
Example 37 — Operator-Tetrad Response Diagnostics
=================================================

Measures how the structural-field tetrad changes after one seeded set of
operator requests. The runtime may replace a rejected request with a
grammar-selected fallback, so every table keeps the requested and actually
recorded glyphs separate.

Physics
-------
The canonical operators reorganize the EPI, nu_f, DELTA_NFR, or phase channel
of the nodal equation dEPI/dt = nu_f * DELTA_NFR(t). Recomputing the tetrad
then supplies complementary read-outs:

    Phi_s     -> global stability     (0th order, harmonic accumulation)
    |grad_phi| -> local stress        (1st order, phase derivative)
    K_phi     -> geometric torsion    (2nd order, curvature)
    xi_C      -> correlation range    (non-local, exponential decay)

The resulting response matrix is descriptive for this graph, node, seed, and
single-call protocol. It does not prove a universal causal fingerprint or that
the tetrad reconstructs the complete graph state.

References
----------
- theory/STRUCTURAL_OPERATORS.md (operator contracts and nominal energy model)
- theory/UNIFIED_GRAMMAR_RULES.md (U1-U6 derivations)
- src/tnfr/physics/fields.py (tetrad computation)
"""

import math
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults
from tnfr.constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
    PHI_S_VON_KOCH_THRESHOLD,
    PI,
)
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
from tnfr.physics.conservation import compute_energy_functional, compute_noether_charge
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
)

# ── reproducibility (Invariant #6) ──────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ── helpers ──────────────────────────────────────────────────────────────


def _build_graph(n: int = 20, p: float = 0.25) -> nx.Graph:
    """Build a random TNFR graph with reproducible seed and non-trivial state."""
    G = nx.erdos_renyi_graph(n, p, seed=SEED)
    # Ensure connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            u = next(iter(components[i - 1]))
            v = next(iter(components[i]))
            G.add_edge(u, v)
    inject_defaults(G)
    # Randomise node attributes to produce non-trivial tetrad fields
    rng = np.random.default_rng(SEED)
    for n_id in G.nodes():
        G.nodes[n_id]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[n_id]["theta"] = G.nodes[n_id]["phase"]
        G.nodes[n_id]["delta_nfr"] = rng.uniform(-0.3, 0.3)
        G.nodes[n_id]["nu_f"] = rng.uniform(0.8, 1.2)
    return G


def _snapshot_tetrad(G: nx.Graph) -> dict[str, float]:
    """Capture tetrad field summary statistics."""
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    xi_c = estimate_coherence_length(G)

    phi_s_values = np.asarray(list(phi_s.values()), dtype=float)
    grad_values = np.asarray(list(grad_phi.values()), dtype=float)
    curvature_values = np.asarray(list(k_phi.values()), dtype=float)

    return {
        "Phi_s_mean": float(np.mean(phi_s_values)),
        "Phi_s_max": float(np.max(np.abs(phi_s_values))),
        "grad_phi_mean": float(np.mean(grad_values)),
        "grad_phi_max": float(np.max(np.abs(grad_values))),
        "K_phi_mean": float(np.mean(np.abs(curvature_values))),
        "K_phi_max": float(np.max(np.abs(curvature_values))),
        "xi_C": float(xi_c),
    }


def _deep_copy_graph(G: nx.Graph) -> nx.Graph:
    """Deep copy to reset state between operator trials."""
    import copy

    return copy.deepcopy(G)


def _history_codes(G: nx.Graph, node: int) -> tuple[str, ...]:
    """Return the normalized runtime glyph history without changing it."""
    history = G.nodes[node].get("glyph_history") or ()
    return tuple(
        str(getattr(item, "value", item)).rsplit(".", 1)[-1].upper()
        for item in history
    )


def _apply_with_trace(G: nx.Graph, node: int, requested: str, op) -> dict[str, object]:
    """Apply one request and return its authoritative runtime trace.

    Operator failures propagate with the request and before/after histories in
    the error message. A successful call must append an auditable history item;
    the last recorded glyph identifies the transformation actually measured.
    """
    before = _history_codes(G, node)
    try:
        op(G, node)
    except Exception as exc:
        after = _history_codes(G, node)
        raise RuntimeError(
            f"request {requested} failed; history {before} -> {after}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    after = _history_codes(G, node)
    if not after or after == before:
        raise RuntimeError(
            f"request {requested} completed without an auditable history "
            f"append: {before} -> {after}"
        )

    actual = after[-1]
    return {
        "requested": requested,
        "actual": actual,
        "fallback": actual != requested,
        "history_before": before,
        "history_after": after,
    }


# ── canonical operator catalogue ─────────────────────────────────────────

ALL_OPERATORS = [
    ("AL", "Emission", Emission),
    ("EN", "Reception", Reception),
    ("IL", "Coherence", Coherence),
    ("OZ", "Dissonance", Dissonance),
    ("UM", "Coupling", Coupling),
    ("RA", "Resonance", Resonance),
    ("SHA", "Silence", Silence),
    ("VAL", "Expansion", Expansion),
    ("NUL", "Contraction", Contraction),
    ("THOL", "SelfOrganization", SelfOrganization),
    ("ZHIR", "Mutation", Mutation),
    ("NAV", "Transition", Transition),
    ("REMESH", "Recursivity", Recursivity),
]


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: One-Call Tetrad Response Matrix
# ═══════════════════════════════════════════════════════════════════════


def experiment_operator_fingerprints():
    """Issue each request on an identical graph copy and measure tetrad delta.

    This builds a 13x5 response matrix: requests (rows) x field summaries
    (columns). Each cell is the relative change observed after the actual
    runtime glyph recorded in ``glyph_history``.
    """
    print("=" * 72)
    print("  EXPERIMENT 1: One-Call Tetrad Response Matrix")
    print("  (requested and runtime-recorded glyphs are kept separate)")
    print("=" * 72)

    G_base = _build_graph(n=20, p=0.25)
    baseline = _snapshot_tetrad(G_base)
    target_node = 0

    print(f"\nBaseline tetrad (random graph, N=20, p=0.25, seed={SEED}):")
    for k, v in baseline.items():
        print(f"  {k:20s} = {v:+.6f}")

    field_names = ["Phi_s_mean", "Phi_s_max", "grad_phi_mean", "K_phi_mean", "xi_C"]

    results = {}
    for glyph, name, cls in ALL_OPERATORS:
        G = _deep_copy_graph(G_base)
        trace = _apply_with_trace(G, target_node, glyph, cls())
        after = _snapshot_tetrad(G)
        deltas = {}
        for field in field_names:
            before_value = baseline[field]
            after_value = after[field]
            # Relative change (percent); avoid division by zero.
            if abs(before_value) > 1e-12:
                deltas[field] = (
                    (after_value - before_value) / abs(before_value) * 100.0
                )
            else:
                deltas[field] = (after_value - before_value) * 100.0
        results[glyph] = {"name": name, "deltas": deltas, **trace}

    # ── print response matrix ──
    print("\n  One-Call Response Matrix (% change per field)")
    print("  " + "-" * 80)
    header = f"  {'Request':7s} {'Actual':7s} {'Name':18s}"
    for f in field_names:
        header += f" {f:>12s}"
    print(header)
    print("  " + "-" * 80)

    for glyph, name, _ in ALL_OPERATORS:
        r = results[glyph]
        row = f"  {glyph:7s} {str(r['actual']):7s} {name:18s}"
        for f in field_names:
            d = r["deltas"][f]
            row += f" {d:+11.3f}%"
        print(row)

    print("\n  Runtime substitutions and recorded histories:")
    substitutions = [r for r in results.values() if r["fallback"]]
    if not substitutions:
        print("  none")
    for result in substitutions:
        print(
            f"  requested {result['requested']} -> actual {result['actual']}; "
            f"history {result['history_before']} -> {result['history_after']}"
        )

    # ── report the largest sampled response ──
    print("\n  Largest Sampled Response Per Request:")
    print("  " + "-" * 58)
    for glyph, name, _ in ALL_OPERATORS:
        r = results[glyph]
        d = r["deltas"]
        dominant = max(field_names, key=lambda f: abs(d[f]))
        if abs(d[dominant]) <= 1e-12:
            response = "no sampled change"
        else:
            response = f"{d[dominant]:+.2f}%"
        print(
            f"  {glyph:7s} -> {str(r['actual']):7s}: "
            f"{dominant} ({response})"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Requested U2 Role and Candidate Energy
# ═══════════════════════════════════════════════════════════════════════


def experiment_energy_signature():
    """Compare candidate energy after requests from the two U2 roles.

    U2 classifies composition debt. It does not prove that every stabilizer
    request decreases this five-term structural-energy candidate or that every
    destabilizer request increases it.
    """
    print("\n" + "=" * 72)
    print("  EXPERIMENT 2: Requested U2 Role vs Sampled Energy Change")
    print("  (one-call observations, not a U2 Lyapunov proof)")
    print("=" * 72)

    stabilisers = [("IL", Coherence), ("THOL", SelfOrganization)]
    destabilisers = [("OZ", Dissonance), ("ZHIR", Mutation), ("VAL", Expansion)]

    G_base = _build_graph(n=20, p=0.25)
    target = 0
    observations = []

    for label, ops in [("STABILISERS", stabilisers), ("DESTABILISERS", destabilisers)]:
        print(f"\n  {label} (requested role):")
        print(
            f'  {"Request":7s} {"Actual":7s} {"E_before":>12s}'
            f' {"E_after":>12s} {"Delta_E":>12s} {"sign":>9s}'
        )
        print("  " + "-" * 70)

        for glyph, cls in ops:
            G = _deep_copy_graph(G_base)
            E_before = compute_energy_functional(G)
            trace = _apply_with_trace(G, target, glyph, cls())
            E_after = compute_energy_functional(G)
            dE = E_after - E_before
            sign = "DECREASE" if dE < 0 else ("INCREASE" if dE > 0 else "ZERO")
            observation = {
                "requested_role": label,
                "E_before": E_before,
                "E_after": E_after,
                "delta_E": dE,
                **trace,
            }
            observations.append(observation)
            print(
                f"  {glyph:7s} {str(trace['actual']):7s} {E_before:12.6f}"
                f" {E_after:12.6f} {dE:+12.6f} {sign:>9s}"
            )
            if trace["fallback"]:
                print(
                    f"           history {trace['history_before']} -> "
                    f"{trace['history_after']}"
                )

    print("\n  Scope:")
    print("  U2 labels classify the requests; the measured delta belongs to the")
    print("  actual glyph and this seeded graph snapshot. Sequence-level stability")
    print("  requires trajectory evidence or a model-specific proof.")
    return observations


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Exact Phase Bounds and Selected Policies
# ═══════════════════════════════════════════════════════════════════════


def experiment_tetrad_safety():
    """Compare exact phase bounds and selected policies along one word.

    ``|grad_phi| <= pi`` and ``|K_phi| <= pi`` are exact wrapped-angle
    bounds. The Phi_s magnitude cut, pi/16 gradient alert, and 0.9*pi
    curvature margin are selected telemetry policies. Grammar compliance alone
    does not imply that these selected policies will hold.
    """
    print("\n" + "=" * 72)
    print("  EXPERIMENT 3: Exact Phase Bounds and Selected Tetrad Policies")
    print("  (one grammar-compliant requested word; actual trace recorded)")
    print("=" * 72)

    G = _build_graph(n=20, p=0.25)
    target = 0

    # Static grammar-valid word: generator AL, phase operation UM,
    # stabilizer IL, and closure SHA.
    sequence = [
        ("AL", Emission()),
        ("UM", Coupling()),
        ("IL", Coherence()),
        ("SHA", Silence()),
    ]

    def policy_flags(snapshot):
        selected = (
            snapshot["Phi_s_max"] < PHI_S_VON_KOCH_THRESHOLD
            and snapshot["grad_phi_max"] < GRAD_PHI_CANONICAL_THRESHOLD
            and snapshot["K_phi_max"] < K_PHI_CANONICAL_THRESHOLD
        )
        exact_phase = (
            snapshot["grad_phi_max"] <= PI + 1e-12
            and snapshot["K_phi_max"] <= PI + 1e-12
        )
        return selected, exact_phase

    def print_row(step, requested, actual, snapshot):
        phi_status = (
            "OK"
            if snapshot["Phi_s_max"] < PHI_S_VON_KOCH_THRESHOLD
            else "WARN"
        )
        grad_status = (
            "OK"
            if snapshot["grad_phi_max"] < GRAD_PHI_CANONICAL_THRESHOLD
            else "WARN"
        )
        curvature_status = (
            "OK"
            if snapshot["K_phi_max"] < K_PHI_CANONICAL_THRESHOLD
            else "WARN"
        )
        print(
            f"  {step:>4} {requested:>5s} {actual:>5s}"
            f" {snapshot['Phi_s_max']:10.4f} {phi_status:>5s}"
            f" {snapshot['grad_phi_max']:10.4f} {grad_status:>5s}"
            f" {snapshot['K_phi_max']:10.4f} {curvature_status:>5s}"
            f" {snapshot['xi_C']:10.4f}"
        )

    requested_word = " -> ".join(requested for requested, _ in sequence)
    print(f"\n  Requested word: {requested_word}")
    print(
        f'\n  {"Step":>4s} {"Req":>5s} {"Act":>5s}'
        f' {"Phi_s_max":>10s} {"pol":>5s}'
        f' {"grad_max":>10s} {"pol":>5s}'
        f' {"K_max":>10s} {"pol":>5s} {"xi_C":>10s}'
    )
    print("  " + "-" * 87)

    snapshot = _snapshot_tetrad(G)
    selected_ok, exact_ok = policy_flags(snapshot)
    all_selected_ok = selected_ok
    all_exact_ok = exact_ok
    print_row("INIT", "---", "---", snapshot)

    traces = []
    for index, (requested, op) in enumerate(sequence, start=1):
        trace = _apply_with_trace(G, target, requested, op)
        traces.append(trace)
        snapshot = _snapshot_tetrad(G)
        selected_ok, exact_ok = policy_flags(snapshot)
        all_selected_ok = all_selected_ok and selected_ok
        all_exact_ok = all_exact_ok and exact_ok
        print_row(str(index), requested, str(trace["actual"]), snapshot)

    print("\n  Recorded runtime history:")
    for trace in traces:
        print(
            f"  requested {trace['requested']} -> actual {trace['actual']}; "
            f"history {trace['history_before']} -> {trace['history_after']}"
        )

    print(
        f'\n  Exact wrapped-phase bounds respected: {"YES" if all_exact_ok else "NO"}'
    )
    print(
        "  All selected policy margins satisfied: "
        f'{"YES" if all_selected_ok else "NO"}'
    )
    print(
        f"    |Phi_s| policy = {PHI_S_VON_KOCH_THRESHOLD:.4f} (pi/4)"
    )
    print(
        f"    |grad_phi| alert = {GRAD_PHI_CANONICAL_THRESHOLD:.4f} (pi/16); "
        f"exact bound = {PI:.4f}"
    )
    print(
        f"    |K_phi| margin = {K_PHI_CANONICAL_THRESHOLD:.4f} (0.9*pi); "
        f"exact bound = {PI:.4f}"
    )
    print("    xi_C is state-dependent and has no universal bound in this example.")
    return {
        "traces": traces,
        "selected_policies_satisfied": all_selected_ok,
        "exact_phase_bounds_respected": all_exact_ok,
    }


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Noether-Like Charge Telemetry
# ═══════════════════════════════════════════════════════════════════════


def experiment_charge_telemetry():
    """Track Noether charge Q = sum(Phi_s + K_phi) through operator steps.

    The finite changes are trajectory observations. Operator labels alone do
    not imply zero charge drift; a conservation assessment also needs the
    implemented current-divergence and source balance.
    """
    print("\n" + "=" * 72)
    print("  EXPERIMENT 4: Noether-Like Charge Telemetry")
    print("  (finite differences along the actual runtime trace)")
    print("=" * 72)

    G = _build_graph(n=20, p=0.25)
    target = 0

    # Grammar-compliant sequence: Bootstrap + Explore + Stabilise
    # [AL, UM, IL, OZ, IL, SHA]
    steps = [
        ("AL", Emission()),
        ("UM", Coupling()),
        ("IL", Coherence()),
        ("OZ", Dissonance()),
        ("IL", Coherence()),
        ("SHA", Silence()),
    ]

    print(
        f'\n  {"Step":6s} {"Req":7s} {"Actual":7s} {"Q":>14s}'
        f' {"E candidate":>14s} {"dQ":>10s}'
    )
    print("  " + "-" * 70)

    Q_prev = compute_noether_charge(G)
    energy = compute_energy_functional(G)
    print(
        f"  {'INIT':6s} {'---':7s} {'---':7s} {Q_prev:14.6f}"
        f" {energy:14.6f} {'---':>10s}"
    )

    traces = []
    charge_deltas = []
    for i, (requested, op) in enumerate(steps, start=1):
        trace = _apply_with_trace(G, target, requested, op)
        traces.append(trace)
        Q = compute_noether_charge(G)
        energy = compute_energy_functional(G)
        dQ = Q - Q_prev
        charge_deltas.append(dQ)
        print(
            f"  {i:6d} {requested:7s} {str(trace['actual']):7s}"
            f" {Q:14.6f} {energy:14.6f} {dQ:+10.6f}"
        )
        Q_prev = Q

    print("\n  Recorded runtime history:")
    for trace in traces:
        print(
            f"  requested {trace['requested']} -> actual {trace['actual']}; "
            f"history {trace['history_before']} -> {trace['history_after']}"
        )

    print("\n  Scope:")
    print("  dQ is a sampled charge difference. Its size does not identify a")
    print("  grammar violation or prove conservation without evaluating the full")
    print("  continuity balance and its source term on the same trajectory.")
    return {"traces": traces, "charge_deltas": charge_deltas}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    print()
    print("  TNFR Example 37: Operator-Tetrad Response Diagnostics")
    print("  Runtime Attribution & Structural-Field Telemetry")
    print("  " + "=" * 50)
    print(f"  Seed: {SEED}  |  Theory: AGENTS.md, STRUCTURAL_OPERATORS.md")
    print()

    responses = experiment_operator_fingerprints()
    energy_observations = experiment_energy_signature()
    policy_result = experiment_tetrad_safety()
    charge_result = experiment_charge_telemetry()

    substitutions = sum(bool(result["fallback"]) for result in responses.values())
    energy_substitutions = sum(
        bool(result["fallback"]) for result in energy_observations
    )
    max_charge_step = max(abs(value) for value in charge_result["charge_deltas"])

    print("\n" + "=" * 72)
    print("  SUMMARY: Bounded Findings From This Seeded Protocol")
    print("=" * 72)
    print(
        f"  1. Fresh one-call trials produced {substitutions} runtime "
        "substitution(s);\n"
        "     every field delta is attributed to its recorded actual glyph.\n\n"
        f"  2. The U2-role comparison produced {energy_substitutions} runtime "
        "substitution(s).\n"
        "     Its energy changes are observations, not universal bounds.\n\n"
        "  3. Exact phase-wrap bounds respected along the requested word: "
        f"{policy_result['exact_phase_bounds_respected']}.\n"
        "     Selected telemetry policies all satisfied: "
        f"{policy_result['selected_policies_satisfied']}.\n\n"
        f"  4. Largest sampled |dQ| was {max_charge_step:.6f}; interpreting it "
        "requires\n"
        "     the continuity balance and source term, not operator labels alone."
    )


if __name__ == "__main__":
    main()
