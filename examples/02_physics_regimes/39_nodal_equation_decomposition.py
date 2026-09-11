#!/usr/bin/env python3
"""
Example 39 — Nodal Equation Operator Decomposition
===================================================

Compares the nodal-equation state dEPI/dt = nu_f * DELTA_NFR(t) before and
after individual operator calls, then captures tetrad and conservation
diagnostics from the resulting graph snapshots:

    Operator -> (nu_f, DELTA_NFR) -> dEPI/dt -> Tetrad Fields -> Conservation

Physics
-------
The nodal equation anchors TNFR evolution. Each canonical operator has a
declared primary channel and additional contract conditions. A one-call state
difference is not a time-discretized integration step and cannot recover that
contract from relative magnitudes alone.

This experiment measures:
  1. Observed nu_f, DELTA_NFR, and EPI differences for one seeded state
  2. The instantaneous nodal right-hand side before selected operator calls
  3. Tetrad, Q, and E snapshot differences after those calls

The output is a reproducible diagnostic trace. It does not prove a complete
causal decomposition, operator symplecticity, conservation, convergence, or
minimality/completeness of the tetrad.

References
----------
- theory/STRUCTURAL_OPERATORS.md (per-operator physics)
- theory/UNIFIED_GRAMMAR_RULES.md (nodal equation derivation)
- src/tnfr/operators/nodal_equation.py (validation implementation)
- AGENTS.md: "Nodal Equation Integrity" (Invariant #1)
"""

import copy
import math
import os
import sys

import networkx as nx
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults
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
from tnfr.operators.nodal_equation import compute_expected_depi_dt
from tnfr.physics.conservation import compute_energy_functional, compute_noether_charge
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
)

# ── reproducibility ──────────────────────────────────────────────────────
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


def _epi_scalar(val):
    """Extract a scalar magnitude from an EPI value.

    After operator application EPI is a dict
    {'continuous': (c1, c2), 'discrete': (d1, d2), 'grid': (g1, g2)}.
    We return the magnitude of the first continuous component,
    matching the canonical _max_bepi_magnitude convention.
    """
    if isinstance(val, dict):
        c = val.get("continuous", ((0.0,),))
        try:
            return float(abs(c[0]))
        except (TypeError, IndexError):
            return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _capture_node(G, node):
    """Capture full node state for nodal equation analysis."""
    d = G.nodes[node]
    return {
        "EPI": _epi_scalar(d.get("EPI", 0.0)),
        "nu_f": float(d.get("nu_f", 0.0)),
        "delta_nfr": float(d.get("delta_nfr", 0.0)),
        "theta": float(d.get("theta", 0.0)),
    }


def _glyph_code(value):
    """Normalize a recorded glyph to its public short code."""
    raw = getattr(value, "value", value)
    return str(raw).rsplit(".", 1)[-1].upper()


def _apply_with_trace(G, node, operator):
    """Apply one request and report the glyph that history actually records.

    ``Operator.__call__`` may replace a request to satisfy the grammar. The
    appended history entry is therefore the authoritative execution label.
    Runtime failures are returned explicitly, including failures that occur
    after an operator has already changed state or appended history.
    """
    before = tuple(G.nodes[node].get("glyph_history", ()))
    error = None
    try:
        operator(G, node)
    except Exception as exc:  # The caller prints every rejection/failure.
        error = f"{type(exc).__name__}: {exc}"

    after = tuple(G.nodes[node].get("glyph_history", ()))
    if after == before or not after:
        executed = None
    elif len(after) > len(before) and after[: len(before)] == before:
        executed = _glyph_code(after[-1])
    else:
        # A bounded history can evict its oldest entry while appending a new
        # glyph. Its final entry still identifies the latest execution.
        executed = _glyph_code(after[-1])
    return executed, error


def _tetrad_summary(G):
    """Compute scalar tetrad summary for the whole network."""
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    xi_c = estimate_coherence_length(G)
    return {
        "Phi_s_mean": float(np.mean(list(phi_s.values()))),
        "grad_phi_mean": float(np.mean(list(grad_phi.values()))),
        "K_phi_rms": float(np.sqrt(np.mean([v**2 for v in k_phi.values()]))),
        "xi_C": float(xi_c),
    }


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Nodal Equation Decomposition Per Operator
# ═══════════════════════════════════════════════════════════════════════


def experiment_nodal_decomposition():
    """Measure one-call state differences for each operator.

    The reported dominant delta is a snapshot heuristic. Canonical operator
    channels come from operator contracts, not this magnitude comparison.
    """
    print("=" * 72)
    print("  EXPERIMENT 1: One-Call State Differences Per Operator")
    print("  dEPI/dt = nu_f * DELTA_NFR: sampled state before and after")
    print("=" * 72)

    G_base = _build_graph()
    target = 0

    ALL_OPS = [
        ("AL", "Emission", Emission),
        ("EN", "Reception", Reception),
        ("IL", "Coherence", Coherence),
        ("OZ", "Dissonance", Dissonance),
        ("UM", "Coupling", Coupling),
        ("RA", "Resonance", Resonance),
        ("SHA", "Silence", Silence),
        ("VAL", "Expansion", Expansion),
        ("NUL", "Contraction", Contraction),
        ("THOL", "SelfOrg", SelfOrganization),
        ("ZHIR", "Mutation", Mutation),
        ("NAV", "Transition", Transition),
        ("REMESH", "Recursivity", Recursivity),
    ]

    print("\n  Req/Requested identify the call; Exec identifies the recorded glyph.")
    print(
        f'  {"Req":5s} {"Exec":5s} {"Requested":14s} {"d(nu_f)":>10s}'
        f' {"d(DNFR)":>10s} {"d(EPI)":>10s} {"RHS before":>10s}'
        f' {"Dominant delta":>16s}'
    )
    print("  " + "-" * 90)

    lever_summary = {}
    for glyph, name, cls in ALL_OPS:
        G = copy.deepcopy(G_base)
        before = _capture_node(G, target)

        executed, error = _apply_with_trace(G, target, cls())
        execution_label = executed or "---"
        trace_label = glyph if executed == glyph else f"{glyph}->{execution_label}"

        if error is not None:
            lever_summary[trace_label] = "ERROR"
            print(
                f"  {glyph:5s} {execution_label:5s} {name:14s}"
                f" [ERROR: {error[:58]}]"
            )
            continue
        if executed is None:
            lever_summary[trace_label] = "UNRECORDED"
            print(
                f"  {glyph:5s} {'---':5s} {name:14s}"
                " [NO GLYPH APPENDED; state differences not attributed]"
            )
            continue

        after = _capture_node(G, target)
        d_nu_f = after["nu_f"] - before["nu_f"]
        d_dnfr = after["delta_nfr"] - before["delta_nfr"]
        d_epi = after["EPI"] - before["EPI"]
        expected = before["nu_f"] * before["delta_nfr"]

        # Classify only the dominant observed delta on this seeded state.
        if abs(d_nu_f) > abs(d_dnfr) * 2 and abs(d_nu_f) > 1e-8:
            lever = "nu_f (capacity)"
        elif abs(d_dnfr) > abs(d_nu_f) * 2 and abs(d_dnfr) > 1e-8:
            lever = "DNFR (pressure)"
        elif abs(d_nu_f) > 1e-8 or abs(d_dnfr) > 1e-8:
            lever = "BOTH"
        else:
            lever = "NEUTRAL"

        lever_summary[trace_label] = lever
        print(
            f"  {glyph:5s} {executed:5s} {name:14s} {d_nu_f:+10.6f}"
            f" {d_dnfr:+10.6f} {d_epi:+10.6f} {expected:10.6f}"
            f" {lever:>16s}"
        )

    print("\n  Observed-delta summary (not the operator-contract partition):")
    for category in ["nu_f (capacity)", "DNFR (pressure)", "BOTH", "NEUTRAL"]:
        ops = [g for g, label in lever_summary.items() if label == category]
        if ops:
            print(f'    {category:20s}: {", ".join(ops)}')

    return lever_summary


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Snapshot Chain — Operator -> Nodal Eq -> Tetrad
# ═══════════════════════════════════════════════════════════════════════


def experiment_causal_chain():
    """Compare selected state and diagnostic snapshots around an operator call.

    For selected operators (one stabiliser, one destabiliser, one coupling):
      1. Measure (nu_f, DELTA_NFR) before/after
      2. Compute the pre-call instantaneous RHS nu_f * DELTA_NFR
      3. Measure tetrad snapshot differences
      4. Measure candidate energy and charge differences

    These read-outs diagnose the graph state. Their finite differences do not
    establish a complete or unidirectional causal factorization.
    """
    print("\n" + "=" * 72)
    print("  EXPERIMENT 2: Operator and Diagnostic Snapshot Chain")
    print("  Operator -> state differences; state -> tetrad, E, and Q read-outs")
    print("=" * 72)

    test_ops = [
        ("IL", "Coherence (stabiliser)", Coherence),
        ("OZ", "Dissonance (destabiliser)", Dissonance),
        ("UM", "Coupling (phase-gated)", Coupling),
        ("AL", "Emission (generator)", Emission),
    ]

    for glyph, label, cls in test_ops:
        G = _build_graph()
        target = 0

        # Before state
        node_before = _capture_node(G, target)
        tetrad_before = _tetrad_summary(G)
        E_before = compute_energy_functional(G)
        Q_before = compute_noether_charge(G)
        rhs_before = compute_expected_depi_dt(G, target)

        # Apply the request and use history, rather than the requested class,
        # as the source of truth for what the grammar executed.
        executed, error = _apply_with_trace(G, target, cls())

        # After state
        node_after = _capture_node(G, target)
        tetrad_after = _tetrad_summary(G)
        E_after = compute_energy_functional(G)
        Q_after = compute_noether_charge(G)

        print(f"\n  --- Requested {label} ({glyph}) ---")
        print(f"  Executed glyph: {executed or 'none recorded'}")
        if error is not None:
            print(f"  ERROR: {error}")
            print("  State differences after a failed call are not attributed.")
            continue
        if executed is None:
            print("  No glyph was appended; state differences are not attributed.")
            continue
        if executed != glyph:
            print(f"  Grammar fallback: requested {glyph}, executed {executed}.")

        print(f"  State differences after executed {executed}:")
        print(
            f'    nu_f:     {node_before["nu_f"]:.6f}'
            f' -> {node_after["nu_f"]:.6f}'
            f'  (d = {node_after["nu_f"] - node_before["nu_f"]:+.6f})'
        )
        print(
            f'    DNFR:     {node_before["delta_nfr"]:.6f}'
            f' -> {node_after["delta_nfr"]:.6f}'
            f'  (d = {node_after["delta_nfr"] - node_before["delta_nfr"]:+.6f})'
        )
        print(
            f'    EPI:      {node_before["EPI"]:.6f}'
            f' -> {node_after["EPI"]:.6f}'
            f'  (d = {node_after["EPI"] - node_before["EPI"]:+.6f})'
        )
        print(f"    RHS before call = nu_f * DNFR = {rhs_before:.6f}")

        print(f"  Tetrad Snapshot Differences:")
        for field in ["Phi_s_mean", "grad_phi_mean", "K_phi_rms", "xi_C"]:
            b = tetrad_before[field]
            a = tetrad_after[field]
            print(f"    {field:15s}: {b:.6f} -> {a:.6f}" f"  (d = {a - b:+.6f})")

        print(f"  Candidate Energy and Charge Read-outs:")
        print(
            f"    E (energy): {E_before:.6f} -> {E_after:.6f}"
            f"  (dE = {E_after - E_before:+.6f})"
        )
        print(
            f"    Q (charge): {Q_before:.6f} -> {Q_after:.6f}"
            f"  (dQ = {Q_after - Q_before:+.6f})"
        )


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Multi-Step Nodal Equation Trajectory
# ═══════════════════════════════════════════════════════════════════════


def experiment_multi_step_trajectory():
    """Track nu_f, DELTA_NFR, and EPI evolution through a full sequence.

    This reports sampled state and instantaneous RHS values after attempted
    operator calls. It is not a numerical integration trace of that RHS.
    """
    print("\n" + "=" * 72)
    print("  EXPERIMENT 3: Multi-Step Nodal Equation Trajectory")
    print("  Sampled nu_f, DELTA_NFR, RHS, and EPI values")
    print("=" * 72)

    G = _build_graph()
    target = 0

    # Deterministic attempted sequence. Individual calls may reject their
    # preconditions; this example reports the resulting sampled state.
    sequence = [
        ("AL", Emission()),
        ("EN", Reception()),
        ("UM", Coupling()),
        ("IL", Coherence()),
        ("OZ", Dissonance()),
        ("IL", Coherence()),
        ("RA", Resonance()),
        ("IL", Coherence()),
        ("SHA", Silence()),
    ]

    print(
        f'\n  {"t":>3s} {"Req":>5s} {"Exec":>5s} {"Status":>8s}'
        f' {"nu_f":>10s} {"DNFR":>10s} {"nu_f*DNFR":>10s}'
        f' {"EPI":>10s} {"dEPI":>10s}'
    )
    print("  " + "-" * 84)

    state = _capture_node(G, target)
    product = state["nu_f"] * state["delta_nfr"]
    print(
        f"  {0:3d} {'---':>5s} {'---':>5s} {'initial':>8s}"
        f" {state['nu_f']:10.6f}"
        f" {state['delta_nfr']:10.6f}"
        f" {product:10.6f} {state['EPI']:10.6f} {'---':>10s}"
    )

    epi_prev = state["EPI"]
    trajectory = [state.copy()]
    for i, (glyph, op) in enumerate(sequence):
        executed, error = _apply_with_trace(G, target, op)
        state = _capture_node(G, target)
        product = state["nu_f"] * state["delta_nfr"]
        d_epi = state["EPI"] - epi_prev
        status = "ERROR" if error is not None else ("OK" if executed else "UNREC")
        print(
            f"  {i + 1:3d} {glyph:>5s} {(executed or '---'):>5s}"
            f" {status:>8s} {state['nu_f']:10.6f}"
            f" {state['delta_nfr']:10.6f}"
            f" {product:10.6f} {state['EPI']:10.6f}"
            f" {d_epi:+10.6f}"
        )
        if error is not None:
            print(f"      ERROR for requested {glyph}: {error}")
        elif executed is None:
            print(f"      Requested {glyph}: no glyph was appended.")
        elif executed != glyph:
            print(f"      Grammar fallback: requested {glyph}, executed {executed}.")
        epi_prev = state["EPI"]
        trajectory.append(state.copy())

    # Compute waveform statistics
    nu_fs = [t["nu_f"] for t in trajectory]
    dnfrs = [t["delta_nfr"] for t in trajectory]
    epis = [t["EPI"] for t in trajectory]

    print(f"\n  Waveform Statistics:")
    print(
        f"    nu_f  range: [{min(nu_fs):.4f}, {max(nu_fs):.4f}]"
        f"  mean = {np.mean(nu_fs):.4f}"
    )
    print(
        f"    DNFR  range: [{min(dnfrs):.4f}, {max(dnfrs):.4f}]"
        f"  mean = {np.mean(dnfrs):.4f}"
    )
    print(
        f"    EPI   range: [{min(epis):.4f}, {max(epis):.4f}]"
        f"  net change = {epis[-1] - epis[0]:+.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Tetrad Response Functions
# ═══════════════════════════════════════════════════════════════════════


def experiment_tetrad_response():
    """Measure finite tetrad differences in a controlled DELTA_NFR scan.

    Requests Coherence at several DELTA_NFR values and records the glyph that
    the grammar actually executes before measuring each read-out difference.
    This finite sample is not a derivative, susceptibility, or universal
    response function.
    """
    print("\n" + "=" * 72)
    print("  EXPERIMENT 4: Finite Tetrad Difference Scan")
    print("  Seeded differences with requested and executed glyph trace")
    print("=" * 72)

    G_base = _build_graph()
    target = 0

    # Vary the initial DELTA_NFR to create different perturbation magnitudes
    perturbations = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8]

    print(
        f'\n  {"DNFR_init":>10s} {"Req":>5s} {"Exec":>5s} {"Status":>8s}'
        f' {"d(Phi_s)":>10s} {"d(grad_phi)":>12s}'
        f' {"d(K_phi)":>10s} {"d(xi_C)":>10s}'
    )
    print("  " + "-" * 78)

    responses = []
    for dnfr_val in perturbations:
        G = copy.deepcopy(G_base)
        # Set controlled perturbation
        G.nodes[target]["delta_nfr"] = dnfr_val

        tetrad_before = _tetrad_summary(G)
        E_before = compute_energy_functional(G)

        # The requested class is not necessarily the grammar-selected glyph.
        executed, error = _apply_with_trace(G, target, Coherence())

        tetrad_after = _tetrad_summary(G)

        d_phi_s = tetrad_after["Phi_s_mean"] - tetrad_before["Phi_s_mean"]
        d_grad = tetrad_after["grad_phi_mean"] - tetrad_before["grad_phi_mean"]
        d_k = tetrad_after["K_phi_rms"] - tetrad_before["K_phi_rms"]
        d_xi = tetrad_after["xi_C"] - tetrad_before["xi_C"]

        status = "ERROR" if error is not None else ("OK" if executed else "UNREC")
        if error is None and executed is not None:
            responses.append(
                {
                    "dnfr": dnfr_val,
                    "d_phi_s": d_phi_s,
                    "d_grad": d_grad,
                    "d_k": d_k,
                    "d_xi": d_xi,
                }
            )

        print(
            f"  {dnfr_val:10.4f} {'IL':>5s} {(executed or '---'):>5s}"
            f" {status:>8s} {d_phi_s:+10.6f} {d_grad:+12.6f}"
            f" {d_k:+10.6f} {d_xi:+10.6f}"
        )
        if error is not None:
            print(f"      ERROR for requested IL: {error}")
        elif executed is None:
            print("      Requested IL: no glyph was appended.")
        elif executed != "IL":
            print(f"      Grammar fallback: requested IL, executed {executed}.")

    # Report a descriptive sample correlation, without inferring a law.
    if len(responses) >= 2:
        dnfrs = [r["dnfr"] for r in responses]
        for field_name, key in [
            ("Phi_s", "d_phi_s"),
            ("grad_phi", "d_grad"),
            ("K_phi", "d_k"),
        ]:
            vals = [r[key] for r in responses]
            if float(np.ptp(vals)) > 1e-10:
                corr = abs(np.corrcoef(dnfrs, vals)[0, 1])
                print(
                    f"\n    {field_name} finite-scan |correlation| = {corr:.4f}"
                )
            else:
                print(
                    f"\n    {field_name} is constant in this scan; "
                    "correlation is undefined."
                )

    print("\n  Interpretation:")
    print("  The execution column, sourced from glyph_history, identifies")
    print("  which operator produced each successful row.")
    print("  Correlation summarizes only these seeded finite differences.")
    print("  Zero or nonzero changes depend on field definitions and operator state.")
    print("  No derivative, threshold law, or susceptibility is inferred.")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    print()
    print("  TNFR Example 39: Nodal Equation Operator Decomposition")
    print("  dEPI/dt = nu_f * DELTA_NFR(t) — The Single Dynamical Law")
    print("  " + "=" * 55)
    print(f"  Seed: {SEED}  |  Theory: Nodal Equation, Invariant #1")
    print()

    lever_summary = experiment_nodal_decomposition()
    experiment_causal_chain()
    experiment_multi_step_trajectory()
    experiment_tetrad_response()

    print("\n" + "=" * 72)
    print("  SUMMARY: Nodal Equation Decomposition Findings")
    print("=" * 72)
    print(
        """
  1. Operator calls produce reproducible state differences on the declared
     seed. The dominant-delta labels describe this sample; operator contracts
     remain the authority for canonical channel assignments.

  2. nu_f * DELTA_NFR is the instantaneous nodal right-hand side. A raw EPI
     difference across an operator call is not automatically dt times that RHS.

  3. The tetrad, candidate energy, and charge are graph-state diagnostics.
     Their observed differences do not by themselves prove conservation,
     convergence, a causal factorization, or grammar compliance.

  4. The finite DELTA_NFR scan shows that the tetrad channels respond
     differently on this graph. It supports joint diagnostic use but does not
     prove that the tetrad is minimal, complete, irreducible, or reconstructive.
"""
    )


if __name__ == "__main__":
    main()
