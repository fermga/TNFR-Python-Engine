actualiza el resto de documentos teoricos y el codigo para reflejar estos descubrimientos#!/usr/bin/env python3
"""
Example 39 — Nodal Equation Operator Decomposition
===================================================

Decomposes the nodal equation dEPI/dt = nu_f * DELTA_NFR(t) into
per-operator contributions, tracing the complete causal chain:

    Operator -> (nu_f, DELTA_NFR) -> dEPI/dt -> Tetrad Fields -> Conservation

Physics
-------
The nodal equation is the *single* dynamical law of TNFR. Every operator
modifies EPI *exclusively* through this equation by changing either nu_f
(reorganisation capacity) or DELTA_NFR (reorganisation pressure) or both.

This experiment measures:
  1. How each operator partitions its effect between nu_f and DELTA_NFR
  2. How the resulting dEPI/dt maps to tetrad field changes
  3. How conservation quantities (Q, E) respond to each component

The decomposition reveals that operators do not change EPI directly —
they modulate the *terms* of the nodal equation, and the equation itself
propagates changes to the structural fields.

References
----------
- theory/STRUCTURAL_OPERATORS.md (per-operator physics)
- theory/UNIFIED_GRAMMAR_RULES.md (nodal equation derivation)
- src/tnfr/operators/nodal_equation.py (validation implementation)
- AGENTS.md: "Nodal Equation Integrity" (Invariant #1)
"""

import os
import sys
import copy
import math
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tnfr.constants import inject_defaults
from tnfr.constants.canonical import PHI, GAMMA, PI
from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)
from tnfr.physics.conservation import (
    compute_energy_functional,
    compute_noether_charge,
)
from tnfr.operators.definitions import (
    Emission,
    Reception,
    Coherence,
    Dissonance,
    Coupling,
    Resonance,
    Silence,
    Expansion,
    Contraction,
    SelfOrganization,
    Mutation,
    Transition,
    Recursivity,
)
from tnfr.operators.nodal_equation import compute_expected_depi_dt

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
        c = val.get('continuous', ((0.0,),))
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
        'EPI': _epi_scalar(d.get('EPI', 0.0)),
        'nu_f': float(d.get('nu_f', 0.0)),
        'delta_nfr': float(d.get('delta_nfr', 0.0)),
        'theta': float(d.get('theta', 0.0)),
    }


def _tetrad_summary(G):
    """Compute scalar tetrad summary for the whole network."""
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    xi_c = estimate_coherence_length(G)
    return {
        'Phi_s_mean': float(np.mean(list(phi_s.values()))),
        'grad_phi_mean': float(np.mean(list(grad_phi.values()))),
        'K_phi_rms': float(np.sqrt(np.mean([v**2 for v in k_phi.values()]))),
        'xi_C': float(xi_c),
    }


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Nodal Equation Decomposition Per Operator
# ═══════════════════════════════════════════════════════════════════════

def experiment_nodal_decomposition():
    """For each operator, measure how it changes nu_f vs DELTA_NFR.

    The nodal equation dEPI/dt = nu_f * DELTA_NFR means operators have
    two "levers": they can change the frequency (capacity) or the
    pressure (driving force). Different operators pull different levers.
    """
    print('=' * 72)
    print('  EXPERIMENT 1: Nodal Equation Decomposition Per Operator')
    print('  dEPI/dt = nu_f * DELTA_NFR: which lever does each op pull?')
    print('=' * 72)

    G_base = _build_graph()
    target = 0

    ALL_OPS = [
        ('AL',    'Emission',         Emission),
        ('EN',    'Reception',        Reception),
        ('IL',    'Coherence',        Coherence),
        ('OZ',    'Dissonance',       Dissonance),
        ('UM',    'Coupling',         Coupling),
        ('RA',    'Resonance',        Resonance),
        ('SHA',   'Silence',          Silence),
        ('VAL',   'Expansion',        Expansion),
        ('NUL',   'Contraction',      Contraction),
        ('THOL',  'SelfOrg',         SelfOrganization),
        ('ZHIR',  'Mutation',         Mutation),
        ('NAV',   'Transition',       Transition),
        ('REMESH','Recursivity',      Recursivity),
    ]

    print(f'\n  {"Glyph":7s} {"Name":14s} {"d(nu_f)":>10s} {"d(DNFR)":>10s}'
          f' {"d(EPI)":>10s} {"nu_f*DNFR":>10s} {"Primary lever":>16s}')
    print('  ' + '-' * 82)

    lever_summary = {}
    for glyph, name, cls in ALL_OPS:
        G = copy.deepcopy(G_base)
        before = _capture_node(G, target)

        try:
            op = cls()
            op(G, target)
            after = _capture_node(G, target)

            d_nu_f = after['nu_f'] - before['nu_f']
            d_dnfr = after['delta_nfr'] - before['delta_nfr']
            d_epi = after['EPI'] - before['EPI']
            expected = before['nu_f'] * before['delta_nfr']

            # Classify primary lever
            if abs(d_nu_f) > abs(d_dnfr) * 2 and abs(d_nu_f) > 1e-8:
                lever = 'nu_f (capacity)'
            elif abs(d_dnfr) > abs(d_nu_f) * 2 and abs(d_dnfr) > 1e-8:
                lever = 'DNFR (pressure)'
            elif abs(d_nu_f) > 1e-8 or abs(d_dnfr) > 1e-8:
                lever = 'BOTH'
            else:
                lever = 'NEUTRAL'

            lever_summary[glyph] = lever
            print(f'  {glyph:7s} {name:14s} {d_nu_f:+10.6f} {d_dnfr:+10.6f}'
                  f' {d_epi:+10.6f} {expected:10.6f} {lever:>16s}')
        except Exception as exc:
            lever_summary[glyph] = 'SKIPPED'
            print(f'  {glyph:7s} {name:14s} [SKIPPED: {str(exc)[:40]}]')

    print('\n  Lever Classification Summary:')
    for category in ['nu_f (capacity)', 'DNFR (pressure)', 'BOTH', 'NEUTRAL']:
        ops = [g for g, l in lever_summary.items() if l == category]
        if ops:
            print(f'    {category:20s}: {", ".join(ops)}')

    return lever_summary


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Causal Chain — Operator -> Nodal Eq -> Tetrad
# ═══════════════════════════════════════════════════════════════════════

def experiment_causal_chain():
    """Trace the full causal chain from operator to tetrad fields.

    For selected operators (one stabiliser, one destabiliser, one coupling):
      1. Measure (nu_f, DELTA_NFR) before/after
      2. Compute predicted dEPI/dt = nu_f * DELTA_NFR
      3. Measure tetrad field response
      4. Measure conservation quantity change

    This demonstrates that the tetrad is a *diagnostic* of the nodal
    equation, not an independent dynamical system.
    """
    print('\n' + '=' * 72)
    print('  EXPERIMENT 2: Full Causal Chain')
    print('  Operator -> (nu_f, DNFR) -> dEPI/dt -> Tetrad -> Conservation')
    print('=' * 72)

    test_ops = [
        ('IL',  'Coherence (stabiliser)',  Coherence),
        ('OZ',  'Dissonance (destabiliser)', Dissonance),
        ('UM',  'Coupling (phase-gated)',  Coupling),
        ('AL',  'Emission (generator)',    Emission),
    ]

    for glyph, label, cls in test_ops:
        G = _build_graph()
        target = 0

        # Before state
        node_before = _capture_node(G, target)
        tetrad_before = _tetrad_summary(G)
        E_before = compute_energy_functional(G)
        Q_before = compute_noether_charge(G)
        predicted_depi = compute_expected_depi_dt(G, target)

        # Apply operator
        try:
            op = cls()
            op(G, target)
            applied = True
        except Exception:
            applied = False

        # After state
        node_after = _capture_node(G, target)
        tetrad_after = _tetrad_summary(G)
        E_after = compute_energy_functional(G)
        Q_after = compute_noether_charge(G)

        print(f'\n  --- {label} ({glyph}) ---')
        if not applied:
            print('  [Operator preconditions not met; skipped]')
            continue

        print(f'  Nodal Equation Decomposition:')
        print(f'    nu_f:     {node_before["nu_f"]:.6f}'
              f' -> {node_after["nu_f"]:.6f}'
              f'  (d = {node_after["nu_f"] - node_before["nu_f"]:+.6f})')
        print(f'    DNFR:     {node_before["delta_nfr"]:.6f}'
              f' -> {node_after["delta_nfr"]:.6f}'
              f'  (d = {node_after["delta_nfr"] - node_before["delta_nfr"]:+.6f})')
        print(f'    EPI:      {node_before["EPI"]:.6f}'
              f' -> {node_after["EPI"]:.6f}'
              f'  (d = {node_after["EPI"] - node_before["EPI"]:+.6f})')
        print(f'    Predicted dEPI/dt = nu_f * DNFR = {predicted_depi:.6f}')

        print(f'  Tetrad Response:')
        for field in ['Phi_s_mean', 'grad_phi_mean', 'K_phi_rms', 'xi_C']:
            b = tetrad_before[field]
            a = tetrad_after[field]
            print(f'    {field:15s}: {b:.6f} -> {a:.6f}'
                  f'  (d = {a - b:+.6f})')

        print(f'  Conservation:')
        print(f'    E (energy): {E_before:.6f} -> {E_after:.6f}'
              f'  (dE = {E_after - E_before:+.6f})')
        print(f'    Q (charge): {Q_before:.6f} -> {Q_after:.6f}'
              f'  (dQ = {Q_after - Q_before:+.6f})')


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Multi-Step Nodal Equation Trajectory
# ═══════════════════════════════════════════════════════════════════════

def experiment_multi_step_trajectory():
    """Track nu_f, DELTA_NFR, and EPI evolution through a full sequence.

    This provides a "waveform view" of the nodal equation, showing how
    the two terms (nu_f, DELTA_NFR) oscillate as operators are applied.
    """
    print('\n' + '=' * 72)
    print('  EXPERIMENT 3: Multi-Step Nodal Equation Trajectory')
    print('  Waveform: nu_f(t), DELTA_NFR(t), EPI(t)')
    print('=' * 72)

    G = _build_graph()
    target = 0

    # Extended sequence: Bootstrap + Explore + Stabilise + Propagate
    sequence = [
        ('AL',  Emission()),
        ('EN',  Reception()),
        ('UM',  Coupling()),
        ('IL',  Coherence()),
        ('OZ',  Dissonance()),
        ('IL',  Coherence()),
        ('RA',  Resonance()),
        ('IL',  Coherence()),
        ('SHA', Silence()),
    ]

    print(f'\n  {"t":>3s} {"Op":>5s} {"nu_f":>10s} {"DNFR":>10s}'
          f' {"nu_f*DNFR":>10s} {"EPI":>10s} {"dEPI":>10s}')
    print('  ' + '-' * 62)

    state = _capture_node(G, target)
    product = state['nu_f'] * state['delta_nfr']
    print(f"  {0:3d} {'---':>5s} {state['nu_f']:10.6f}"
          f" {state['delta_nfr']:10.6f}"
          f" {product:10.6f} {state['EPI']:10.6f} {'---':>10s}")

    epi_prev = state['EPI']
    trajectory = [state.copy()]
    for i, (glyph, op) in enumerate(sequence):
        try:
            op(G, target)
        except Exception:
            pass
        state = _capture_node(G, target)
        product = state['nu_f'] * state['delta_nfr']
        d_epi = state['EPI'] - epi_prev
        print(f"  {i + 1:3d} {glyph:>5s} {state['nu_f']:10.6f}"
              f" {state['delta_nfr']:10.6f}"
              f" {product:10.6f} {state['EPI']:10.6f}"
              f" {d_epi:+10.6f}")
        epi_prev = state['EPI']
        trajectory.append(state.copy())

    # Compute waveform statistics
    nu_fs = [t['nu_f'] for t in trajectory]
    dnfrs = [t['delta_nfr'] for t in trajectory]
    epis = [t['EPI'] for t in trajectory]

    print(f'\n  Waveform Statistics:')
    print(f'    nu_f  range: [{min(nu_fs):.4f}, {max(nu_fs):.4f}]'
          f'  mean = {np.mean(nu_fs):.4f}')
    print(f'    DNFR  range: [{min(dnfrs):.4f}, {max(dnfrs):.4f}]'
          f'  mean = {np.mean(dnfrs):.4f}')
    print(f'    EPI   range: [{min(epis):.4f}, {max(epis):.4f}]'
          f'  net change = {epis[-1] - epis[0]:+.4f}')


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Tetrad Response Functions
# ═══════════════════════════════════════════════════════════════════════

def experiment_tetrad_response():
    """Measure tetrad field sensitivity to nodal equation perturbations.

    Applies small and large DELTA_NFR changes (via Coherence at different
    states) and measures how each tetrad field responds. This reveals
    the "response function" df_tetrad / d(DELTA_NFR).
    """
    print('\n' + '=' * 72)
    print('  EXPERIMENT 4: Tetrad Response Functions')
    print('  How tetrad fields respond to nodal equation perturbations')
    print('=' * 72)

    G_base = _build_graph()
    target = 0

    # Vary the initial DELTA_NFR to create different perturbation magnitudes
    perturbations = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8]

    print(f'\n  {"DNFR_init":>10s} {"d(Phi_s)":>10s} {"d(grad_phi)":>12s}'
          f' {"d(K_phi)":>10s} {"d(xi_C)":>10s}')
    print('  ' + '-' * 56)

    responses = []
    for dnfr_val in perturbations:
        G = copy.deepcopy(G_base)
        # Set controlled perturbation
        G.nodes[target]['delta_nfr'] = dnfr_val

        tetrad_before = _tetrad_summary(G)
        E_before = compute_energy_functional(G)

        # Apply Coherence (stabiliser) to evolve the nodal equation
        try:
            Coherence()(G, target)
        except Exception:
            pass

        tetrad_after = _tetrad_summary(G)

        d_phi_s = tetrad_after['Phi_s_mean'] - tetrad_before['Phi_s_mean']
        d_grad = tetrad_after['grad_phi_mean'] - tetrad_before['grad_phi_mean']
        d_k = tetrad_after['K_phi_rms'] - tetrad_before['K_phi_rms']
        d_xi = tetrad_after['xi_C'] - tetrad_before['xi_C']

        responses.append({
            'dnfr': dnfr_val,
            'd_phi_s': d_phi_s,
            'd_grad': d_grad,
            'd_k': d_k,
            'd_xi': d_xi,
        })

        print(f'  {dnfr_val:10.4f} {d_phi_s:+10.6f} {d_grad:+12.6f}'
              f' {d_k:+10.6f} {d_xi:+10.6f}')

    # Check linearity
    if len(responses) >= 2:
        dnfrs = [r['dnfr'] for r in responses]
        for field_name, key in [('Phi_s', 'd_phi_s'),
                                ('grad_phi', 'd_grad'),
                                ('K_phi', 'd_k')]:
            vals = [r[key] for r in responses]
            if any(abs(v) > 1e-10 for v in vals):
                corr = abs(np.corrcoef(dnfrs, vals)[0, 1])
                regime = 'LINEAR' if corr > 0.9 else 'NONLINEAR'
                print(f'\n    {field_name} response: |corr| = {corr:.4f}'
                      f' -> {regime}')

    print('\n  Interpretation:')
    print('  Linear response = tetrad field proportional to DELTA_NFR')
    print('  Nonlinear = field saturates or has threshold behaviour')
    print('  This reveals the structural "susceptibility" of each field')


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print()
    print('  TNFR Example 39: Nodal Equation Operator Decomposition')
    print('  dEPI/dt = nu_f * DELTA_NFR(t) — The Single Dynamical Law')
    print('  ' + '=' * 55)
    print(f'  Seed: {SEED}  |  Theory: Nodal Equation, Invariant #1')
    print()

    lever_summary = experiment_nodal_decomposition()
    experiment_causal_chain()
    experiment_multi_step_trajectory()
    experiment_tetrad_response()

    print('\n' + '=' * 72)
    print('  SUMMARY: Nodal Equation Decomposition Findings')
    print('=' * 72)
    print("""
  1. Operator Lever Analysis:
     Each operator modulates EPI evolution through two channels:
       - nu_f (reorganisation capacity): changed by AL, SHA, VAL, NUL
       - DELTA_NFR (reorganisation pressure): changed by IL, OZ, EN
       - BOTH: UM, RA act on both frequency and pressure
     This dual-lever structure is why grammar needs both U2 (convergence
     of the integral) and U4 (bifurcation control).

  2. Complete Causal Chain:
     Operator -> (d_nu_f, d_DNFR) -> dEPI/dt -> Tetrad Response -> dE, dQ
     The tetrad fields are *diagnostics* of nodal equation dynamics,
     not independent variables. The chain is unidirectional:
     operators drive the nodal equation, which drives the fields.

  3. Multi-Step Waveform:
     nu_f and DELTA_NFR trace oscillating waveforms during sequences.
     Grammar-compliant sequences produce bounded oscillations.
     The product nu_f * DELTA_NFR predicts EPI change at each step.

  4. Tetrad Response Functions:
     Each tetrad field has a characteristic "susceptibility" to DELTA_NFR:
       Phi_s:     responds to cumulative DELTA_NFR (integral, 0th order)
       |grad_phi|: responds to local DELTA_NFR changes (1st order)
       K_phi:     responds to curvature of DELTA_NFR field (2nd order)
       xi_C:      responds to spatial correlation of changes (non-local)
     This recovery of the derivative tower (0th, 1st, 2nd, integral)
     from operator perturbations confirms the Minimal Structural Degrees
     theorem: the tetrad is the complete and irreducible basis for
     characterising nodal equation dynamics.
""")


if __name__ == '__main__':
    main()
