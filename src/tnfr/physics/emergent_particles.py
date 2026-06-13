"""
TNFR Emergent Particles: Coherent Modes Classified by Intrinsic Invariants

Pure-TNFR derivation of particle-like classes from the nodal dynamics, mirroring
the number-theory template (intrinsic invariant -> equilibrium criterion) and the
emergent-chemistry layer (``tnfr.physics.emergent_chemistry``).

The legacy ``tnfr.physics.patterns`` module *places* a structure on a manifold
(a vortex) and *labels* it ("electron-like"). That is an analogy: the label is an
input. Here the classification is an OUTPUT: we measure the intrinsic, quantized
structural invariants of a coherent mode and let the class emerge.

Foundational fact (already canonical in TNFR): the topological charge of a phase
field on a closed structural manifold is QUANTIZED. The winding number

    W = (1 / 2π) ∮ ∇φ · dl  ∈  ℤ

is an integer because the phase field is single-valued on the loop. This is the
exact analogue of the discrete-mode (quantum) regime: just as a bounded manifold
admits only discrete resonant eigenmodes, a closed manifold admits only integer
topological charges. Nothing is imposed — the integrality emerges.

Classification (emergent, from measured invariants only):
  - |W| = 0  : scalar / boson-like coherent mode (no topological charge)
  - |W| = 1  : fermion-like mode (unit topological charge — a structural vortex)
  - |W| ≥ 2  : composite / multi-winding mode (|W| bound vortices)
  - sign(W)  : structural chirality (matter-like W>0 vs. antimatter-like W<0)

The canonical per-node topological-charge density 𝒬 = |∇φ|·J_φ − K_φ·J_ΔNFR and
energy density ℰ (from ``tnfr.physics.unified``) are reported as supporting
telemetry; the conserved *integer* invariant is the winding number.

Theoretical foundation: AGENTS.md (nodal equation; topological charge 𝒬;
discrete-mode regime), theory/TNFR_NUMBER_THEORY.md (intrinsic-invariant template).

Status: RESEARCH (pure-TNFR derivation; classes emerge from measured invariants).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from ..mathematics.unified_numerical import np

_TWO_PI = 2.0 * math.pi


def _wrap_pi(x: float) -> float:
    """Wrap an angle to (-π, π]."""
    y = (x + math.pi) % _TWO_PI - math.pi
    if y <= -math.pi:
        y += _TWO_PI
    return float(y)


def _wrap_2pi(x: float) -> float:
    """Wrap an angle to [0, 2π)."""
    y = x % _TWO_PI
    if y < 0.0:
        y += _TWO_PI
    return float(y)


# ============================================================================
# CLOSED STRUCTURAL MANIFOLD WITH A WINDING PHASE FIELD
# ============================================================================


def winding_ring(n_nodes: int, winding: float, *, base_dnfr: float = 0.05) -> nx.Graph:
    """Build a closed 1D structural manifold (ring) carrying a phase field with
    the requested winding.

    The phase advances by ``winding`` full turns around the loop:
    φ_i = wrap(2π · winding · i / n). The node ordering 0,1,...,n-1,0 defines the
    closed loop. ΔNFR is set to a mild baseline; coherence = 1/(1+|ΔNFR|).

    Note: ``winding`` may be non-integer on input — the *measured* winding number
    is always an integer (that is the point of the quantization demonstration).
    """
    if n_nodes < 3:
        raise ValueError("n_nodes must be >= 3 to form a closed loop")
    G = nx.cycle_graph(n_nodes)
    for i in range(n_nodes):
        ang = _TWO_PI * winding * i / n_nodes
        phi = _wrap_2pi(ang)
        G.nodes[i]["theta"] = phi
        G.nodes[i]["phase"] = phi
        G.nodes[i]["delta_nfr"] = float(base_dnfr)
        G.nodes[i]["dnfr"] = float(base_dnfr)
        G.nodes[i]["coherence"] = 1.0 / (1.0 + abs(base_dnfr))
        G.nodes[i]["EPI"] = 1.0 / (1.0 + abs(base_dnfr))
        G.nodes[i]["nu_f"] = 1.0
    return G


# ============================================================================
# QUANTIZED TOPOLOGICAL CHARGE (WINDING NUMBER) — EMERGES, NOT IMPOSED
# ============================================================================


def winding_number(G: nx.Graph, *, order: list | None = None) -> tuple[int, float]:
    """Compute the integer topological charge (winding number) of the phase
    field around the closed loop.

    Returns (integer_charge, raw_circulation_over_2pi). The raw value lies very
    close to an integer for any single-valued phase field; the integer charge is
    the conserved topological invariant.
    """
    nodes = order if order is not None else sorted(G.nodes())
    n = len(nodes)
    circulation = 0.0
    for k in range(n):
        a = G.nodes[nodes[k]].get("phase", G.nodes[nodes[k]].get("theta", 0.0))
        b = G.nodes[nodes[(k + 1) % n]].get(
            "phase", G.nodes[nodes[(k + 1) % n]].get("theta", 0.0)
        )
        circulation += _wrap_pi(float(b) - float(a))
    raw = circulation / _TWO_PI
    return int(round(raw)), float(raw)


# ============================================================================
# EMERGENT CLASSIFICATION FROM MEASURED INVARIANTS
# ============================================================================


@dataclass(frozen=True)
class EmergentParticle:
    """Structural classification of a coherent mode from intrinsic invariants."""

    winding: int                 # integer topological charge W
    raw_winding: float           # circulation / 2π (≈ W)
    chirality: int               # sign(W): +1 matter-like, -1 antimatter-like, 0 neutral
    energy_density: float        # mean canonical ℰ over the manifold
    charge_density_mean: float   # mean canonical 𝒬 density (telemetry)
    particle_class: str          # emergent label
    is_quantized: bool           # |raw - W| small (topological integrality holds)

    def as_dict(self) -> dict[str, object]:
        return {
            "winding": self.winding,
            "raw_winding": self.raw_winding,
            "chirality": self.chirality,
            "energy_density": self.energy_density,
            "charge_density_mean": self.charge_density_mean,
            "particle_class": self.particle_class,
            "is_quantized": self.is_quantized,
        }


def _class_from_winding(w: int) -> str:
    aw = abs(w)
    if aw == 0:
        return "scalar/boson-like (no topological charge)"
    if aw == 1:
        return "fermion-like (unit topological charge)"
    return f"composite (multi-winding |W|={aw})"


def classify_particle(G: nx.Graph, *, order: list | None = None) -> EmergentParticle:
    """Classify a coherent mode purely from its measured structural invariants.

    The class is an OUTPUT of the measured quantized topological charge, not an
    input label. Energy density ℰ and the canonical topological-charge density 𝒬
    are reported as supporting telemetry.
    """
    w, raw = winding_number(G, order=order)
    chirality = (w > 0) - (w < 0)

    # Canonical telemetry (best-effort; the integer invariant above is primary).
    energy_density = 0.0
    charge_density_mean = 0.0
    try:
        from .unified import compute_energy_density, compute_topological_charge

        ed = compute_energy_density(G)
        tc = compute_topological_charge(G)
        if ed:
            energy_density = float(np.mean([ed[n] for n in G.nodes()]))
        if tc:
            charge_density_mean = float(np.mean([tc[n] for n in G.nodes()]))
    except Exception:
        pass

    return EmergentParticle(
        winding=w,
        raw_winding=raw,
        chirality=int(chirality),
        energy_density=energy_density,
        charge_density_mean=charge_density_mean,
        particle_class=_class_from_winding(w),
        is_quantized=abs(raw - round(raw)) < 1e-6,
    )


__all__ = [
    "EmergentParticle",
    "winding_ring",
    "winding_number",
    "classify_particle",
]
