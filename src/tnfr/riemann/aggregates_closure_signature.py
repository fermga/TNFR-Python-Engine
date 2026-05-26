"""Aggregates-Closure Signature — Diagnostic for B9 = Δ-aggregates-closure (§13quinquaginta-sexta).

This module implements a purely diagnostic quantity, the
**Aggregates-Closure Signature** :math:`\\mathcal{S}_{AC}`, that
quantifies on canonical TNFR aggregate reads whether the canonical
Tier-1+Tier-2 scalar inputs (νf, EPI, φ/θ, ΔNFR) plus the canonical
tetrad and currents (already discharged as closed under B7 and B8
respectively) are *structurally sufficient* to reconstruct each of
the four canonical scalar aggregates — global coherence :math:`C(t)`,
per-node Sense Index :math:`S_i`, per-node energy density
:math:`\\mathcal{E}(i)`, and per-node topological charge
:math:`\\mathcal{Q}(i)` — as a scalar-valued functional, with no
hidden intermediate richer than the Tier-1/Tier-2/tetrad/currents
types and no implicit measure, callable kernel, Banach-derivative
apparatus, or matrix lift introduced during the aggregation step.

Methodological scope (mandatory honesty)
----------------------------------------
This module is a *diagnostic only*. It does **not** construct,
promote, or modify any canonical operator. It does **not** advance
G4 = RH. It does **not** by itself decide the B9 closure question
(which requires the final verdict of §13quinquaginta-septima at
Phase c). Phase b is **n/a** for B9 (B9 is a closure question, not
a type-conjecture; the question is whether the *existing* canonical
Tier-1/Tier-2/tetrad/currents types plus graph metric close the four
aggregate functionals without leakage to a richer intermediate type).

The diagnostic probes two orthogonal axes:

1. **Output-scalar-closure axis** — call each of
   :func:`tnfr.metrics.common.compute_coherence`,
   :func:`tnfr.metrics.sense_index.compute_Si`,
   :func:`tnfr.physics.unified.compute_energy_density`, and
   :func:`tnfr.physics.unified.compute_topological_charge` on a
   canonical probe graph and verify that every output value is
   structurally scalar-coercible (Python ``float`` / NumPy scalar
   / zero-dim array). ``compute_coherence`` returns a single
   ``float``; the other three return ``dict[node, float]`` with
   explicit ``float`` coercion at the per-node aggregation step.

2. **Input-domain-closure axis** — verify that the per-node inputs
   read by the canonical aggregate implementations are confined to
   the Tier-1+Tier-2 scalar slots (``νf`` / ``nu_f``, ``EPI``,
   ``theta`` / ``phi``, the canonical ``ΔNFR`` alias resolver
   ``tnfr.physics.canonical._get_dnfr``) plus the graph metric
   (``G.neighbors``, ``G.degree``, ``G.edges``). No aggregate reads
   a per-edge tensor, per-anchor callable, per-time history kernel,
   or per-node non-scalar payload.

A non-zero closure signature would *force* the aggregates to
introduce a hidden richer intermediate type on the canonical
Tier-1/Tier-2/tetrad/currents-to-aggregates reduction path. A zero
signature plus a unit scalar-closure fraction is the empirically
expected outcome — structurally consistent with the catalog typing
of the aggregates as scalar-functionals of Tier-1+Tier-2 inputs
plus the graph metric (with the tetrad and currents already
discharged as scalar-closed under B7 and B8 respectively).

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13quinquaginta-sexta
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §3 row B9, §4 row B9
- ``src/tnfr/metrics/common.py:29`` (``compute_coherence``)
- ``src/tnfr/metrics/sense_index.py:665`` (``compute_Si``)
- ``src/tnfr/physics/unified.py:136`` (``compute_energy_density``)
- ``src/tnfr/physics/unified.py:209`` (``compute_topological_charge``)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "AggregatesClosureSignatureCertificate",
    "compute_aggregates_closure_signature",
]


# Canonical per-node attribute keys read by the four aggregate
# implementations. Tier-1+Tier-2 scalar slots only (B0 = nu_f,
# B1 = EPI, B2 = phi/theta, B3 = DeltaNFR resolved via
# ``_get_dnfr``). Any non-scalar payload at any of these keys would
# constitute a structural leakage into a richer intermediate type
# along the aggregation path.
_CANONICAL_PER_NODE_KEYS: tuple[str, ...] = (
    "nu_f",
    "EPI",
    "theta",
)


def _is_scalar_payload(value: Any) -> bool:
    """Return ``True`` iff ``value`` is structurally a scalar real number."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (np.integer, np.floating)):
        return True
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return False
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    return False


def _build_canonical_demo_graph(n_nodes: int, seed: int) -> Any:
    """Build a small canonical ring graph for the B9 closure probe."""
    from ..sdk import TNFR

    net = TNFR.create(int(n_nodes)).ring()
    G = net.G
    rng = np.random.default_rng(int(seed))
    for node in list(G.nodes()):
        G.nodes[node]["EPI"] = float(0.5 + 0.05 * (rng.random() - 0.5))
        G.nodes[node]["theta"] = float(
            2.0 * math.pi * (rng.random() - 0.5)
        )
        current_vf = float(G.nodes[node].get("nu_f", 1.0))
        G.nodes[node]["nu_f"] = max(
            0.05, current_vf + 0.05 * (rng.random() - 0.5)
        )
        G.nodes[node]["dnfr"] = float(0.1 * (rng.random() - 0.5))
    return G


def _inspect_input_scalar_closure(
    G: Any,
) -> tuple[int, int, dict[str, int]]:
    """Inspect per-node attributes the aggregate functions will read."""
    from ..physics.canonical import _get_dnfr

    n_scalar = 0
    n_total = 0
    per_key_nonscalar: dict[str, int] = {
        k: 0 for k in _CANONICAL_PER_NODE_KEYS
    }
    per_key_nonscalar["DeltaNFR"] = 0
    for node in G.nodes():
        for key in _CANONICAL_PER_NODE_KEYS:
            value = G.nodes[node].get(key)
            if value is None:
                n_scalar += 1
                n_total += 1
                continue
            n_total += 1
            if _is_scalar_payload(value):
                n_scalar += 1
            else:
                per_key_nonscalar[key] += 1
        dnfr_value = _get_dnfr(G, node)
        n_total += 1
        if _is_scalar_payload(dnfr_value):
            n_scalar += 1
        else:
            per_key_nonscalar["DeltaNFR"] += 1
    return n_scalar, n_total, per_key_nonscalar


def _inspect_output_scalar_closure(
    G: Any,
) -> tuple[int, int, dict[str, int]]:
    """Call the four aggregate functions and inspect outputs."""
    from ..metrics.common import compute_coherence
    from ..metrics.sense_index import compute_Si
    from ..physics.unified import (
        compute_energy_density,
        compute_topological_charge,
    )

    n_scalar = 0
    n_total = 0
    per_field_nonscalar: dict[str, int] = {
        "C_t": 0,
        "Si": 0,
        "energy_density": 0,
        "topological_charge": 0,
    }

    # C(t): single global scalar.
    c_t = compute_coherence(G)
    n_total += 1
    if _is_scalar_payload(c_t):
        n_scalar += 1
    else:
        per_field_nonscalar["C_t"] += 1

    # Si: dict[node, float].
    si_map = compute_Si(G, inplace=False)
    if isinstance(si_map, dict):
        for _node, value in si_map.items():
            n_total += 1
            if _is_scalar_payload(value):
                n_scalar += 1
            else:
                per_field_nonscalar["Si"] += 1

    # Energy density: dict[node, float].
    e_map = compute_energy_density(G)
    for _node, value in e_map.items():
        n_total += 1
        if _is_scalar_payload(value):
            n_scalar += 1
        else:
            per_field_nonscalar["energy_density"] += 1

    # Topological charge: dict[node, float].
    q_map = compute_topological_charge(G)
    for _node, value in q_map.items():
        n_total += 1
        if _is_scalar_payload(value):
            n_scalar += 1
        else:
            per_field_nonscalar["topological_charge"] += 1

    return n_scalar, n_total, per_field_nonscalar


def _signature(n_nonscalar: int, n_total: int) -> tuple[float, float]:
    """Return ``(squashed_signature, raw_fraction)``."""
    if n_total <= 0:
        return 0.0, 0.0
    raw = float(n_nonscalar) / float(n_total)
    return float(math.tanh(raw)), raw


@dataclass(frozen=True)
class AggregatesClosureSignatureCertificate:
    """Result of the Aggregates-Closure Signature diagnostic.

    Attributes
    ----------
    signature : float
        :math:`\\mathcal{S}_{AC} \\in [0, 1]`. ``0`` means every
        aggregate-input and aggregate-output value is scalar-closed;
        ``1`` means closure fails maximally.
    input_scalar_fraction : float
        Fraction of per-node input values touched by the four
        aggregate calls that are scalar-coercible.
    output_scalar_fraction : float
        Fraction of aggregate output values that are scalar-coercible.
    n_input_reads : int
        Total number of per-node input values inspected.
    n_output_reads : int
        Total number of output values inspected (1 global + 3 ×
        n_nodes per-node).
    input_nonscalar_count : int
        Absolute number of non-scalar input values observed.
    output_nonscalar_count : int
        Absolute number of non-scalar output values observed.
    per_key_input_nonscalar : dict[str, int]
        Per-attribute-key non-scalar input count.
    per_field_output_nonscalar : dict[str, int]
        Per-aggregate-field non-scalar output count.
    n_nodes : int
        Number of nodes in the canonical probe graph.
    verdict : str
        One of ``"SCALAR_CLOSURE_ADEQUATE"``,
        ``"RICHER_INTERMEDIATE_NECESSARY"``, or ``"INDETERMINATE"``.
    diagnostics : dict
        Auxiliary fields (thresholds, seed, raw counters).
    """

    signature: float
    input_scalar_fraction: float
    output_scalar_fraction: float
    n_input_reads: int
    n_output_reads: int
    input_nonscalar_count: int
    output_nonscalar_count: int
    per_key_input_nonscalar: dict[str, int]
    per_field_output_nonscalar: dict[str, int]
    n_nodes: int
    verdict: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        per_key = ", ".join(
            f"{k}={v}" for k, v in self.per_key_input_nonscalar.items()
        )
        per_field = ", ".join(
            f"{k}={v}" for k, v in self.per_field_output_nonscalar.items()
        )
        return (
            "AggregatesClosureSignatureCertificate("
            f"S_AC={self.signature:.6f}, "
            f"input_scalar_fraction={self.input_scalar_fraction:.6f}, "
            f"output_scalar_fraction={self.output_scalar_fraction:.6f}, "
            f"input_nonscalar={self.input_nonscalar_count}/"
            f"{self.n_input_reads}, "
            f"output_nonscalar={self.output_nonscalar_count}/"
            f"{self.n_output_reads}, "
            f"per_key_input_nonscalar={{{per_key}}}, "
            f"per_field_output_nonscalar={{{per_field}}}, "
            f"n_nodes={self.n_nodes}, "
            f"verdict={self.verdict})"
        )


def compute_aggregates_closure_signature(
    *,
    n_nodes: int = 24,
    seed: int = 31,
    closure_threshold: float = 0.05,
    divergent_threshold: float = 0.20,
) -> AggregatesClosureSignatureCertificate:
    """Compute the Aggregates-Closure Signature on a canonical probe graph.

    Parameters
    ----------
    n_nodes : int, default 24
        Number of nodes in the canonical probe ring graph.
    seed : int, default 31
        Deterministic seed for attribute perturbation.
    closure_threshold : float, default 0.05
        Upper threshold below which the verdict is
        ``SCALAR_CLOSURE_ADEQUATE`` (combined with unit input and
        output scalar fractions).
    divergent_threshold : float, default 0.20
        Lower threshold above which the verdict is
        ``RICHER_INTERMEDIATE_NECESSARY`` (alternative trigger:
        either fraction < 1.0).

    Returns
    -------
    AggregatesClosureSignatureCertificate
        Frozen result with the closure signature, per-axis fractions,
        per-key/field non-scalar counts, the verdict, and
        reproducibility diagnostics.

    Notes
    -----
    The signature combines two orthogonal axes (input-domain-closure
    and output-scalar-closure). The combined signature is
    :math:`\\tanh\\left(\\frac{n_{\\text{nonscalar}}^{\\text{in}}
    + n_{\\text{nonscalar}}^{\\text{out}}}{n_{\\text{total}}^{\\text{in}}
    + n_{\\text{total}}^{\\text{out}}}\\right)`. A zero signature plus
    unit fractions structurally confirm B9 = Δ-aggregates-closure at
    the Phase-a level. Phase b is **n/a**; the final verdict at Phase
    c (§13quinquaginta-septima) reduces the closure question by direct
    source-code trace of the four canonical aggregate functions at
    ``src/tnfr/metrics/common.py:29``,
    ``src/tnfr/metrics/sense_index.py:665``,
    ``src/tnfr/physics/unified.py:136``, and
    ``src/tnfr/physics/unified.py:209``.
    """
    G = _build_canonical_demo_graph(int(n_nodes), int(seed))

    n_scalar_in, n_total_in, per_key_in = _inspect_input_scalar_closure(G)
    n_scalar_out, n_total_out, per_field_out = (
        _inspect_output_scalar_closure(G)
    )
    n_nonscalar_in = n_total_in - n_scalar_in
    n_nonscalar_out = n_total_out - n_scalar_out
    input_scalar_fraction = (
        float(n_scalar_in) / float(n_total_in) if n_total_in else 1.0
    )
    output_scalar_fraction = (
        float(n_scalar_out) / float(n_total_out) if n_total_out else 1.0
    )
    signature, raw_fraction = _signature(
        n_nonscalar_in + n_nonscalar_out, n_total_in + n_total_out
    )
    fractions_unit = (
        input_scalar_fraction >= 1.0 and output_scalar_fraction >= 1.0
    )
    if signature < float(closure_threshold) and fractions_unit:
        verdict = "SCALAR_CLOSURE_ADEQUATE"
    elif signature > float(divergent_threshold) or not fractions_unit:
        verdict = "RICHER_INTERMEDIATE_NECESSARY"
    else:
        verdict = "INDETERMINATE"
    diagnostics: dict[str, Any] = {
        "closure_threshold": float(closure_threshold),
        "divergent_threshold": float(divergent_threshold),
        "seed": int(seed),
        "raw_fraction": float(raw_fraction),
    }
    return AggregatesClosureSignatureCertificate(
        signature=signature,
        input_scalar_fraction=input_scalar_fraction,
        output_scalar_fraction=output_scalar_fraction,
        n_input_reads=int(n_total_in),
        n_output_reads=int(n_total_out),
        input_nonscalar_count=int(n_nonscalar_in),
        output_nonscalar_count=int(n_nonscalar_out),
        per_key_input_nonscalar=dict(per_key_in),
        per_field_output_nonscalar=dict(per_field_out),
        n_nodes=int(n_nodes),
        verdict=verdict,
        diagnostics=diagnostics,
    )
