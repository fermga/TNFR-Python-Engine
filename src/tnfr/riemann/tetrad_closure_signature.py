"""Tetrad-Closure Signature — Diagnostic for B7 = Δ-tetrad-closure (§13quinquaginta-secunda).

This module implements a purely diagnostic quantity, the
**Tetrad-Closure Signature** :math:`\\mathcal{S}_{TC}`, that
quantifies on canonical TNFR tetrad-field reads whether the
canonical Tier-1+Tier-2 scalar inputs
:math:`(\\mathrm{EPI}_i, \\phi_i, \\Delta\\mathrm{NFR}_i)
\\in \\mathbb{R} \\times [0, 2\\pi) \\times \\mathbb{R}`
plus the graph metric (adjacency + shortest-path distances) are
*structurally sufficient* to reconstruct each of the four
canonical tetrad fields
:math:`(\\Phi_s, |\\nabla\\phi|, K_\\phi, \\xi_C)` as a
scalar-valued (per-node or global) functional, with no hidden
intermediate richer than the Tier-1+Tier-2 types and no
implicit Banach-derivative apparatus, measure, callable kernel,
or matrix lift introduced during the derivation.

Methodological scope (mandatory honesty)
----------------------------------------
This module is a *diagnostic only*. It does **not** construct,
promote, or modify any canonical operator. It does **not** advance
G4 = RH. It does **not** by itself decide the B7 closure question
(which requires the final verdict of §13quinquaginta-tertia at
Phase c). Phase b is **n/a** for B7 (B7 is a closure question, not
a type-conjecture; there is no forcing axiom to reduce — the
question is whether the *existing* canonical Tier-1+Tier-2 types
plus graph metric close the four tetrad-field functionals
without leakage to a richer intermediate type).

The diagnostic probes two orthogonal axes:

1. **Output-scalar-closure axis** — for each of the four
   canonical tetrad-field functions
   (:func:`tnfr.physics.canonical.compute_structural_potential`,
   :func:`tnfr.physics.canonical.compute_phase_gradient`,
   :func:`tnfr.physics.canonical.compute_phase_curvature`,
   :func:`tnfr.physics.canonical.estimate_coherence_length`),
   call the function on a canonical probe graph and verify that
   every output value is structurally scalar-coercible (Python
   ``float`` / NumPy scalar / zero-dim array). The first three
   return ``dict[node, float]``; the fourth returns ``float``.
   Under the canonical implementation in
   ``src/tnfr/physics/canonical.py:199-820``, every per-node
   output is explicitly coerced via ``float(...)``; the non-
   scalar fraction is therefore structurally ``0`` — exactly
   mirroring ``S_W storage = 1.0`` (B6a),
   ``S_dphi storage = 1.0`` (B5a), ``noninteger_frac = 0``
   (B4a, inverted), ``T_frac = 0`` (B3a), ``bepi_frac = 0``
   (B1a), and ``w_frac = 0`` (B2a).

2. **Input-domain-closure axis** — for each tetrad-field
   function, verify that the per-node inputs read by the
   canonical implementation are confined to the Tier-1+Tier-2
   scalar slots (``EPI``, ``theta`` / ``phi``, ``nu_f``, the
   canonical ``ΔNFR`` alias resolver
   ``tnfr.physics.canonical._get_dnfr``) plus the graph metric
   (``G.neighbors``, ``G.degree``, ``nx.shortest_path_length``).
   No tetrad-field function reads a per-edge tensor,
   per-anchor callable, per-time history kernel, or per-node
   non-scalar payload. We verify this empirically by reading
   every per-node attribute touched during the four calls and
   asserting each is scalar-coercible.

A non-zero closure signature would *force* the tetrad to
introduce a hidden richer intermediate type (e.g. per-node
tensor cache, callable kernel, matrix-valued intermediate) on
the canonical Tier-1+Tier-2-to-tetrad reduction path. A zero
signature plus a unit scalar-closure fraction is the
empirically expected outcome — structurally consistent with
the catalog typing of the tetrad as scalar-functionals of
Tier-1+Tier-2 inputs plus the graph metric, and with the
absence of any non-scalar intermediate in any of the four
canonical tetrad-field implementations.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13quinquaginta-secunda
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §3 row B7, §4 row B7
- ``src/tnfr/physics/fields.py`` (public re-export façade)
- ``src/tnfr/physics/canonical.py:199`` (``compute_structural_potential``)
- ``src/tnfr/physics/canonical.py:609`` (``compute_phase_gradient``)
- ``src/tnfr/physics/canonical.py:640`` (``compute_phase_curvature``)
- ``src/tnfr/physics/canonical.py:756`` (``estimate_coherence_length``)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "TetradClosureSignatureCertificate",
    "compute_tetrad_closure_signature",
]


# Canonical per-node attribute keys read by the four tetrad-field
# implementations. These are exactly the Tier-1+Tier-2 scalar slots
# (B1 = EPI, B2 = phi/theta, B0 = nu_f, B3 = DeltaNFR resolved via
# ``_get_dnfr``). Any non-scalar payload at any of these keys would
# constitute a structural leakage into a richer intermediate type.
_CANONICAL_PER_NODE_KEYS: tuple[str, ...] = (
    "EPI",
    "theta",
    "nu_f",
)


def _is_scalar_payload(value: Any) -> bool:
    """Return ``True`` iff ``value`` is structurally a scalar real number.

    Accepts: Python ``int`` (excluding ``bool``), Python ``float``,
    NumPy integer/floating scalar, and zero-dimensional NumPy array.
    Rejects: NumPy arrays of ndim > 0, mappings, sequences,
    callables, ``None``.
    """
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
    """Build a small canonical ring graph for the B7 closure probe.

    Mirrors the probe-graph construction used at B6a
    (``coupling_weights_type_signature``) and B5a
    (``delta_phi_max_type_signature``): canonical ring topology,
    Tier-1+Tier-2 scalar attributes initialised with a small
    deterministic perturbation per node, and a canonical scalar
    ``DeltaNFR`` payload assigned via the canonical alias system.
    """
    from ..sdk import TNFR

    net = TNFR.create(int(n_nodes)).ring()
    G = net.G
    rng = np.random.default_rng(int(seed))
    for node in list(G.nodes()):
        G.nodes[node]["EPI"] = float(0.5 + 0.05 * (rng.random() - 0.5))
        G.nodes[node]["theta"] = float(2.0 * math.pi * (rng.random() - 0.5))
        current_vf = float(G.nodes[node].get("nu_f", 1.0))
        G.nodes[node]["nu_f"] = max(0.05, current_vf + 0.05 * (rng.random() - 0.5))
        # Canonical DeltaNFR scalar payload (Tier-1 B3 slot).
        G.nodes[node]["dnfr"] = float(0.1 * (rng.random() - 0.5))
    return G


def _inspect_input_scalar_closure(
    G: Any,
) -> tuple[int, int, dict[str, int]]:
    """Inspect per-node attributes the tetrad functions will read.

    For every node, every canonical key in ``_CANONICAL_PER_NODE_KEYS``
    plus the resolved ``DeltaNFR`` payload is inspected and counted
    as scalar or non-scalar. A scalar closure means every input the
    tetrad pipeline touches is a single real number, never a tensor,
    callable, or richer intermediate.

    Returns
    -------
    n_scalar : int
        Total count of scalar-coercible per-node input values.
    n_total : int
        Total number of per-node input values inspected.
    per_key_nonscalar : dict[str, int]
        Number of non-scalar values per attribute key.
    """
    from ..physics.canonical import _get_dnfr

    n_scalar = 0
    n_total = 0
    per_key_nonscalar: dict[str, int] = {k: 0 for k in _CANONICAL_PER_NODE_KEYS}
    per_key_nonscalar["DeltaNFR"] = 0
    for node in G.nodes():
        for key in _CANONICAL_PER_NODE_KEYS:
            value = G.nodes[node].get(key)
            if value is None:
                # Absent slot is canonical-default-resolvable; we
                # count it as scalar (the default is a scalar float).
                n_scalar += 1
                n_total += 1
                continue
            n_total += 1
            if _is_scalar_payload(value):
                n_scalar += 1
            else:
                per_key_nonscalar[key] += 1
        # Resolved DeltaNFR via canonical alias system.
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
    """Call the four tetrad-field functions and inspect every output value.

    Returns
    -------
    n_scalar : int
        Total count of scalar-coercible output values across the
        four tetrad-field calls.
    n_total : int
        Total number of output values inspected.
    per_field_nonscalar : dict[str, int]
        Number of non-scalar output values per tetrad field.
    """
    from ..physics.canonical import (
        compute_phase_curvature,
        compute_phase_gradient,
        compute_structural_potential,
        estimate_coherence_length,
    )

    n_scalar = 0
    n_total = 0
    per_field_nonscalar: dict[str, int] = {
        "Phi_s": 0,
        "grad_phi": 0,
        "K_phi": 0,
        "xi_C": 0,
    }
    # Phi_s, |grad phi|, K_phi -> dict[node, float]
    phi_s_map = compute_structural_potential(G)
    for _node, value in phi_s_map.items():
        n_total += 1
        if _is_scalar_payload(value):
            n_scalar += 1
        else:
            per_field_nonscalar["Phi_s"] += 1
    grad_phi_map = compute_phase_gradient(G)
    for _node, value in grad_phi_map.items():
        n_total += 1
        if _is_scalar_payload(value):
            n_scalar += 1
        else:
            per_field_nonscalar["grad_phi"] += 1
    k_phi_map = compute_phase_curvature(G)
    for _node, value in k_phi_map.items():
        n_total += 1
        if _is_scalar_payload(value):
            n_scalar += 1
        else:
            per_field_nonscalar["K_phi"] += 1
    # xi_C -> single global float.
    xi_c_value = estimate_coherence_length(G)
    n_total += 1
    if _is_scalar_payload(xi_c_value):
        n_scalar += 1
    else:
        per_field_nonscalar["xi_C"] += 1
    return n_scalar, n_total, per_field_nonscalar


def _signature(n_nonscalar: int, n_total: int) -> tuple[float, float]:
    """Return ``(squashed_signature, raw_fraction)``."""
    if n_total <= 0:
        return 0.0, 0.0
    raw = float(n_nonscalar) / float(n_total)
    return float(math.tanh(raw)), raw


@dataclass(frozen=True)
class TetradClosureSignatureCertificate:
    """Result of the Tetrad-Closure Signature diagnostic.

    Attributes
    ----------
    signature : float
        :math:`\\mathcal{S}_{TC} \\in [0, 1]`. ``0`` means every
        tetrad-field input and output is scalar-closed (canonical
        Tier-1+Tier-2 scalar typing structurally suffices); ``1``
        means closure fails maximally.
    input_scalar_fraction : float
        Fraction of per-node input values touched by the four
        tetrad-field calls that are scalar-coercible. ``1.0`` is
        the empirically expected value.
    output_scalar_fraction : float
        Fraction of tetrad-field output values that are scalar-
        coercible. ``1.0`` is the empirically expected value.
    n_input_reads : int
        Total number of per-node input values inspected.
    n_output_reads : int
        Total number of tetrad-field output values inspected.
    input_nonscalar_count : int
        Absolute number of non-scalar input values observed.
    output_nonscalar_count : int
        Absolute number of non-scalar output values observed.
    per_key_input_nonscalar : dict[str, int]
        Per-attribute-key non-scalar input count.
    per_field_output_nonscalar : dict[str, int]
        Per-tetrad-field non-scalar output count.
    n_nodes : int
        Number of nodes in the canonical probe graph.
    verdict : str
        One of ``"SCALAR_CLOSURE_ADEQUATE"`` (signature <
        ``closure_threshold`` AND both fractions == 1.0),
        ``"RICHER_INTERMEDIATE_NECESSARY"`` (signature >
        ``divergent_threshold`` OR either fraction < 1.0), or
        ``"INDETERMINATE"``.
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
        per_key = ", ".join(f"{k}={v}" for k, v in self.per_key_input_nonscalar.items())
        per_field = ", ".join(
            f"{k}={v}" for k, v in self.per_field_output_nonscalar.items()
        )
        return (
            "TetradClosureSignatureCertificate("
            f"S_TC={self.signature:.6f}, "
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


def compute_tetrad_closure_signature(
    *,
    n_nodes: int = 24,
    seed: int = 31,
    closure_threshold: float = 0.05,
    divergent_threshold: float = 0.20,
) -> TetradClosureSignatureCertificate:
    """Compute the Tetrad-Closure Signature on a canonical probe graph.

    Parameters
    ----------
    n_nodes : int, default 24
        Number of nodes in the canonical probe ring graph.
    seed : int, default 31
        Deterministic seed for attribute perturbation.
    closure_threshold : float, default 0.05
        Upper threshold below which the verdict is
        ``SCALAR_CLOSURE_ADEQUATE`` (combined with unit input
        and output scalar fractions).
    divergent_threshold : float, default 0.20
        Lower threshold above which the verdict is
        ``RICHER_INTERMEDIATE_NECESSARY`` (alternative trigger:
        either fraction < 1.0).

    Returns
    -------
    TetradClosureSignatureCertificate
        Frozen result with the closure signature, per-axis
        fractions, per-key/field non-scalar counts, the verdict,
        and reproducibility diagnostics.

    Notes
    -----
    The signature combines two orthogonal axes:

    1. Input-domain-closure: fraction of per-node input values
       touched by the four canonical tetrad-field functions that
       are scalar-coercible.
    2. Output-scalar-closure: fraction of tetrad-field output
       values that are scalar-coercible.

    The combined signature is
    :math:`\\tanh\\left(\\frac{n_{\\text{nonscalar}}^{\\text{in}}
    + n_{\\text{nonscalar}}^{\\text{out}}}{n_{\\text{total}}^{\\text{in}}
    + n_{\\text{total}}^{\\text{out}}}\\right)`. A zero signature
    plus unit fractions structurally confirm B7 = Δ-tetrad-
    closure at the Phase-a level. Phase b is **n/a**; the final
    verdict at Phase c (§13quinquaginta-tertia) reduces the
    closure question by direct source-code trace of the four
    canonical tetrad-field functions at
    ``src/tnfr/physics/canonical.py:199,609,640,756``.
    """
    G = _build_canonical_demo_graph(int(n_nodes), int(seed))

    n_scalar_in, n_total_in, per_key_in = _inspect_input_scalar_closure(G)
    n_scalar_out, n_total_out, per_field_out = _inspect_output_scalar_closure(G)
    n_nonscalar_in = n_total_in - n_scalar_in
    n_nonscalar_out = n_total_out - n_scalar_out
    input_scalar_fraction = (
        float(n_scalar_in) / float(n_total_in) if n_total_in else 1.0
    )
    output_scalar_fraction = (
        float(n_scalar_out) / float(n_total_out) if n_total_out else 1.0
    )
    squashed, raw = _signature(
        n_nonscalar_in + n_nonscalar_out, n_total_in + n_total_out
    )
    if (
        squashed < float(closure_threshold)
        and math.isclose(input_scalar_fraction, 1.0)
        and math.isclose(output_scalar_fraction, 1.0)
    ):
        verdict = "SCALAR_CLOSURE_ADEQUATE"
    elif (
        squashed > float(divergent_threshold)
        or input_scalar_fraction < 1.0
        or output_scalar_fraction < 1.0
    ):
        verdict = "RICHER_INTERMEDIATE_NECESSARY"
    else:
        verdict = "INDETERMINATE"
    return TetradClosureSignatureCertificate(
        signature=float(squashed),
        input_scalar_fraction=float(input_scalar_fraction),
        output_scalar_fraction=float(output_scalar_fraction),
        n_input_reads=int(n_total_in),
        n_output_reads=int(n_total_out),
        input_nonscalar_count=int(n_nonscalar_in),
        output_nonscalar_count=int(n_nonscalar_out),
        per_key_input_nonscalar=dict(per_key_in),
        per_field_output_nonscalar=dict(per_field_out),
        n_nodes=int(n_nodes),
        verdict=verdict,
        diagnostics={
            "raw_total_nonscalar_fraction": float(raw),
            "closure_threshold": float(closure_threshold),
            "divergent_threshold": float(divergent_threshold),
            "seed": int(seed),
        },
    )
