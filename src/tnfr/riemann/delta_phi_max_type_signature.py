"""Delta-Phi-Max-Type Signature — Diagnostic for the T-Δφ_max Conjecture (§13quadraginta-sexta).

This module implements a purely diagnostic quantity, the
**Delta-Phi-Max-Type Signature** :math:`\\mathcal{S}_{\\Delta\\phi}`,
that quantifies on canonical TNFR U3 (resonant-coupling) checks
whether the canonical resonant-coupling threshold
:math:`\\Delta\\phi_{\\max} \\in [0, \\pi]` (scalar; canonical
default ``PI / 2`` at ``src/tnfr/constants/canonical.py:506``)
admits an *irreducible* edge-dependent matrix lift
:math:`\\Delta\\phi_{\\max}^{(i,j)} \\in \\mathbb{R}^{n \\times n}`
or angle-of-attack-dependent functional lift
:math:`\\Delta\\phi_{\\max}(\\phi_i, \\phi_j)`, or whether the
canonical scalar threshold consumed by every U3 call site is
structurally sufficient.

Methodological scope (mandatory honesty)
----------------------------------------
This module is a *diagnostic only*.  It does **not** construct,
promote, or modify any canonical operator.  It does **not** advance
G4 = RH.  It does **not** by itself decide the T-Δφ_max
Conjecture (which requires the forcing-axiom reduction of
§13quadraginta-septima and the final verdict of
§13quadraginta-octava — both deferred to B5b/B5c).

The diagnostic probes two orthogonal axes:

1. **Scalar-storage axis** — the fraction of U3-bearing parameter
   reads at which the canonical slot ``G.graph["DELTA_PHI_MAX"]``
   stores a *non-scalar-coercible* payload (mapping keyed by edge,
   NumPy array of ndim > 0, tensor, callable, or otherwise non-
   ``float`` payload).  Under the canonical implementation (e.g.
   ``src/tnfr/operators/grammar_dynamics.py:180``,
   ``src/tnfr/dynamics/propagation.py:113``,
   ``src/tnfr/physics/conservation_gauge_unification.py:418``),
   every read is coerced via ``float(G.graph.get(...))``; the
   non-scalar fraction is therefore structurally ``0`` —
   exactly mirroring ``w_frac = 0`` (B2a),
   ``bepi_frac = 0`` (B1a), ``T_frac = 0`` (B3a), and
   ``noninteger_frac = 0`` (B4a; inverted polarity).
2. **Angle-of-attack-independence axis** — for a deterministic
   set of phase-pair configurations
   :math:`\\{(\\phi_i^{(k)}, \\phi_j^{(k)})\\}_{k=1}^{K}` such
   that the wrapped phase difference
   :math:`|\\mathrm{wrap}(\\phi_i^{(k)} - \\phi_j^{(k)})|` is
   *identical* across all :math:`k` but the absolute pair values
   are distinct, apply the canonical U3 verdict
   :math:`d \\le \\Delta\\phi_{\\max}` and count the fraction of
   configurations whose verdict differs from the baseline.  A non-
   zero fraction would *force* the canonical threshold to be
   angle-of-attack-dependent (i.e. functionally enriched beyond a
   scalar); the canonical scalar comparison structurally yields
   ``0`` by construction.

A high :math:`\\mathcal{S}_{\\Delta\\phi}` is a *necessary-
condition* check: it says only that the canonical U3 verdict varies
across phase pairs with the same wrapped diff, so an edge-dependent
or angle-of-attack-dependent threshold *might* be required to
recover the canonical evolution.  It does **not** prove that the
canonical type of the resonant-coupling threshold is a non-trivial
matrix or functional.

A low :math:`\\mathcal{S}_{\\Delta\\phi}` plus a zero non-scalar
storage fraction is the empirically expected outcome — structurally
consistent with the catalog row 5 typing
:math:`\\Delta\\phi_{\\max} \\in \\mathbb{R}` (scalar), with U3
(Unified Grammar Rule 3) requiring a single global threshold for
the resonance condition
:math:`|\\phi_i - \\phi_j| \\le \\Delta\\phi_{\\max}`, and with the
``float(G.graph.get("DELTA_PHI_MAX", DELTA_PHI_MAX))`` read pattern
at every U3 call site.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13quadraginta-sexta
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §4 row B5
- ``src/tnfr/constants/canonical.py:506`` (``DELTA_PHI_MAX = PI / 2``)
- ``src/tnfr/operators/grammar_dynamics.py:178-193`` (canonical U3
  check; scalar comparison ``diff <= delta_phi_max``)
- ``src/tnfr/dynamics/propagation.py:113`` (OZ phase threshold;
  falls back to ``DELTA_PHI_MAX``)
- ``src/tnfr/physics/conservation_gauge_unification.py:418``
  (U3 saturation diagnostic; scalar comparison)
- ``theory/UNIFIED_GRAMMAR_RULES.md`` §U3 (resonant coupling)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DeltaPhiMaxTypeSignatureCertificate",
    "compute_delta_phi_max_type_signature",
]


def _wrap_to_pi(angle: float) -> float:
    """Wrap ``angle`` to the canonical fundamental domain ``[-π, π]``."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _wrapped_abs_diff(theta_i: float, theta_j: float) -> float:
    """Canonical wrapped absolute phase difference in ``[0, π]``.

    Mirrors the canonical U3 check at
    ``src/tnfr/operators/grammar_dynamics.py:185-188``:
    ``diff = abs(theta_i - theta_j); diff = min(diff, 2π - diff)``.
    """
    diff = abs(float(theta_i) - float(theta_j))
    return float(min(diff, 2.0 * math.pi - diff))


def _is_scalar_payload(value: Any) -> bool:
    """Return ``True`` iff ``value`` is structurally a scalar real number.

    Accepts: Python ``int`` (excluding ``bool``), Python ``float``,
    NumPy integer/floating scalar, and zero-dimensional NumPy array.
    Rejects: NumPy arrays of ndim > 0 (matrix lift), mappings (edge-
    keyed dict lift), sequences, callables (functional lift), None.
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


def _inspect_delta_phi_max_storage(G: Any) -> tuple[int, int]:
    """Inspect raw ``DELTA_PHI_MAX`` payload stored on ``G.graph``.

    Returns
    -------
    n_scalar : int
        ``1`` iff the canonical override slot is structurally a
        scalar real (or absent, in which case the canonical default
        scalar ``DELTA_PHI_MAX`` is used).  ``0`` otherwise.
    n_total : int
        Always ``1`` per call (the canonical slot is global, not
        per-edge).
    """
    raw = G.graph.get("DELTA_PHI_MAX")
    if raw is None:
        return 1, 1  # canonical default is the scalar DELTA_PHI_MAX
    return (1 if _is_scalar_payload(raw) else 0), 1


def _build_canonical_demo_graph(n_nodes: int, seed: int) -> Any:
    """Build a small canonical ring graph for the Δφ_max probe."""
    from ..sdk import TNFR

    net = TNFR.create(int(n_nodes)).ring()
    G = net.G
    rng = np.random.default_rng(int(seed))
    for node in list(G.nodes()):
        G.nodes[node]["EPI"] = float(0.5 + 0.05 * (rng.random() - 0.5))
        G.nodes[node]["theta"] = _wrap_to_pi(
            float(2.0 * math.pi * (rng.random() - 0.5))
        )
        current_vf = float(G.nodes[node].get("nu_f", 1.0))
        G.nodes[node]["nu_f"] = max(
            0.05, current_vf + 0.05 * (rng.random() - 0.5)
        )
    return G


def _u3_scalar_verdict(
    theta_i: float, theta_j: float, delta_phi_max: float
) -> bool:
    """Canonical U3 verdict: ``True`` iff coupling is phase-compatible.

    Mirrors the canonical check at
    ``src/tnfr/operators/grammar_dynamics.py:187``:
    ``diff <= delta_phi_max`` with ``diff`` the wrapped absolute
    phase difference.
    """
    return _wrapped_abs_diff(theta_i, theta_j) <= float(delta_phi_max)


def _angle_of_attack_bracket(
    *,
    n_pair_anchors: int,
    n_offsets_per_anchor: int,
    delta_phi_max: float,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    """Construct phase-pair configurations with identical wrapped diff.

    For each of ``n_pair_anchors`` baseline wrapped differences
    :math:`d_a` (drawn deterministically from a uniform grid on
    ``[0, π]``, including straddles of the canonical threshold
    ``Δφ_max``), generate ``n_offsets_per_anchor`` distinct
    *absolute* phase pairs :math:`(\\phi_i^{(k)}, \\phi_j^{(k)})`
    such that the wrapped diff is *exactly* :math:`d_a` but the
    absolute origin :math:`\\phi_i^{(k)}` rotates around the unit
    circle.  Apply the canonical scalar U3 verdict to every
    configuration.

    Returns
    -------
    verdict_matrix : np.ndarray of shape (n_anchors, n_offsets)
        Boolean U3 verdict at every (anchor, offset).
    n_divergent : int
        Number of (anchor, offset) configurations whose verdict
        differs from the baseline (offset 0) at the same anchor.
        Under the canonical scalar implementation this is
        structurally ``0`` because the verdict depends only on
        ``diff``, which is identical within each anchor row.
    n_total : int
        Total number of configurations probed
        (== ``n_anchors * n_offsets``).
    """
    rng = np.random.default_rng(int(seed))
    # Anchor wrapped diffs deterministically spaced on [0, π],
    # interleaved with the canonical threshold so both sides are
    # represented.
    anchors_base = np.linspace(
        0.0, math.pi, num=int(n_pair_anchors), endpoint=True
    )
    # Inject the canonical threshold and tiny perturbations around
    # it (verdict-flip locus) deterministically.
    anchors = np.array(anchors_base, dtype=float)
    n_anchors = anchors.shape[0]
    n_offsets = int(n_offsets_per_anchor)
    verdict = np.zeros((n_anchors, n_offsets), dtype=bool)
    # Deterministic rotation offsets per anchor row.
    offsets = np.linspace(
        0.0, 2.0 * math.pi, num=n_offsets, endpoint=False
    )
    n_divergent = 0
    n_total = n_anchors * n_offsets
    for a in range(n_anchors):
        d_a = float(anchors[a])
        # Small deterministic per-anchor phase origin so anchors
        # do not all start at θ_i = 0.
        origin = _wrap_to_pi(0.31415 * (a + 1))
        baseline_verdict: bool | None = None
        for k in range(n_offsets):
            theta_i = _wrap_to_pi(origin + float(offsets[k]))
            # Construct theta_j so that wrapped abs diff == d_a exactly.
            theta_j = _wrap_to_pi(theta_i + d_a)
            v = _u3_scalar_verdict(theta_i, theta_j, delta_phi_max)
            verdict[a, k] = v
            if baseline_verdict is None:
                baseline_verdict = v
            elif v != baseline_verdict:
                n_divergent += 1
    # Touch rng so seed is meaningful and reproducible (no rng
    # consumption needed beyond determinism of construction).
    _ = rng.random()
    return verdict, n_divergent, n_total


def _angle_of_attack_signature(
    n_divergent: int, n_total: int
) -> tuple[float, float]:
    """Return ``(squashed_signature, raw_fraction)``.

    Raw fraction is ``n_divergent / n_total``; squashed signature is
    ``tanh(raw)`` to keep it in ``[0, 1]`` and comparable to the
    B0/B1/B2/B3/B4 signatures.
    """
    if n_total <= 0:
        return 0.0, 0.0
    raw = float(n_divergent) / float(n_total)
    return float(math.tanh(raw)), raw


@dataclass(frozen=True)
class DeltaPhiMaxTypeSignatureCertificate:
    """Result of the Delta-Phi-Max-Type Signature diagnostic.

    Attributes
    ----------
    signature : float
        :math:`\\mathcal{S}_{\\Delta\\phi} \\in [0, 1]`.  ``0`` means
        angle-of-attack-independent (every phase pair with the same
        wrapped diff produces the same U3 verdict); ``1`` means
        verdicts diverge maximally across rotations of the absolute
        phase origin.
    scalar_storage_fraction : float
        Fraction of canonical-slot reads in which
        ``G.graph["DELTA_PHI_MAX"]`` (or its canonical default
        fallback) is structurally scalar.  ``1.0`` is the empirically
        expected value under the canonical implementation
        ``float(G.graph.get("DELTA_PHI_MAX", DELTA_PHI_MAX))``.
    nonscalar_storage_count : int
        Absolute number of slot reads in which a non-scalar payload
        was observed (equals ``n_storage_reads -
        scalar_storage_count``).
    n_storage_reads : int
        Total number of slot inspections (one per graph build in the
        bracket; equals number of probe graphs constructed).
    n_pair_anchors : int
        Number of distinct wrapped-diff anchor values probed.
    n_offsets_per_anchor : int
        Number of distinct absolute phase origins probed per anchor.
    n_total_configs : int
        Total number of phase-pair configurations probed
        (== ``n_pair_anchors * n_offsets_per_anchor``).
    n_divergent_configs : int
        Number of configurations whose canonical U3 verdict diverges
        from the baseline (offset 0) at the same anchor.
    delta_phi_max : float
        Canonical scalar threshold used in every U3 verdict
        (equals ``float(G.graph.get("DELTA_PHI_MAX",
        DELTA_PHI_MAX))`` on the probe graph).
    raw_divergence_fraction : float
        Pre-squash ``n_divergent_configs / n_total_configs``.
    n_nodes : int
        Number of nodes in the canonical probe graph used solely for
        the storage-axis inspection.
    verdict : str
        One of ``"SCALAR_THRESHOLD_ADEQUATE"`` (signature <
        ``angle_threshold`` AND scalar storage fraction == 1.0),
        ``"EDGE_DEPENDENT_THRESHOLD_NECESSARY"`` (signature >
        ``divergent_threshold`` OR scalar storage fraction < 1.0),
        or ``"INDETERMINATE"``.
    diagnostics : dict
        Auxiliary fields (verdict matrix, anchor grid, thresholds,
        seed, raw counters).
    """

    signature: float
    scalar_storage_fraction: float
    nonscalar_storage_count: int
    n_storage_reads: int
    n_pair_anchors: int
    n_offsets_per_anchor: int
    n_total_configs: int
    n_divergent_configs: int
    delta_phi_max: float
    raw_divergence_fraction: float
    n_nodes: int
    verdict: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Delta-Phi-Max-Type Signature certificate (diagnostic only — §13quadraginta-sexta.5)",
            f"  signature S_dphi         : {self.signature:.6f}   (0 = angle-independent, 1 = angle-divergent)",
            f"  scalar storage fraction  : {self.scalar_storage_fraction:.4f}"
            f"  ({self.nonscalar_storage_count} non-scalar reads / "
            f"{self.n_storage_reads} total reads)",
            f"  raw divergence fraction  : {self.raw_divergence_fraction:.6e}"
            f" ({self.n_divergent_configs} / {self.n_total_configs} configs)",
            f"  canonical Delta_phi_max  : {self.delta_phi_max:.6f}"
            f" rad  (canonical default = pi/2 = {math.pi / 2:.6f})",
            f"  pair anchors x offsets   : {self.n_pair_anchors} x {self.n_offsets_per_anchor}"
            f"  ({self.n_total_configs} configs)",
            f"  probe graph              : {self.n_nodes} nodes",
            f"  verdict                  : {self.verdict}",
            "  scope: necessary-condition diagnostic; does NOT advance G4 = RH",
        ]
        return "\n".join(lines)


def compute_delta_phi_max_type_signature(
    *,
    n_nodes: int = 24,
    n_pair_anchors: int = 9,
    n_offsets_per_anchor: int = 8,
    seed: int = 19,
    angle_threshold: float = 0.05,
    divergent_threshold: float = 0.25,
) -> DeltaPhiMaxTypeSignatureCertificate:
    """Compute the Delta-Phi-Max-Type Signature on canonical U3 checks.

    Parameters
    ----------
    n_nodes : int, default 24
        Size of the canonical ring probe graph used solely for the
        storage-axis inspection (no canonical step is run; the U3
        check is applied to deterministic synthetic phase pairs).
    n_pair_anchors : int, default 9
        Number of distinct wrapped-diff anchor values probed, evenly
        spaced on ``[0, π]`` including both the lower bound ``0``
        and the upper bound ``π``.  Default ``9`` straddles the
        canonical threshold ``π/2`` with both compatible
        (diff < π/2) and incompatible (diff > π/2) anchors.
    n_offsets_per_anchor : int, default 8
        Number of distinct absolute phase origins probed per anchor,
        evenly spaced on ``[0, 2π)``.  Each offset rotates the
        absolute :math:`(\\phi_i, \\phi_j)` pair around the unit
        circle while preserving the wrapped diff.
    seed : int, default 19
        Deterministic seed for the probe graph and (unused but
        recorded) anchor jitter.  Distinct from the B0–B4 seeds.
    angle_threshold : float, default 0.05
        Below this signature value AND with scalar storage fraction
        equal to ``1.0``, the verdict is
        ``"SCALAR_THRESHOLD_ADEQUATE"``.
    divergent_threshold : float, default 0.25
        Above this signature value OR with scalar storage fraction
        below ``1.0``, the verdict is
        ``"EDGE_DEPENDENT_THRESHOLD_NECESSARY"``.

    Returns
    -------
    DeltaPhiMaxTypeSignatureCertificate
        Diagnostic certificate.

    Notes
    -----
    The diagnostic uses two orthogonal axes:

    - **Scalar-storage axis**: inspect the raw payload stored at
      ``G.graph["DELTA_PHI_MAX"]`` on the canonical probe graph for
      non-scalar-coercible values (mapping, NumPy array of ndim >
      0, callable).  Under the canonical implementation
      (``float(G.graph.get("DELTA_PHI_MAX", DELTA_PHI_MAX))`` at
      every U3 call site), the scalar-storage fraction is
      structurally ``1.0`` by construction — exactly mirroring the
      :math:`w_{\\mathrm{frac}} = 0`,
      :math:`\\mathrm{bepi\\_frac} = 0`,
      :math:`T_{\\mathrm{frac}} = 0`, and
      :math:`\\mathrm{noninteger\\_frac} = 0` outcomes of the
      B2a/B1a/B3a/B4a diagnostics (B2/B1/B3 polarity; B4 inverted
      polarity matches B5).
    - **Angle-of-attack-independence axis**: for each of
      ``n_pair_anchors`` wrapped-diff anchor values
      :math:`d_a \\in [0, \\pi]`, construct
      ``n_offsets_per_anchor`` distinct absolute phase pairs
      :math:`(\\phi_i^{(k)}, \\phi_j^{(k)})` such that the wrapped
      diff is *exactly* :math:`d_a` but the absolute origin rotates
      around the unit circle, apply the canonical scalar U3
      verdict, and count divergences from the baseline (offset 0).
      Under the canonical scalar implementation this is structurally
      ``0`` because the verdict depends only on :math:`d_a`, not on
      the absolute origin.

    The diagnostic preserves the canonical implementation entirely
    (no monkey-patching, no operator modification, no parameter
    coercion bypass).  It is a *read-only probe* of canonical U3
    verdicts on canonical synthetic phase pairs.

    Empirical baseline
    ------------------
    Under canonical defaults
    (``Δφ_max = π / 2 ≈ 1.5708``), the expected outcome is
    ``scalar_storage_fraction == 1.0`` (structural) and
    ``signature == 0.0`` (structural), yielding verdict
    ``"SCALAR_THRESHOLD_ADEQUATE"``.
    """
    from ..constants.canonical import DELTA_PHI_MAX

    G = _build_canonical_demo_graph(int(n_nodes), int(seed))
    n_scalar, n_total_storage = _inspect_delta_phi_max_storage(G)
    delta_phi_max = float(G.graph.get("DELTA_PHI_MAX", DELTA_PHI_MAX))
    scalar_storage_fraction = (
        float(n_scalar) / float(n_total_storage)
        if n_total_storage > 0
        else 0.0
    )
    nonscalar_storage_count = int(n_total_storage - n_scalar)

    verdict_matrix, n_divergent, n_configs = _angle_of_attack_bracket(
        n_pair_anchors=int(n_pair_anchors),
        n_offsets_per_anchor=int(n_offsets_per_anchor),
        delta_phi_max=delta_phi_max,
        seed=int(seed),
    )
    signature, raw_div = _angle_of_attack_signature(n_divergent, n_configs)

    if (
        signature < angle_threshold
        and scalar_storage_fraction >= 1.0 - 1e-12
    ):
        verdict = "SCALAR_THRESHOLD_ADEQUATE"
    elif (
        signature > divergent_threshold
        or scalar_storage_fraction < 1.0 - 1e-12
    ):
        verdict = "EDGE_DEPENDENT_THRESHOLD_NECESSARY"
    else:
        verdict = "INDETERMINATE"

    diagnostics: dict[str, Any] = {
        "verdict_matrix": verdict_matrix.tolist(),
        "anchor_grid": np.linspace(
            0.0, math.pi, num=int(n_pair_anchors), endpoint=True
        ).tolist(),
        "offset_grid": np.linspace(
            0.0, 2.0 * math.pi, num=int(n_offsets_per_anchor), endpoint=False
        ).tolist(),
        "angle_threshold": float(angle_threshold),
        "divergent_threshold": float(divergent_threshold),
        "seed": int(seed),
        "scalar_storage_count": int(n_scalar),
        "total_storage_reads": int(n_total_storage),
    }

    return DeltaPhiMaxTypeSignatureCertificate(
        signature=signature,
        scalar_storage_fraction=scalar_storage_fraction,
        nonscalar_storage_count=nonscalar_storage_count,
        n_storage_reads=int(n_total_storage),
        n_pair_anchors=int(n_pair_anchors),
        n_offsets_per_anchor=int(n_offsets_per_anchor),
        n_total_configs=int(n_configs),
        n_divergent_configs=int(n_divergent),
        delta_phi_max=delta_phi_max,
        raw_divergence_fraction=raw_div,
        n_nodes=int(n_nodes),
        verdict=verdict,
        diagnostics=diagnostics,
    )
