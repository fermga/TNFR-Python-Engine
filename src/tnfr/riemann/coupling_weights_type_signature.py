"""Coupling-Weights-Type Signature — Diagnostic for the T-W Conjecture (§13quadraginta-nona).

This module implements a purely diagnostic quantity, the
**Coupling-Weights-Type Signature** :math:`\\mathcal{S}_{W}`, that
quantifies on canonical TNFR mixing-weight reads whether the
canonical scalar-dict typing
:math:`\\{w_{c} \\in \\mathbb{R} : c \\in C\\}_{\\text{global}}`
(one global scalar per component name :math:`c`; canonical defaults
at ``src/tnfr/config/defaults_core.py:57`` for ``DNFR_WEIGHTS``,
``defaults_core.py:65`` for ``SI_WEIGHTS``, and
``defaults_core.py:150`` for ``SELECTOR_WEIGHTS``) admits an
*irreducible* node-indexed enrichment
:math:`\\{w_{c}^{(i)}\\}_{i \\in V}`, edge-indexed enrichment
:math:`\\{w_{c}^{(i,j)}\\}_{(i,j) \\in E}`, matrix lift
:math:`W_{c} \\in \\mathbb{R}^{n \\times n}`, or callable
functional :math:`w_{c}(\\cdot)`, or whether the canonical global
scalar typing consumed by every canonical mixing site is
structurally sufficient.

Methodological scope (mandatory honesty)
----------------------------------------
This module is a *diagnostic only*. It does **not** construct,
promote, or modify any canonical operator. It does **not** advance
G4 = RH. It does **not** by itself decide the T-W Conjecture
(which requires the forcing-axiom reduction of
§13quinquaginta and the final verdict of §13quinquaginta-prima —
both deferred to B6b/B6c).

The diagnostic probes two orthogonal axes:

1. **Scalar-storage axis** — the fraction of canonical mixing-
   weight component values stored at
   ``G.graph["DNFR_WEIGHTS"]``, ``G.graph["SI_WEIGHTS"]``, and
   ``G.graph["SELECTOR_WEIGHTS"]`` (or their canonical
   defaults) that are structurally scalar-coercible (Python
   ``int``/``float``, NumPy scalar, zero-dim NumPy array).
   Under the canonical implementation (e.g.
   ``src/tnfr/dynamics/dnfr.py:2763``,
   ``src/tnfr/metrics/sense_index.py:425-448``,
   ``src/tnfr/backends/torch_backend.py:172-176``), every read
   is coerced via ``float(weights.get(component, default))``;
   the non-scalar fraction is therefore structurally ``0`` —
   exactly mirroring ``S_dphi storage = 1.0`` (B5a),
   ``noninteger_frac = 0`` (B4a; inverted), ``T_frac = 0``
   (B3a), ``bepi_frac = 0`` (B1a), and ``w_frac = 0`` (B2a).
2. **Node-permutation-invariance axis** — for a deterministic
   set of node relabelings :math:`\\{\\pi_{k}\\}_{k=1}^{K}` of
   the canonical probe graph, compute the canonical scalar
   weighted sum
   :math:`\\Sigma_{c}(i) = \\sum_{c} w_{c} \\cdot g_{c}(i)`
   (where :math:`g_{c}(i)` is a deterministic per-node
   component sample) on each relabeled graph; compare to the
   baseline (identity relabeling). The relabeling acts on node
   labels only, not on weight values. A non-zero divergence
   fraction would *force* the canonical weights to be node-
   indexed (i.e. functionally enriched beyond a single global
   scalar per component); the canonical scalar broadcast
   structurally yields ``0`` by construction because every node
   sees the *same* scalar weight per component.

A high :math:`\\mathcal{S}_{W}` is a *necessary-condition*
check: it says only that the canonical weighted-sum verdict
varies across node relabelings, so a node-indexed or edge-
indexed weight type *might* be required to recover the
canonical evolution. It does **not** prove that the canonical
type of TNFR coupling weights is a non-trivial tensor or
functional.

A low :math:`\\mathcal{S}_{W}` plus a unit scalar-storage
fraction is the empirically expected outcome — structurally
consistent with the catalog typing of weights as global scalar
dicts, with consumer sites reading single floats per component,
and with the absence of any per-node or per-edge weight slot in
the canonical graph schema.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13quadraginta-nona
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §4 row B6
- ``src/tnfr/config/defaults_core.py:57`` (``DNFR_WEIGHTS`` default)
- ``src/tnfr/config/defaults_core.py:65`` (``SI_WEIGHTS`` default)
- ``src/tnfr/config/defaults_core.py:150`` (``SELECTOR_WEIGHTS`` default)
- ``src/tnfr/dynamics/dnfr.py:2762-2764`` (canonical scalar read)
- ``src/tnfr/metrics/sense_index.py:425-448`` (canonical scalar read)
- ``src/tnfr/backends/torch_backend.py:172-176`` (canonical scalar read)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = [
    "CouplingWeightsTypeSignatureCertificate",
    "compute_coupling_weights_type_signature",
]


_CANONICAL_WEIGHT_SLOTS: tuple[str, ...] = (
    "DNFR_WEIGHTS",
    "SI_WEIGHTS",
    "SELECTOR_WEIGHTS",
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


def _canonical_weight_dict(slot: str) -> Mapping[str, Any]:
    """Return the canonical default for ``slot`` from the consolidated DEFAULTS."""
    from ..config.defaults import DEFAULTS

    raw = DEFAULTS.get(slot, {})
    if isinstance(raw, Mapping):
        return raw
    return {}


def _inspect_weight_storage(G: Any) -> tuple[int, int, dict[str, int]]:
    """Inspect raw canonical weight payloads stored on ``G.graph``.

    For each canonical slot in ``_CANONICAL_WEIGHT_SLOTS``, inspect
    every component value (using the graph override if present, else
    the canonical default from ``DEFAULTS``).  Count component values
    that are structurally scalar-coercible.

    Returns
    -------
    n_scalar : int
        Total count of scalar-coercible component values across all
        three canonical slots.
    n_total : int
        Total number of component values inspected across all three
        canonical slots.
    per_slot_nonscalar : dict[str, int]
        Number of non-scalar component values per slot.
    """
    n_scalar = 0
    n_total = 0
    per_slot_nonscalar: dict[str, int] = {}
    for slot in _CANONICAL_WEIGHT_SLOTS:
        raw_override = G.graph.get(slot)
        if isinstance(raw_override, Mapping):
            payload = raw_override
        else:
            payload = _canonical_weight_dict(slot)
        slot_nonscalar = 0
        for _component, value in payload.items():
            n_total += 1
            if _is_scalar_payload(value):
                n_scalar += 1
            else:
                slot_nonscalar += 1
        per_slot_nonscalar[slot] = slot_nonscalar
    return n_scalar, n_total, per_slot_nonscalar


def _build_canonical_demo_graph(n_nodes: int, seed: int) -> Any:
    """Build a small canonical ring graph for the T-W probe."""
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
    return G


def _canonical_weighted_sum(
    weights: Mapping[str, float], components: Mapping[str, float]
) -> float:
    """Canonical scalar weighted sum :math:`\\sum_{c} w_{c} g_{c}`.

    Mirrors the canonical mixing pattern at
    ``src/tnfr/dynamics/dnfr.py:2762-2764`` and analogues:
    ``float(weights.get(c, default)) * float(g_c)``.
    """
    total = 0.0
    for component, w in weights.items():
        g = float(components.get(component, 0.0))
        total += float(w) * g
    return total


def _node_permutation_bracket(
    *,
    G: Any,
    n_permutations: int,
    seed: int,
) -> tuple[int, int, np.ndarray]:
    """Probe canonical weighted-sum verdict under node relabelings.

    For each of ``n_permutations`` deterministic node relabelings
    (including identity at index 0), recompute the canonical
    scalar weighted sum
    :math:`\\Sigma_{c}(i) = \\sum_{c} w_{c} \\cdot g_{c}(i)` where
    :math:`g_{c}(i)` is a deterministic per-node component sample
    (derived from the canonical per-node attributes via a fixed
    map) and :math:`w_{c}` are the canonical scalar weights of
    ``DNFR_WEIGHTS``.

    Under canonical scalar broadcasting, the multiset of per-node
    sums is invariant under node relabeling (since every node sees
    the same scalar weight per component); the sorted vector of
    per-node sums on the relabeled graph equals the sorted vector
    on the identity graph to floating-point precision.

    Returns
    -------
    n_divergent : int
        Number of relabelings whose sorted per-node sum vector
        diverges from the identity baseline beyond ``1e-9``.
        Structurally ``0`` under canonical scalar broadcasting.
    n_total : int
        Total number of relabelings probed (== ``n_permutations``).
    diff_matrix : np.ndarray of shape (n_permutations,)
        Per-relabeling :math:`\\ell^{\\infty}` norm of the sorted
        sum-vector difference from baseline.
    """
    from ..config.defaults import DEFAULTS

    weights = dict(DEFAULTS.get("DNFR_WEIGHTS", {}))
    components = ("phase", "epi", "vf", "topo")
    # Deterministic per-node component samples derived from node
    # attributes; canonical scalar weights are then applied.
    node_list = list(G.nodes())
    rng = np.random.default_rng(int(seed))
    g_per_node: dict[Any, dict[str, float]] = {}
    for node in node_list:
        theta = float(G.nodes[node].get("theta", 0.0))
        epi = float(G.nodes[node].get("EPI", 0.0))
        nuf = float(G.nodes[node].get("nu_f", 0.0))
        g_per_node[node] = {
            "phase": math.cos(theta),
            "epi": epi,
            "vf": nuf,
            "topo": float(G.degree(node)),
        }

    def _sums_under_relabel(perm: np.ndarray) -> np.ndarray:
        # perm[i] = new index of node_list[i].  Compute sums on
        # relabeled graph: per-node sum at *new* index = sum at
        # original node with that new label.  Under scalar
        # broadcasting the multiset of sums is invariant; we
        # compute the sorted vector explicitly for robustness.
        sums = np.zeros(len(node_list), dtype=float)
        for i, node in enumerate(node_list):
            sums[int(perm[i])] = _canonical_weighted_sum(
                weights, g_per_node[node]
            )
        return np.sort(sums)

    identity = np.arange(len(node_list), dtype=int)
    baseline_sorted = _sums_under_relabel(identity)
    n_divergent = 0
    n_total = int(n_permutations)
    diffs = np.zeros(n_total, dtype=float)
    for k in range(n_total):
        if k == 0:
            perm = identity
        else:
            perm = rng.permutation(len(node_list))
        sorted_k = _sums_under_relabel(perm)
        diff = float(np.max(np.abs(sorted_k - baseline_sorted)))
        diffs[k] = diff
        if diff > 1e-9:
            n_divergent += 1
    return n_divergent, n_total, diffs


def _signature(n_divergent: int, n_total: int) -> tuple[float, float]:
    """Return ``(squashed_signature, raw_fraction)``."""
    if n_total <= 0:
        return 0.0, 0.0
    raw = float(n_divergent) / float(n_total)
    return float(math.tanh(raw)), raw


@dataclass(frozen=True)
class CouplingWeightsTypeSignatureCertificate:
    """Result of the Coupling-Weights-Type Signature diagnostic.

    Attributes
    ----------
    signature : float
        :math:`\\mathcal{S}_{W} \\in [0, 1]`.  ``0`` means node-
        relabeling-invariant (canonical scalar broadcasting is
        structurally sufficient); ``1`` means verdicts diverge
        maximally across relabelings.
    scalar_storage_fraction : float
        Fraction of canonical-slot component values that are
        structurally scalar.  ``1.0`` is the empirically expected
        value under canonical defaults.
    nonscalar_storage_count : int
        Absolute number of non-scalar component values observed.
    n_storage_reads : int
        Total number of component values inspected (sum of dict
        sizes across the three canonical weight slots).
    per_slot_nonscalar : dict[str, int]
        Number of non-scalar component values per canonical slot.
    n_permutations : int
        Number of deterministic node relabelings probed.
    n_divergent_permutations : int
        Number of relabelings whose canonical weighted-sum verdict
        diverges from the identity baseline beyond ``1e-9``.
    raw_divergence_fraction : float
        Pre-squash ``n_divergent_permutations / n_permutations``.
    n_nodes : int
        Number of nodes in the canonical probe graph.
    verdict : str
        One of ``"SCALAR_WEIGHTS_ADEQUATE"`` (signature <
        ``perm_threshold`` AND scalar storage fraction == 1.0),
        ``"NODE_INDEXED_WEIGHTS_NECESSARY"`` (signature >
        ``divergent_threshold`` OR scalar storage fraction < 1.0),
        or ``"INDETERMINATE"``.
    diagnostics : dict
        Auxiliary fields (per-permutation diff vector, thresholds,
        seed, raw counters).
    """

    signature: float
    scalar_storage_fraction: float
    nonscalar_storage_count: int
    n_storage_reads: int
    per_slot_nonscalar: dict[str, int]
    n_permutations: int
    n_divergent_permutations: int
    raw_divergence_fraction: float
    n_nodes: int
    verdict: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        per_slot = ", ".join(
            f"{slot}={count}"
            for slot, count in self.per_slot_nonscalar.items()
        )
        lines = [
            "Coupling-Weights-Type Signature certificate (diagnostic only - Sec 13quadraginta-nona.5)",
            f"  signature S_W            : {self.signature:.6f}   (0 = relabel-invariant, 1 = relabel-divergent)",
            f"  scalar storage fraction  : {self.scalar_storage_fraction:.4f}"
            f"  ({self.nonscalar_storage_count} non-scalar / "
            f"{self.n_storage_reads} total component values)",
            f"  per-slot non-scalar      : {per_slot}",
            f"  raw divergence fraction  : {self.raw_divergence_fraction:.6e}"
            f"  ({self.n_divergent_permutations} / {self.n_permutations} relabelings)",
            f"  probe graph              : {self.n_nodes} nodes",
            f"  verdict                  : {self.verdict}",
            "  scope: necessary-condition diagnostic; does NOT advance G4 = RH",
        ]
        return "\n".join(lines)


def compute_coupling_weights_type_signature(
    *,
    n_nodes: int = 24,
    n_permutations: int = 12,
    seed: int = 23,
    perm_threshold: float = 0.05,
    divergent_threshold: float = 0.25,
) -> CouplingWeightsTypeSignatureCertificate:
    """Compute the Coupling-Weights-Type Signature on canonical mixing reads.

    Parameters
    ----------
    n_nodes : int, default 24
        Size of the canonical ring probe graph used for both axes.
    n_permutations : int, default 12
        Number of deterministic node relabelings probed (including
        identity at index 0).
    seed : int, default 23
        Deterministic seed for the probe graph and relabelings.
        Distinct from the B0-B5 seeds.
    perm_threshold : float, default 0.05
        Below this signature AND with scalar storage fraction equal
        to ``1.0``, the verdict is ``"SCALAR_WEIGHTS_ADEQUATE"``.
    divergent_threshold : float, default 0.25
        Above this signature OR with scalar storage fraction below
        ``1.0``, the verdict is ``"NODE_INDEXED_WEIGHTS_NECESSARY"``.

    Returns
    -------
    CouplingWeightsTypeSignatureCertificate
        Diagnostic certificate.

    Empirical baseline
    ------------------
    Under canonical defaults (``DNFR_WEIGHTS``, ``SI_WEIGHTS``,
    ``SELECTOR_WEIGHTS`` from ``defaults_core.py``), the expected
    outcome is ``scalar_storage_fraction == 1.0`` (structural) and
    ``signature == 0.0`` (structural), yielding verdict
    ``"SCALAR_WEIGHTS_ADEQUATE"``.
    """
    G = _build_canonical_demo_graph(int(n_nodes), int(seed))
    n_scalar, n_total_storage, per_slot_nonscalar = _inspect_weight_storage(
        G
    )
    scalar_storage_fraction = (
        float(n_scalar) / float(n_total_storage)
        if n_total_storage > 0
        else 0.0
    )
    nonscalar_storage_count = int(n_total_storage - n_scalar)

    n_divergent, n_total_perms, diffs = _node_permutation_bracket(
        G=G,
        n_permutations=int(n_permutations),
        seed=int(seed),
    )
    signature, raw_div = _signature(n_divergent, n_total_perms)

    if (
        signature < perm_threshold
        and scalar_storage_fraction >= 1.0 - 1e-12
    ):
        verdict = "SCALAR_WEIGHTS_ADEQUATE"
    elif (
        signature > divergent_threshold
        or scalar_storage_fraction < 1.0 - 1e-12
    ):
        verdict = "NODE_INDEXED_WEIGHTS_NECESSARY"
    else:
        verdict = "INDETERMINATE"

    diagnostics: dict[str, Any] = {
        "permutation_diffs": diffs.tolist(),
        "perm_threshold": float(perm_threshold),
        "divergent_threshold": float(divergent_threshold),
        "seed": int(seed),
        "scalar_storage_count": int(n_scalar),
        "total_storage_reads": int(n_total_storage),
        "canonical_slots": list(_CANONICAL_WEIGHT_SLOTS),
    }

    return CouplingWeightsTypeSignatureCertificate(
        signature=signature,
        scalar_storage_fraction=scalar_storage_fraction,
        nonscalar_storage_count=nonscalar_storage_count,
        n_storage_reads=int(n_total_storage),
        per_slot_nonscalar=dict(per_slot_nonscalar),
        n_permutations=int(n_total_perms),
        n_divergent_permutations=int(n_divergent),
        raw_divergence_fraction=raw_div,
        n_nodes=int(n_nodes),
        verdict=verdict,
        diagnostics=diagnostics,
    )
