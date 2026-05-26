"""ΔNFR-Type Signature — Diagnostic for the T-ΔNFR Conjecture (§13quadraginta).

This module implements a purely diagnostic quantity, the **ΔNFR-Type
Signature** :math:`\\mathcal{S}_{\\Delta\\mathrm{NFR}}`, that
quantifies on canonical TNFR network evolutions whether the canonical
nodal gradient :math:`\\Delta\\mathrm{NFR} \\in \\mathbb{R}` admits an
*irreducible* tensor-rank lift (e.g. a per-node vector or rank-2
tensor over the constituent gradient channels ``phase``, ``EPI``,
``νf``), or whether the single scalar slot written by
:func:`tnfr.dynamics.dnfr.default_compute_delta_nfr` is structurally
sufficient.

Methodological scope (mandatory honesty)
----------------------------------------
This module is a *diagnostic only*.  It does **not** construct,
promote, or modify any canonical operator.  It does **not** advance
G4 = RH.  It does **not** by itself decide the T-ΔNFR Conjecture
(which requires the forcing-axiom reduction of §13quadraginta-prima
and the final verdict of §13quadraginta-secunda — both deferred to
B3b/B3c).

The diagnostic probes two orthogonal axes:

1. **Tensor storage axis** — the fraction of ``(node, step)`` samples
   at which the canonical per-node ΔNFR slot stores a non-scalar
   value (vector, tensor, mapping, or otherwise non-``float`` payload).
   Under the canonical implementation
   :func:`tnfr.dynamics.dnfr.default_compute_delta_nfr`, the slot is
   always a scalar by construction, so this fraction is structurally
   ``0`` — exactly mirroring ``w_frac = 0`` of the B2a φ-diagnostic
   and ``bepi_frac = 0`` of the B1a EPI-diagnostic.
2. **Rank-entropy axis** — Shannon entropy of the normalised
   singular-value distribution of the per-node gradient-component
   matrix :math:`M_i \\in \\mathbb{R}^{T \\times 3}` whose columns are
   the per-step mean-neighbour differences in the three canonical
   gradient channels :math:`(d\\theta, d\\mathrm{EPI}, d\\nu_f)`,
   averaged across nodes and normalised by :math:`\\log 3` (the
   maximum rank).  A multi-rank gradient stream is a *necessary*
   condition for the canonical dynamics to require a tensor-valued
   ΔNFR (a rank-1 scalar projection is fully equivalent to the
   tensor stream if the three channels collapse to a single rank).

A high :math:`\\mathcal{S}_{\\Delta\\mathrm{NFR}}` is a
*necessary-condition* check: it says only that canonical gradient
accumulation on a TNFR graph carries irreducible multi-channel
content that a rank-1 scalar reading necessarily compresses.  It does
**not** prove that the canonical type of ΔNFR is a non-trivial
tensor element.

A low :math:`\\mathcal{S}_{\\Delta\\mathrm{NFR}}` plus a zero tensor
storage fraction is the empirically expected outcome — structurally
consistent with the catalog row 1 typing
:math:`\\Delta\\mathrm{NFR} \\in \\mathbb{R}` and with the scalar
nodal-equation contract
:math:`(\\nu_f, \\Delta\\mathrm{NFR}) \\mapsto
\\partial\\mathrm{EPI}/\\partial t` enforced at
``src/tnfr/operators/nodal_equation.py:1-160``.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13quadraginta
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §4 row B3
- ``src/tnfr/dynamics/dnfr.py:2387::default_compute_delta_nfr``
  (canonical scalar-writing implementation)
- ``src/tnfr/constants/aliases.py:9::ALIAS_DNFR`` (canonical
  per-node storage alias)
- ``src/tnfr/operators/nodal_equation.py:1-160`` (scalar contract)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DnfrTypeSignatureCertificate",
    "compute_dnfr_type_signature",
]


def _shannon_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats of a probability vector.

    Zero-probability entries are skipped (``0 · log 0 := 0``).
    """
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _wrap_to_pi(angle: float) -> float:
    """Wrap ``angle`` to the canonical fundamental domain ``[-π, π]``.

    Mirrors :func:`tnfr.physics._helpers.wrap_angle` without importing
    it (the diagnostic must be runnable in isolation if needed).
    """
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _build_canonical_demo_graph(n_nodes: int, seed: int) -> Any:
    """Build a small canonical ring graph for the ΔNFR trajectory probe.

    Uses :func:`tnfr.sdk.TNFR.create` to obtain a TNFR network with
    canonical defaults and a fixed ring topology so the diagnostic is
    deterministic given the seed.  The initial phase / EPI / νf
    distributions are deterministic mild perturbations so canonical
    evolution starts away from a trivial symmetric fixed point.
    """
    from ..sdk import TNFR

    net = TNFR.create(int(n_nodes)).ring()
    G = net.G
    rng = np.random.default_rng(int(seed))
    for node in list(G.nodes()):
        G.nodes[node]["EPI"] = float(0.5 + 0.05 * (rng.random() - 0.5))
        G.nodes[node]["theta"] = _wrap_to_pi(
            float(2.0 * math.pi * (rng.random() - 0.5))
        )
        # νf must remain strictly positive; perturb mildly around the default.
        current_vf = float(G.nodes[node].get("nu_f", 1.0))
        G.nodes[node]["nu_f"] = max(0.05, current_vf + 0.05 * (rng.random() - 0.5))
    return G


def _read_node_scalars(G: Any, node: Any) -> tuple[float, float, float, Any]:
    """Read the canonical scalar triple ``(θ, EPI, νf)`` plus the raw ΔNFR slot.

    Reads use canonical accessors only.  The raw ΔNFR slot is returned
    unconverted so the tensor-storage axis can detect any non-scalar
    payload.
    """
    from ..physics._helpers import get_phase

    attrs = G.nodes[node]
    theta = _wrap_to_pi(get_phase(G, node))
    epi = float(attrs.get("EPI", 0.0))
    vf = float(attrs.get("nu_f", attrs.get("νf", 1.0)))
    raw_dnfr = attrs.get("dnfr", attrs.get("ΔNFR", 0.0))
    return theta, epi, vf, raw_dnfr


def _is_scalar_payload(value: Any) -> bool:
    """Return ``True`` iff ``value`` is a single real scalar.

    A scalar is: Python ``int``, Python ``float``, NumPy 0-d array, or
    NumPy scalar.  Anything iterable / sequence / mapping / object with
    ``len(...)`` > 1 is *non-scalar* and is counted toward the tensor
    storage fraction.
    """
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, np.generic):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return False


def _collect_neighbour_gradient_triples(
    G: Any, nodes: list[Any]
) -> np.ndarray:
    """For each node, compute the mean-neighbour gradient triple.

    Returns a ``(n_nodes, 3)`` array whose row ``i`` is
    :math:`(\\overline{\\Delta\\theta_i},
    \\overline{\\Delta\\mathrm{EPI}_i},
    \\overline{\\Delta\\nu_{f,i}})` averaged over the neighbours of
    node ``i``.  Isolated nodes contribute a zero row.
    """
    n = len(nodes)
    triples = np.zeros((n, 3), dtype=float)
    if n == 0:
        return triples
    # Cache scalars for each node once per step.
    scalars: dict[Any, tuple[float, float, float]] = {}
    for node in nodes:
        theta, epi, vf, _ = _read_node_scalars(G, node)
        scalars[node] = (theta, epi, vf)
    for i, node in enumerate(nodes):
        neighbours = list(G.neighbors(node))
        if not neighbours:
            continue
        theta_i, epi_i, vf_i = scalars[node]
        d_theta = 0.0
        d_epi = 0.0
        d_vf = 0.0
        for nb in neighbours:
            if nb not in scalars:
                theta_nb, epi_nb, vf_nb, _ = _read_node_scalars(G, nb)
                scalars[nb] = (theta_nb, epi_nb, vf_nb)
            theta_nb, epi_nb, vf_nb = scalars[nb]
            d_theta += _wrap_to_pi(theta_i - theta_nb)
            d_epi += epi_i - epi_nb
            d_vf += vf_i - vf_nb
        k = float(len(neighbours))
        triples[i, 0] = d_theta / k
        triples[i, 1] = d_epi / k
        triples[i, 2] = d_vf / k
    return triples


def _per_node_rank_entropy(
    component_history: np.ndarray, *, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node Shannon entropy of normalised singular values.

    ``component_history`` has shape ``(n_nodes, n_steps, 3)``.  For each
    node, stack the per-step rows into a ``(n_steps, 3)`` matrix and
    compute its SVD; normalise the singular values to a probability
    vector and return the Shannon entropy (nats).  Also returns the
    raw singular-value matrix ``(n_nodes, 3)``.
    """
    n_nodes = component_history.shape[0]
    entropies = np.zeros(n_nodes, dtype=float)
    sv_matrix = np.zeros((n_nodes, 3), dtype=float)
    for i in range(n_nodes):
        M = component_history[i]  # shape (n_steps, 3)
        if M.shape[0] < 1:
            continue
        if not np.any(np.abs(M) > eps):
            sv_matrix[i] = np.zeros(3)
            continue
        # SVD on a (T, 3) matrix returns at most 3 singular values.
        try:
            sv = np.linalg.svd(M, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        if sv.size < 3:
            sv = np.pad(sv, (0, 3 - sv.size), constant_values=0.0)
        sv_matrix[i] = sv
        total = float(np.sum(sv))
        if total <= eps:
            continue
        p = sv / total
        entropies[i] = _shannon_entropy(p)
    return entropies, sv_matrix


def _evolve_and_collect(
    G: Any, nodes: list[Any], n_steps: int
) -> tuple[np.ndarray, int, int]:
    """Run ``n_steps`` canonical evolution steps and collect gradient triples.

    Returns
    -------
    component_history : np.ndarray of shape ``(n_nodes, n_steps, 3)``
        Per-node, per-step ``(d_theta, d_epi, d_vf)`` triples.
    n_scalar_samples : int
        Number of ``(node, step)`` samples in which the canonical ΔNFR
        slot stored a scalar payload.
    n_total_samples : int
        Total number of ``(node, step)`` samples inspected.
    """
    from ..constants import inject_defaults
    from ..dynamics import step

    inject_defaults(G)
    n = len(nodes)
    history = np.zeros((n, int(n_steps), 3), dtype=float)
    n_scalar_samples = 0
    n_total_samples = 0
    for t in range(int(n_steps)):
        step(G)
        # Inspect canonical ΔNFR slot for non-scalar payloads.
        for node in nodes:
            n_total_samples += 1
            raw = G.nodes[node].get("dnfr", G.nodes[node].get("ΔNFR", 0.0))
            if _is_scalar_payload(raw):
                n_scalar_samples += 1
        # Re-compute the per-node mean-neighbour gradient triple.
        history[:, t, :] = _collect_neighbour_gradient_triples(G, nodes)
    return history, n_scalar_samples, n_total_samples


@dataclass(frozen=True)
class DnfrTypeSignatureCertificate:
    """Result of the ΔNFR-Type Signature diagnostic on a canonical network.

    Attributes
    ----------
    signature : float
        :math:`\\mathcal{S}_{\\Delta\\mathrm{NFR}} \\in [0, 1]`.  ``0``
        means scalar-adequate (rank-1 collapse of the three gradient
        channels); ``1`` means maximum-rank-entropy (uniform mass on
        all three singular values).
    tensor_storage_fraction : float
        Fraction of ``(node, step)`` samples in which the canonical
        ΔNFR slot stored a *non-scalar* payload.  ``0.0`` is the
        empirically expected value under the canonical scalar contract
        of :func:`tnfr.dynamics.dnfr.default_compute_delta_nfr`; any
        non-zero value flags that canonical evolution writes a
        non-scalar that the scalar reader necessarily compresses.
    tensor_storage_count : int
        Absolute number of ``(node, step)`` samples whose ΔNFR slot
        held a non-scalar payload.
    n_nodes : int
        Number of nodes in the diagnostic graph.
    n_steps : int
        Number of evolution steps taken (gradient-triple history has
        ``n_steps`` rows per node).
    mean_rank_entropy_nats : float
        Mean Shannon entropy (across nodes) of the normalised
        singular-value distribution of the gradient-triple matrix,
        in nats.
    effective_rank : float
        :math:`R_{\\mathrm{eff}} = \\exp(H_{\\mathrm{rank}})` —
        effective tensor rank in :math:`[1, 3]`.  Scalar-adequate iff
        :math:`R_{\\mathrm{eff}} \\approx 1`.
    mean_singular_values : tuple[float, float, float]
        Mean across nodes of the three singular values of
        :math:`M_i` (largest first).
    verdict : str
        One of ``"SCALAR_DNFR_ADEQUATE"`` (signature <
        ``scalar_threshold`` AND zero tensor storage fraction),
        ``"TENSOR_LIFT_NECESSARY"`` (signature > ``tensor_threshold``
        OR non-zero tensor storage fraction), or ``"INDETERMINATE"``.
    diagnostics : dict
        Auxiliary fields (per-node entropies, per-node singular values,
        thresholds, seed, scalar/total sample counts, etc.).
    """

    signature: float
    tensor_storage_fraction: float
    tensor_storage_count: int
    n_nodes: int
    n_steps: int
    mean_rank_entropy_nats: float
    effective_rank: float
    mean_singular_values: tuple[float, float, float]
    verdict: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        sv = self.mean_singular_values
        lines = [
            "DeltaNFR-Type Signature certificate (diagnostic only — §13quadraginta.5)",
            f"  signature S_DNFR         : {self.signature:.6f}   (0 = scalar rank-1, 1 = uniform rank-3)",
            f"  tensor storage fraction  : {self.tensor_storage_fraction:.4f}"
            f"  ({self.tensor_storage_count}/{self.n_nodes * self.n_steps} samples)",
            f"  mean rank entropy        : {self.mean_rank_entropy_nats:.4f} nats"
            f" (max log 3 = {math.log(3.0):.4f})",
            f"  effective rank R_eff     : {self.effective_rank:.4f}   (in [1, 3])",
            f"  mean singular values     : (sigma1={sv[0]:.4f},"
            f" sigma2={sv[1]:.4f}, sigma3={sv[2]:.4f})",
            f"  graph: {self.n_nodes} nodes, {self.n_steps} evolution steps",
            f"  verdict                  : {self.verdict}",
            "  scope: necessary-condition diagnostic; does NOT advance G4 = RH",
        ]
        return "\n".join(lines)


def compute_dnfr_type_signature(
    *,
    n_nodes: int = 24,
    n_steps: int = 64,
    seed: int = 17,
    scalar_threshold: float = 0.15,
    tensor_threshold: float = 0.5,
) -> DnfrTypeSignatureCertificate:
    """Compute the ΔNFR-Type Signature on a canonical TNFR ring evolution.

    Parameters
    ----------
    n_nodes : int, default 24
        Size of the ring graph used as the canonical probe.
    n_steps : int, default 64
        Number of evolution steps after the initial state.  Per-node
        gradient-triple history has ``n_steps`` rows.
    seed : int, default 17
        Deterministic seed for the initial phase / EPI / νf
        perturbation.
    scalar_threshold : float, default 0.15
        Below this signature value AND with zero tensor storage
        fraction, the verdict is ``"SCALAR_DNFR_ADEQUATE"``.
    tensor_threshold : float, default 0.5
        Above this signature value OR with non-zero tensor storage
        fraction, the verdict is ``"TENSOR_LIFT_NECESSARY"``.

    Returns
    -------
    DnfrTypeSignatureCertificate
        Diagnostic certificate.

    Notes
    -----
    The diagnostic uses two orthogonal axes:

    - **Tensor storage axis**: per ``(node, step)`` sample, inspect
      the canonical ΔNFR slot
      (``G.nodes[node]["dnfr"]`` / ``ALIAS_DNFR``) for non-scalar
      payloads.  Under the canonical implementation
      :func:`tnfr.dynamics.dnfr.default_compute_delta_nfr`, every
      slot is a Python ``float``; the storage fraction is therefore
      structurally ``0`` by construction — exactly mirroring the
      :math:`w_{\\mathrm{frac}} = 0` outcome of the B2a φ-diagnostic
      and the :math:`\\mathrm{bepi\\_frac} = 0` outcome of the B1a
      EPI-diagnostic.
    - **Rank-entropy axis**: per node, build the gradient-triple
      matrix :math:`M_i \\in \\mathbb{R}^{n_{\\mathrm{steps}} \\times 3}`
      from the per-step mean-neighbour differences in the three
      canonical gradient channels :math:`(d\\theta, d\\mathrm{EPI},
      d\\nu_f)`.  Compute the SVD, normalise the three singular
      values to a probability vector, and report the Shannon entropy
      averaged across nodes, normalised by :math:`\\log 3`.

    This is a *purely diagnostic* computation on canonical TNFR data.
    It does not construct any new operator and does not modify the
    13-operator catalog.
    """
    if int(n_nodes) < 3:
        raise ValueError("n_nodes must be >= 3 for a meaningful ring graph")
    if int(n_steps) < 4:
        raise ValueError("n_steps must be >= 4 for a meaningful trajectory")

    G = _build_canonical_demo_graph(int(n_nodes), int(seed))
    nodes = list(G.nodes())
    component_history, n_scalar_samples, n_total_samples = _evolve_and_collect(
        G, nodes, int(n_steps)
    )
    actual_n_nodes = len(nodes)
    actual_n_steps = int(component_history.shape[1]) if actual_n_nodes > 0 else 0

    # Tensor storage axis (always 0 under canonical scalar contract).
    if n_total_samples > 0:
        n_tensor_samples = n_total_samples - n_scalar_samples
        tensor_fraction = float(n_tensor_samples) / float(n_total_samples)
    else:
        n_tensor_samples = 0
        tensor_fraction = 0.0

    # Rank-entropy axis.
    per_node_entropy, per_node_sv = _per_node_rank_entropy(component_history)
    mean_entropy = float(np.mean(per_node_entropy)) if actual_n_nodes > 0 else 0.0
    mean_sv = (
        float(np.mean(per_node_sv[:, 0])),
        float(np.mean(per_node_sv[:, 1])),
        float(np.mean(per_node_sv[:, 2])),
    ) if actual_n_nodes > 0 else (0.0, 0.0, 0.0)

    max_entropy = math.log(3.0)
    signature = mean_entropy / max_entropy if max_entropy > 0.0 else 0.0
    signature = float(min(max(signature, 0.0), 1.0))
    effective_rank = float(math.exp(mean_entropy)) if mean_entropy > 0.0 else 1.0

    if signature < float(scalar_threshold) and tensor_fraction == 0.0:
        verdict = "SCALAR_DNFR_ADEQUATE"
    elif signature > float(tensor_threshold) or tensor_fraction > 0.0:
        verdict = "TENSOR_LIFT_NECESSARY"
    else:
        verdict = "INDETERMINATE"

    diagnostics: dict[str, Any] = {
        "per_node_rank_entropy_nats": per_node_entropy.tolist(),
        "per_node_singular_values": per_node_sv.tolist(),
        "scalar_threshold": float(scalar_threshold),
        "tensor_threshold": float(tensor_threshold),
        "max_entropy_nats": float(max_entropy),
        "n_scalar_samples": int(n_scalar_samples),
        "n_tensor_samples": int(n_tensor_samples),
        "n_total_samples": int(n_total_samples),
        "seed": int(seed),
    }

    return DnfrTypeSignatureCertificate(
        signature=signature,
        tensor_storage_fraction=float(tensor_fraction),
        tensor_storage_count=int(n_tensor_samples),
        n_nodes=int(actual_n_nodes),
        n_steps=int(actual_n_steps),
        mean_rank_entropy_nats=float(mean_entropy),
        effective_rank=float(effective_rank),
        mean_singular_values=mean_sv,
        verdict=verdict,
        diagnostics=diagnostics,
    )
