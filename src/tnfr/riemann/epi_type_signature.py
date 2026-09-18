"""Historical EPI-Type Signature: storage and temporal-spectrum observations.

The storage axis checks the shared exact uniform-real EPI chart. Uniform
``ensure_bepi(scalar)`` embeddings remain scalar; nonuniform or complex
elements lie outside that chart. This is a statement about inspected storage,
not the minimum dimension needed by a model or a physical system.

The spectral axis describes the binned magnitude spectrum of a scalar temporal
read-out. An ordinary scalar signal can contain arbitrarily many frequencies;
its entropy is not evidence of vector-valued EPI or a type-necessity theorem.
For richer storage the established maximum-component read-out is lossy.

Public callable and result-field names are retained for compatibility. The
historical ``SCALAR_ADEQUATE`` and ``BEPI_VALUED_NECESSARY`` verdicts are retired;
the thresholds no longer classify storage. Historical results in the research
notes are not recomputed by this correction. No RH claim or new operator follows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..types import ensure_bepi, real_scalar_epi

__all__ = [
    "EpiTypeSignatureCertificate",
    "compute_epi_type_signature",
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


def _binned_psd_distribution(trajectory: np.ndarray, n_bins: int) -> np.ndarray:
    """Normalised binned magnitude-spectrum distribution of a 1-D trajectory.

    Uses the real-FFT magnitude after mean removal and bins the
    resulting magnitude distribution onto ``n_bins`` uniform bins.
    Trajectories with zero variance return a degenerate distribution
    (all mass in the first bin).
    """
    x = np.asarray(trajectory, dtype=float).ravel()
    if x.size < 2:
        p = np.zeros(n_bins, dtype=float)
        p[0] = 1.0
        return p
    # Remove DC mean before FFT to focus on AC modal content.
    x_centered = x - float(np.mean(x))
    if not np.any(np.abs(x_centered) > 0.0):
        p = np.zeros(n_bins, dtype=float)
        p[0] = 1.0
        return p
    spectrum = np.abs(np.fft.rfft(x_centered))
    total = float(np.sum(spectrum))
    if total <= 0.0:
        p = np.zeros(n_bins, dtype=float)
        p[0] = 1.0
        return p
    # Histogram of spectral magnitude across uniform frequency bins.
    freqs = np.arange(spectrum.size, dtype=float)
    counts, _ = np.histogram(
        freqs,
        bins=n_bins,
        range=(0.0, float(spectrum.size)),
        weights=spectrum,
    )
    total_counts = float(np.sum(counts))
    if total_counts <= 0.0:
        p = np.zeros(n_bins, dtype=float)
        p[0] = 1.0
        return p
    return counts / total_counts


def _bepi_storage_fraction(
    storage_values: Iterable[Any], *, atol: float = 1e-12
) -> tuple[float, int, int]:
    """Fraction outside the shared exact uniform-real scalar chart.

    ``atol`` is retained for call compatibility but does not relax exact chart
    membership. Valid serialized BEPI values use the same shared decoder as
    live elements. Unsupported or nonfinite storage raises rather than being
    silently counted as scalar. This does not test minimal state dimension.
    """
    n_total = 0
    n_nontrivial = 0
    for value in storage_values:
        n_total += 1
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("EPI storage must not be boolean")
        element = ensure_bepi(value)
        if not all(
            np.all(np.isfinite(component))
            for component in (element.f_continuous, element.a_discrete)
        ):
            raise ValueError("EPI storage components must be finite")
        if real_scalar_epi(element) is None:
            n_nontrivial += 1
    if n_total == 0:
        return 0.0, 0, 0
    return float(n_nontrivial) / float(n_total), n_nontrivial, n_total


def _build_canonical_demo_graph(n_nodes: int, seed: int) -> Any:
    """Build a small canonical ring graph for the EPI trajectory probe.

    Uses :func:`tnfr.sdk.TNFR.create` to obtain a TNFR network with
    canonical defaults and a fixed ring topology so the diagnostic is
    deterministic given the seed.
    """
    from ..sdk import TNFR

    net = TNFR.create(int(n_nodes)).ring()
    G = net.G
    rng = np.random.default_rng(int(seed))
    # Mild deterministic EPI perturbation around the canonical mid-point.
    for node in list(G.nodes()):
        G.nodes[node]["EPI"] = float(0.5 + 0.05 * (rng.random() - 0.5))
    return G


def _evolve_and_collect(G: Any, n_steps: int) -> np.ndarray:
    """Run ``n_steps`` canonical evolution steps and collect EPI per node.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(n_nodes, n_steps + 1)`` with the EPI value
        of every node at every collected step (including the initial
        state). ``scalarize_epi`` preserves signed uniform-real embeddings;
        richer BEPI elements use the established lossy magnitude projection.
    """
    from ..alias import get_attr
    from ..constants import inject_defaults
    from ..constants.aliases import ALIAS_EPI
    from ..dynamics import step
    from ..types import scalarize_epi

    # Inject canonical defaults (VF_ADAPT_MU, VF_ADAPT_TAU, etc.) so the
    # canonical step() function has its required graph parameters.
    inject_defaults(G)

    nodes = list(G.nodes())
    snapshots: list[list[float]] = [
        [
            get_attr(G.nodes[n], ALIAS_EPI, 0.0, conv=scalarize_epi, strict=True)
            for n in nodes
        ]
    ]
    # step() falls back to default_compute_delta_nfr if no hook is set.
    for _ in range(int(n_steps)):
        step(G)
        snapshots.append(
            [
                get_attr(G.nodes[n], ALIAS_EPI, 0.0, conv=scalarize_epi, strict=True)
                for n in nodes
            ]
        )
    return np.asarray(snapshots, dtype=float).T


@dataclass(frozen=True)
class EpiTypeSignatureCertificate:
    """Historical result schema for two descriptive observations.

    Attributes
    ----------
    signature : float
        Mean temporal magnitude-spectrum entropy divided by ``log(n_bins)``.
        Zero denotes a concentrated histogram, one a uniform histogram;
        neither determines the dimension or type of the underlying EPI.
    storage_bepi_fraction : float
        Fraction of inspected entries outside the shared uniform-real chart.
        The legacy field name does not count trivial scalar BEPI embeddings.
    storage_bepi_count : int
        Number of inspected entries outside the uniform-real scalar chart.
    storage_total : int
        Total number of inspected storage entries.
    mean_spectral_entropy_nats : float
        Mean Shannon entropy of the binned EPI temporal trajectory
        spectrum across nodes, in nats.
    effective_modes : float
        ``exp(H)``: effective occupied spectral-bin count, not state dimension.
    n_nodes : int
        Number of nodes in the diagnostic graph.
    n_steps : int
        Number of evolution steps taken (trajectory length is
        ``n_steps + 1`` per node).
    n_bins : int
        Number of histogram bins used for the spectral distribution.
    verdict : str
        ``REAL_SCALAR_STORAGE``, ``NONSCALAR_BEPI_STORAGE_OBSERVED`` or
        ``NO_STORAGE_OBSERVED``. Spectral entropy never selects this label.
    diagnostics : dict
        Auxiliary fields (per-node entropies, trajectory variance, etc.).
    """

    signature: float
    storage_bepi_fraction: float
    storage_bepi_count: int
    storage_total: int
    mean_spectral_entropy_nats: float
    effective_modes: float
    n_nodes: int
    n_steps: int
    n_bins: int
    verdict: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "EPI-Type Signature (descriptive storage and temporal spectrum)",
            f"  signature S_EPI         : {self.signature:.6f}   (normalized entropy)",
            f"  non-scalar storage      : {self.storage_bepi_fraction:.4f}"
            f"  ({self.storage_bepi_count}/{self.storage_total} nodes)",
            f"  mean spectral entropy   : {self.mean_spectral_entropy_nats:.4f} nats"
            f" over {self.n_bins} bins",
            f"  effective modes N_eff   : {self.effective_modes:.2f}",
            f"  graph: {self.n_nodes} nodes, {self.n_steps} evolution steps",
            f"  verdict                 : {self.verdict}",
            "  scope: inspected storage and projected temporal spectrum only;",
            "         no type necessity, state-dimension or RH conclusion",
        ]
        return "\n".join(lines)


def compute_epi_type_signature(
    *,
    n_nodes: int = 24,
    n_steps: int = 64,
    n_bins: int = 32,
    seed: int = 13,
    scalar_threshold: float = 0.15,
    bepi_threshold: float = 0.5,
    storage_atol: float = 1e-12,
) -> EpiTypeSignatureCertificate:
    """Compute the EPI-Type Signature on a canonical TNFR ring evolution.

    Parameters
    ----------
    n_nodes : int, default 24
        Size of the ring graph used as the canonical probe.
    n_steps : int, default 64
        Number of evolution steps after initial state collection.
        Trajectory length per node is ``n_steps + 1``.
    n_bins : int, default 32
        Histogram resolution :math:`B` for the spectral distribution.
        The maximum entropy is :math:`\\log B`; the signature is
        normalised by this maximum so that
        :math:`\\mathcal{S}_{\\mathrm{EPI}} \\in [0, 1]`.
    seed : int, default 13
        Deterministic seed for the initial EPI perturbation.
    scalar_threshold, bepi_threshold, storage_atol : float
        Deprecated compatibility arguments, retained in diagnostics only.
        They neither infer dimensional necessity nor relax exact scalar-chart
        membership. No new spectral threshold replaces them.

    Returns
    -------
    EpiTypeSignatureCertificate
        Diagnostic certificate.

    Notes
    -----
    A scalar temporal signal may have a broad spectrum. Final storage and
    temporal spectral entropy are separate observations: neither establishes
    a minimal realization or validity of an unobserved dynamical model. The
    current demo still executes its declared native steps; it adds no operator.
    """
    if int(n_nodes) < 3:
        raise ValueError("n_nodes must be >= 3 for a meaningful ring graph")
    if int(n_steps) < 4:
        raise ValueError("n_steps must be >= 4 for a meaningful trajectory")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be >= 2")

    G = _build_canonical_demo_graph(int(n_nodes), int(seed))
    trajectories = _evolve_and_collect(G, int(n_steps))
    if trajectories.ndim != 2 or not np.all(np.isfinite(trajectories)):
        raise ValueError("temporal read-outs must form a finite two-dimensional array")
    actual_n_nodes, actual_traj_len = trajectories.shape
    actual_n_steps = max(actual_traj_len - 1, 0)

    # Spectral axis: per-node binned spectral entropy.
    per_node_entropy = np.zeros(actual_n_nodes, dtype=float)
    per_node_variance = np.zeros(actual_n_nodes, dtype=float)
    for i in range(actual_n_nodes):
        traj = trajectories[i]
        per_node_variance[i] = float(np.var(traj))
        p = _binned_psd_distribution(traj, int(n_bins))
        per_node_entropy[i] = _shannon_entropy(p)
    mean_entropy = float(np.mean(per_node_entropy)) if actual_n_nodes > 0 else 0.0
    max_entropy = math.log(float(n_bins))
    signature = float(mean_entropy / max_entropy) if max_entropy > 0.0 else 0.0
    signature = max(0.0, min(1.0, signature))
    effective_modes = float(math.exp(mean_entropy))

    # Storage axis: scan raw EPI storage form across nodes.
    from ..alias import get_attr
    from ..constants.aliases import ALIAS_EPI

    storage_values = [
        get_attr(G.nodes[n], ALIAS_EPI, conv=lambda value: value, strict=True)
        for n in list(G.nodes())
    ]
    bepi_fraction, bepi_count, bepi_total = _bepi_storage_fraction(
        storage_values, atol=float(storage_atol)
    )

    # Only inspected storage determines this factual label.
    if bepi_total == 0:
        verdict = "NO_STORAGE_OBSERVED"
    elif bepi_count:
        verdict = "NONSCALAR_BEPI_STORAGE_OBSERVED"
    else:
        verdict = "REAL_SCALAR_STORAGE"

    diagnostics: dict[str, Any] = {
        "per_node_spectral_entropy_nats": per_node_entropy.tolist(),
        "per_node_trajectory_variance": per_node_variance.tolist(),
        "mean_trajectory_variance": (
            float(np.mean(per_node_variance)) if actual_n_nodes > 0 else 0.0
        ),
        "max_entropy_nats_log_b": max_entropy,
        "scalar_threshold": float(scalar_threshold),
        "bepi_threshold": float(bepi_threshold),
        "storage_atol": float(storage_atol),
        "legacy_thresholds_used_for_verdict": False,
        "storage_tolerance_used": False,
        "storage_membership_owner": "tnfr.types.real_scalar_epi",
        "spectral_projection": "signed uniform-real; otherwise maximum component magnitude",
        "dimensional_necessity_assessed": False,
        "minimal_realization_assessed": False,
        "seed": int(seed),
        "scope": (
            "Final storage and projected temporal-spectrum observations only. "
            "Neither spectral entropy nor a storage label proves dimensional necessity. "
            "Historical result fields are retained; no RH conclusion."
        ),
    }

    return EpiTypeSignatureCertificate(
        signature=signature,
        storage_bepi_fraction=bepi_fraction,
        storage_bepi_count=bepi_count,
        storage_total=bepi_total,
        mean_spectral_entropy_nats=mean_entropy,
        effective_modes=effective_modes,
        n_nodes=actual_n_nodes,
        n_steps=actual_n_steps,
        n_bins=int(n_bins),
        verdict=verdict,
        diagnostics=diagnostics,
    )
