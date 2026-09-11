"""EPI-Type Signature — Diagnostic for the T-EPI Conjecture (§13triginta-quarta).

This module implements a purely diagnostic quantity, the **EPI-Type
Signature** :math:`\\mathcal{S}_{\\mathrm{EPI}}`, that quantifies on
canonical TNFR network evolutions the irreducible vectorial /
BEPIElement-valued content of EPI(t) trajectories.

Methodological scope (mandatory honesty)
----------------------------------------
This module is a *diagnostic only*.  It does **not** construct, promote,
or modify any canonical operator.  It does **not** advance G4 = RH.
It does **not** by itself decide the T-EPI Conjecture (which requires
the foundational analysis of §13triginta-quarta.3–.5 about the
Banach-space promotion via the BEPIElement formalisation).

The diagnostic probes two orthogonal axes:

1. **Storage axis** — what fraction of nodes carry actual non-trivial
   :class:`~tnfr.mathematics.epi.BEPIElement` storage (non-trivial
   ``f_continuous`` variance or non-trivial ``a_discrete`` magnitude)
   after a canonical operator sequence has run.
2. **Spectral axis** — Shannon entropy of the binned EPI temporal
   trajectory spectrum, averaged across nodes, normalised by
   :math:`\\log B`.

A high :math:`\\mathcal{S}_{\\mathrm{EPI}}` is a *necessary-condition*
check: it says only that the canonical EPI trajectories on a given
TNFR graph carry irreducible multi-modal structure that a single-mode
scalar reading cannot represent without loss.  It does **not** prove
that the canonical type of EPI is a non-trivial BEPIElement.

A low :math:`\\mathcal{S}_{\\mathrm{EPI}}` plus a zero storage fraction
is the empirically expected outcome, structurally confirming the
catalog's scalar-reading discipline.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13triginta-quarta
- ``src/tnfr/mathematics/epi.py`` (BEPIElement definition)
- ``src/tnfr/mathematics/spaces.py`` (BanachSpaceEPI)
- ``src/tnfr/alias.py::_bepi_to_float`` (scalar projection)
- ``src/tnfr/operators/nodal_equation.py`` (literal scalar contract)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

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

    Uses the real-FFT magnitude (DC component included) and bins the
    resulting energy distribution onto ``n_bins`` uniform bins.
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
    # Histogram of spectral energy across uniform frequency bins.
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
    """Fraction of storage entries that are non-trivially BEPI-valued.

    A storage value counts as *non-trivially BEPI* iff:

    - it is a :class:`~tnfr.mathematics.epi.BEPIElement` instance (or
      duck-compatible: has ``f_continuous`` and ``a_discrete``
      array attributes), AND
    - the standard deviation of ``f_continuous`` exceeds ``atol``, OR
    - the maximum magnitude of ``a_discrete`` exceeds ``atol``.

    Plain ``float``/``int`` storage and constant-mode BEPIElement
    instances (trivial embedding of scalars) count as scalar-form.
    """
    n_total = 0
    n_nontrivial = 0
    for value in storage_values:
        n_total += 1
        f_cont = getattr(value, "f_continuous", None)
        a_disc = getattr(value, "a_discrete", None)
        if f_cont is None or a_disc is None:
            continue
        try:
            f_arr = np.asarray(f_cont)
            a_arr = np.asarray(a_disc)
        except Exception:
            continue
        f_std = float(np.std(np.abs(f_arr))) if f_arr.size > 0 else 0.0
        a_max = float(np.max(np.abs(a_arr))) if a_arr.size > 0 else 0.0
        if f_std > atol or a_max > atol:
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
        state).  Values are taken through the canonical scalar reading
        ``_bepi_to_float`` so storage form is irrelevant for the
        spectral axis.
    """
    from ..alias import _bepi_to_float, get_attr
    from ..constants import inject_defaults
    from ..constants.aliases import ALIAS_EPI
    from ..dynamics import step

    # Inject canonical defaults (VF_ADAPT_MU, VF_ADAPT_TAU, etc.) so the
    # canonical step() function has its required graph parameters.
    inject_defaults(G)

    nodes = list(G.nodes())
    snapshots: list[list[float]] = [
        [_bepi_to_float(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) for n in nodes]
    ]
    # step() falls back to default_compute_delta_nfr if no hook is set.
    for _ in range(int(n_steps)):
        step(G)
        snapshots.append(
            [_bepi_to_float(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) for n in nodes]
        )
    return np.asarray(snapshots, dtype=float).T


@dataclass(frozen=True)
class EpiTypeSignatureCertificate:
    """Result of the EPI-Type Signature diagnostic on a canonical network.

    Attributes
    ----------
    signature : float
        :math:`\\mathcal{S}_{\\mathrm{EPI}} \\in [0, 1]`.  ``0`` means
        scalar-adequate temporal trajectories (single-mode evolution);
        ``1`` means maximum non-scalar (uniform-spectrum) content.
    storage_bepi_fraction : float
        Fraction of node EPI storage entries that are non-trivially
        BEPI-valued (non-constant ``f_continuous`` or non-zero
        ``a_discrete``).  ``0.0`` is the empirically expected value
        when no canonical operator constructs non-trivial BEPI elements.
    storage_bepi_count : int
        Absolute number of non-trivially BEPI-valued storage entries.
    storage_total : int
        Total number of inspected storage entries.
    mean_spectral_entropy_nats : float
        Mean Shannon entropy of the binned EPI temporal trajectory
        spectrum across nodes, in nats.
    effective_modes : float
        :math:`N_{\\mathrm{eff}} = \\exp(H)` — effective spectral mode
        count.  Scalar-adequate iff :math:`N_{\\mathrm{eff}} \\approx 1`.
    n_nodes : int
        Number of nodes in the diagnostic graph.
    n_steps : int
        Number of evolution steps taken (trajectory length is
        ``n_steps + 1`` per node).
    n_bins : int
        Number of histogram bins used for the spectral distribution.
    verdict : str
        One of ``"SCALAR_ADEQUATE"`` (signature < ``scalar_threshold``
        AND zero BEPI storage), ``"BEPI_VALUED_NECESSARY"``
        (signature > ``bepi_threshold`` OR non-zero BEPI storage),
        or ``"INDETERMINATE"``.
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
            "EPI-Type Signature certificate (diagnostic only — §13triginta-quarta.6)",
            f"  signature S_EPI         : {self.signature:.6f}   (0 = scalar, 1 = uniform)",
            f"  storage BEPI fraction   : {self.storage_bepi_fraction:.4f}"
            f"  ({self.storage_bepi_count}/{self.storage_total} nodes)",
            f"  mean spectral entropy   : {self.mean_spectral_entropy_nats:.4f} nats"
            f" over {self.n_bins} bins",
            f"  effective modes N_eff   : {self.effective_modes:.2f}",
            f"  graph: {self.n_nodes} nodes, {self.n_steps} evolution steps",
            f"  verdict                 : {self.verdict}",
            "  scope: necessary-condition diagnostic; does NOT advance G4 = RH",
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
    scalar_threshold : float, default 0.15
        Below this signature value AND with zero BEPI storage, the
        verdict is ``"SCALAR_ADEQUATE"``.
    bepi_threshold : float, default 0.5
        Above this signature value OR with non-zero BEPI storage, the
        verdict is ``"BEPI_VALUED_NECESSARY"``.
    storage_atol : float, default 1e-12
        Absolute tolerance below which a BEPIElement is treated as a
        trivial embedding of a scalar (constant ``f_continuous``,
        zero ``a_discrete``).

    Returns
    -------
    EpiTypeSignatureCertificate
        Diagnostic certificate.

    Notes
    -----
    The diagnostic uses two orthogonal axes:

    - **Spectral axis**: per-node binned spectral entropy of the EPI
      temporal trajectory (mean across nodes), normalised by
      :math:`\\log B`.  This probes how multi-modal canonical EPI
      evolution actually is.
    - **Storage axis**: a direct scan of the EPI storage form for
      non-trivial BEPIElement content.  Under the canonical 13-operator
      catalog, this fraction is expected to be ``0`` (no operator
      constructs non-trivial ``f_continuous`` or ``a_discrete``),
      empirically witnessing the catalog's scalar-reading discipline.

    This is a *purely diagnostic* computation on canonical TNFR data.
    It does not construct any new operator and does not modify the
    13-operator catalog.
    """
    if int(n_nodes) < 3:
        raise ValueError("n_nodes must be >= 3 for a meaningful ring graph")
    if int(n_steps) < 4:
        raise ValueError("n_steps must be >= 4 for a meaningful trajectory")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be >= 2")

    G = _build_canonical_demo_graph(int(n_nodes), int(seed))
    trajectories = _evolve_and_collect(G, int(n_steps))
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
    storage_values = [G.nodes[n].get("EPI") for n in list(G.nodes())]
    bepi_fraction, bepi_count, bepi_total = _bepi_storage_fraction(
        storage_values, atol=float(storage_atol)
    )

    # Verdict.
    if bepi_fraction > 0.0 or signature > bepi_threshold:
        verdict = "BEPI_VALUED_NECESSARY"
    elif signature < scalar_threshold and bepi_fraction == 0.0:
        verdict = "SCALAR_ADEQUATE"
    else:
        verdict = "INDETERMINATE"

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
        "seed": int(seed),
        "scope": (
            "Necessary-condition diagnostic for T-EPI Conjecture "
            "(§13triginta-quarta). Does NOT advance G4 = RH."
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
