"""φ-Type Signature — Diagnostic for the T-φ Conjecture (§13triginta-octava).

This module implements a purely diagnostic quantity, the **φ-Type
Signature** :math:`\\mathcal{S}_{\\phi}`, that quantifies on canonical
TNFR network evolutions whether the canonical phase ``φ ∈ S¹`` admits
an *irreducible* covering-space (multi-sheet) lift, or whether the
single-sheet :math:`[-\\pi, \\pi]` (a.k.a. :math:`[0, 2\\pi)`) storage
is structurally sufficient.

Methodological scope (mandatory honesty)
----------------------------------------
This module is a *diagnostic only*.  It does **not** construct,
promote, or modify any canonical operator.  It does **not** advance
G4 = RH.  It does **not** by itself decide the T-φ Conjecture (which
requires the foundational analysis of §13triginta-octava.3–.5 about
the covering-space promotion and the forcing-axiom inventory of
§13triginta-nona — both deferred to B2b/B2c).

The diagnostic probes two orthogonal axes:

1. **Winding storage axis** — the fraction of nodes whose unwrapped
   phase trajectory across canonical evolution accumulates at least
   one full revolution (:math:`|\\Delta\\phi_{\\mathrm{unwrap}}| \\ge
   2\\pi`).  A non-zero fraction would mean canonical evolution
   actually produces topologically-charged windings that the single
   fundamental-domain representation :math:`\\phi \\in [-\\pi, \\pi]`
   discards — flagging a covering-space lift as potentially
   necessary.
2. **Lift-spectral axis** — Shannon entropy of the binned
   phase-velocity spectrum (frequencies of ``dφ/dt`` computed by
   wrapped finite differences), averaged across nodes, normalised by
   :math:`\\log B`.  A multi-modal phase-velocity spectrum is a
   *necessary* condition for the canonical dynamics to require a
   multi-sheet lift (a single covering sheet is fully equivalent to
   the base if all phase velocity sits in one canonical mode).

A high :math:`\\mathcal{S}_{\\phi}` and/or non-zero winding fraction is a
*necessary-condition* check: it says only that the canonical phase
evolution on a given TNFR graph carries irreducible structure that a
naive single-sheet :math:`S^{1}` reading cannot represent without
loss.  It does **not** prove that the canonical type of φ is a
non-trivial covering element.

A low :math:`\\mathcal{S}_{\\phi}` plus a zero winding fraction is the
empirically expected outcome — structurally consistent with the
catalog row 5 result of §13triginta-prima.4 that
:math:`\\widehat{\\mathbb{Z}} = S^{1}` is the canonical Pontryagin
partner of the (already-NEGATIVE B0) :math:`\\nu_f`-type, and with
the canonical wrapping discipline of
:func:`~tnfr.physics._helpers.wrap_angle`.

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13triginta-octava
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §4 row B2
- ``src/tnfr/physics/_helpers.py::wrap_angle`` (canonical wrapping)
- ``src/tnfr/physics/_helpers.py::get_phase`` (scalar reader)
- ``src/tnfr/constants/aliases.py::ALIAS_THETA`` (storage aliases)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "PhiTypeSignatureCertificate",
    "compute_phi_type_signature",
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

    Uses the real-FFT magnitude (DC component removed by mean
    subtraction) and bins the resulting energy distribution onto
    ``n_bins`` uniform bins.  Trajectories with zero variance return a
    degenerate distribution (all mass in the first bin).
    """
    x = np.asarray(trajectory, dtype=float).ravel()
    if x.size < 2:
        p = np.zeros(n_bins, dtype=float)
        p[0] = 1.0
        return p
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


def _wrap_to_pi(angle: float) -> float:
    """Wrap ``angle`` to the canonical fundamental domain ``[-π, π]``.

    Mirrors :func:`tnfr.physics._helpers.wrap_angle` without importing
    it (the diagnostic must be runnable in isolation if needed).
    """
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _winding_fraction(
    phase_trajectories: np.ndarray, *, winding_atol: float = 1e-9
) -> tuple[float, int, int, np.ndarray]:
    """Fraction of nodes whose unwrapped phase escapes the fundamental domain.

    Parameters
    ----------
    phase_trajectories : np.ndarray
        Matrix of shape ``(n_nodes, n_steps + 1)`` of *wrapped* phase
        values in :math:`[-\\pi, \\pi]`.
    winding_atol : float, default 1e-9
        Numerical tolerance on the cumulative winding magnitude before
        a node is counted as winding-non-trivial.  Set strictly below
        :math:`2\\pi - 1\\,\\mathrm{rad}` to avoid false positives from
        rounding alone.

    Returns
    -------
    (fraction, count, total, windings) : tuple
        ``fraction`` is the share of nodes with
        :math:`|\\Delta\\phi_{\\mathrm{unwrap}}| \\ge 2\\pi -
        \\mathrm{winding\\_atol}`.  ``windings`` is the per-node
        rounded winding count :math:`\\lfloor
        |\\Delta\\phi_{\\mathrm{unwrap}}| / (2\\pi) + 0.5 \\rfloor`.
    """
    if phase_trajectories.ndim != 2 or phase_trajectories.shape[1] < 2:
        n_nodes = int(phase_trajectories.shape[0]) if phase_trajectories.ndim > 0 else 0
        return 0.0, 0, n_nodes, np.zeros(n_nodes, dtype=int)
    n_nodes = phase_trajectories.shape[0]
    unwrapped = np.unwrap(phase_trajectories, axis=1)
    deltas = unwrapped[:, -1] - unwrapped[:, 0]
    abs_deltas = np.abs(deltas)
    threshold = 2.0 * math.pi - float(winding_atol)
    winding_mask = abs_deltas >= threshold
    n_winding = int(np.sum(winding_mask))
    fraction = float(n_winding) / float(n_nodes) if n_nodes > 0 else 0.0
    windings = np.floor(abs_deltas / (2.0 * math.pi) + 0.5).astype(int)
    return fraction, n_winding, n_nodes, windings


def _build_canonical_demo_graph(n_nodes: int, seed: int) -> Any:
    """Build a small canonical ring graph for the φ trajectory probe.

    Uses :func:`tnfr.sdk.TNFR.create` to obtain a TNFR network with
    canonical defaults and a fixed ring topology so the diagnostic is
    deterministic given the seed.  The initial phase distribution is a
    deterministic mild perturbation around zero so canonical evolution
    starts away from a trivial symmetric fixed point.
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
    return G


def _evolve_and_collect(G: Any, n_steps: int) -> np.ndarray:
    """Run ``n_steps`` canonical evolution steps and collect φ per node.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(n_nodes, n_steps + 1)`` with the *wrapped*
        phase value of every node at every collected step (including
        the initial state).  Values are read through the canonical
        scalar reader :func:`~tnfr.physics._helpers.get_phase` and
        re-wrapped to :math:`[-\\pi, \\pi]` to be a faithful sample of
        the canonical storage form.
    """
    from ..constants import inject_defaults
    from ..dynamics import step
    from ..physics._helpers import get_phase

    inject_defaults(G)

    nodes = list(G.nodes())
    snapshots: list[list[float]] = [[
        _wrap_to_pi(get_phase(G, n)) for n in nodes
    ]]
    for _ in range(int(n_steps)):
        step(G)
        snapshots.append([
            _wrap_to_pi(get_phase(G, n)) for n in nodes
        ])
    return np.asarray(snapshots, dtype=float).T


@dataclass(frozen=True)
class PhiTypeSignatureCertificate:
    """Result of the φ-Type Signature diagnostic on a canonical network.

    Attributes
    ----------
    signature : float
        :math:`\\mathcal{S}_{\\phi} \\in [0, 1]`.  ``0`` means
        scalar-adequate single-mode phase-velocity dynamics; ``1``
        means maximum-entropy (uniform-spectrum) phase-velocity
        content.
    winding_fraction : float
        Fraction of nodes whose unwrapped phase trajectory accumulates
        :math:`|\\Delta\\phi_{\\mathrm{unwrap}}| \\ge 2\\pi`.  ``0.0``
        is the empirically expected value under the canonical
        wrap-to-:math:`[-\\pi, \\pi]` discipline; a non-zero value
        flags that canonical evolution actually produces topological
        windings discarded by the single-sheet storage.
    winding_count : int
        Absolute number of nodes counted as winding-non-trivial.
    n_nodes : int
        Number of nodes in the diagnostic graph.
    max_unwrap_delta_rad : float
        Maximum across nodes of :math:`|\\Delta\\phi_{\\mathrm{unwrap}}|`
        in radians (sanity check for the winding fraction).
    mean_winding_count : float
        Mean across nodes of the rounded winding count
        :math:`\\lfloor |\\Delta\\phi_{\\mathrm{unwrap}}| / (2\\pi) +
        0.5 \\rfloor`.
    mean_spectral_entropy_nats : float
        Mean Shannon entropy of the binned phase-velocity spectrum
        across nodes, in nats.
    effective_modes : float
        :math:`N_{\\mathrm{eff}} = \\exp(H)` — effective spectral mode
        count for the phase-velocity spectrum.  Scalar-adequate iff
        :math:`N_{\\mathrm{eff}} \\approx 1`.
    n_steps : int
        Number of evolution steps taken (trajectory length is
        ``n_steps + 1`` per node).
    n_bins : int
        Number of histogram bins used for the spectral distribution.
    verdict : str
        One of ``"SCALAR_S1_ADEQUATE"`` (signature < ``scalar_threshold``
        AND zero winding fraction), ``"COVER_LIFT_NECESSARY"``
        (signature > ``cover_threshold`` OR non-zero winding
        fraction), or ``"INDETERMINATE"``.
    diagnostics : dict
        Auxiliary fields (per-node entropies, per-node winding counts,
        per-node phase variance, etc.).
    """

    signature: float
    winding_fraction: float
    winding_count: int
    n_nodes: int
    max_unwrap_delta_rad: float
    mean_winding_count: float
    mean_spectral_entropy_nats: float
    effective_modes: float
    n_steps: int
    n_bins: int
    verdict: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "phi-Type Signature certificate (diagnostic only — §13triginta-octava.5)",
            f"  signature S_phi          : {self.signature:.6f}   (0 = scalar S^1, 1 = uniform)",
            f"  winding fraction         : {self.winding_fraction:.4f}"
            f"  ({self.winding_count}/{self.n_nodes} nodes)",
            f"  max unwrap |delta_phi|   : {self.max_unwrap_delta_rad:.4f} rad",
            f"  mean rounded windings    : {self.mean_winding_count:.4f}",
            f"  mean spectral entropy    : {self.mean_spectral_entropy_nats:.4f} nats"
            f" over {self.n_bins} bins",
            f"  effective modes N_eff    : {self.effective_modes:.2f}",
            f"  graph: {self.n_nodes} nodes, {self.n_steps} evolution steps",
            f"  verdict                  : {self.verdict}",
            "  scope: necessary-condition diagnostic; does NOT advance G4 = RH",
        ]
        return "\n".join(lines)


def compute_phi_type_signature(
    *,
    n_nodes: int = 24,
    n_steps: int = 64,
    n_bins: int = 32,
    seed: int = 13,
    scalar_threshold: float = 0.15,
    cover_threshold: float = 0.5,
    winding_atol: float = 1e-9,
) -> PhiTypeSignatureCertificate:
    """Compute the φ-Type Signature on a canonical TNFR ring evolution.

    Parameters
    ----------
    n_nodes : int, default 24
        Size of the ring graph used as the canonical probe.
    n_steps : int, default 64
        Number of evolution steps after initial state collection.
        Trajectory length per node is ``n_steps + 1``.
    n_bins : int, default 32
        Histogram resolution :math:`B` for the phase-velocity
        spectral distribution.  The maximum entropy is :math:`\\log B`;
        the signature is normalised by this maximum so that
        :math:`\\mathcal{S}_{\\phi} \\in [0, 1]`.
    seed : int, default 13
        Deterministic seed for the initial phase / EPI perturbation.
    scalar_threshold : float, default 0.15
        Below this signature value AND with zero winding fraction,
        the verdict is ``"SCALAR_S1_ADEQUATE"``.
    cover_threshold : float, default 0.5
        Above this signature value OR with non-zero winding fraction,
        the verdict is ``"COVER_LIFT_NECESSARY"``.
    winding_atol : float, default 1e-9
        Tolerance on the unwrapped-phase delta below which a single
        revolution is not counted (guards against rounding-only
        winding counts).

    Returns
    -------
    PhiTypeSignatureCertificate
        Diagnostic certificate.

    Notes
    -----
    The diagnostic uses two orthogonal axes:

    - **Winding axis**: per-node unwrapped phase delta across the
      trajectory; counts nodes with :math:`|\\Delta\\phi_{\\mathrm{unwrap}}|
      \\ge 2\\pi - \\mathrm{winding\\_atol}`.  Under the canonical
      wrap-to-:math:`[-\\pi, \\pi]` discipline of
      :func:`~tnfr.physics._helpers.wrap_angle`, the storage form
      *discards* this information at every step; a non-zero winding
      fraction in the *unwrapped* reconstruction is a necessary
      condition for a covering-space lift to be canonically required.
    - **Lift-spectral axis**: per-node binned spectral entropy of the
      phase-velocity time series :math:`\\dot\\phi(t) \\approx
      \\mathrm{wrap}(\\phi(t+1) - \\phi(t))` (mean across nodes),
      normalised by :math:`\\log B`.  Probes how multi-modal canonical
      phase-velocity evolution actually is.

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
    phase_trajectories = _evolve_and_collect(G, int(n_steps))
    actual_n_nodes, actual_traj_len = phase_trajectories.shape
    actual_n_steps = max(actual_traj_len - 1, 0)

    # Winding axis (operates on the wrapped trajectory; np.unwrap inside).
    winding_fraction, winding_count, n_nodes_eff, windings = _winding_fraction(
        phase_trajectories, winding_atol=float(winding_atol)
    )
    if actual_traj_len >= 2:
        unwrapped = np.unwrap(phase_trajectories, axis=1)
        max_unwrap = float(
            np.max(np.abs(unwrapped[:, -1] - unwrapped[:, 0]))
        ) if actual_n_nodes > 0 else 0.0
    else:
        max_unwrap = 0.0
    mean_winding_count = (
        float(np.mean(windings)) if windings.size > 0 else 0.0
    )

    # Lift-spectral axis: phase-velocity (wrapped finite differences).
    per_node_entropy = np.zeros(actual_n_nodes, dtype=float)
    per_node_phase_variance = np.zeros(actual_n_nodes, dtype=float)
    for i in range(actual_n_nodes):
        traj = phase_trajectories[i]
        per_node_phase_variance[i] = float(np.var(traj))
        if traj.size < 2:
            per_node_entropy[i] = 0.0
            continue
        # Phase velocity via canonical wrap of finite differences.
        raw_diff = np.diff(traj)
        velocity = np.asarray(
            [_wrap_to_pi(float(d)) for d in raw_diff], dtype=float
        )
        p = _binned_psd_distribution(velocity, int(n_bins))
        per_node_entropy[i] = _shannon_entropy(p)
    mean_entropy = (
        float(np.mean(per_node_entropy)) if actual_n_nodes > 0 else 0.0
    )

    max_entropy = math.log(float(n_bins))
    signature = mean_entropy / max_entropy if max_entropy > 0.0 else 0.0
    signature = float(min(max(signature, 0.0), 1.0))
    effective_modes = float(math.exp(mean_entropy)) if mean_entropy > 0.0 else 1.0

    if signature < float(scalar_threshold) and winding_fraction == 0.0:
        verdict = "SCALAR_S1_ADEQUATE"
    elif signature > float(cover_threshold) or winding_fraction > 0.0:
        verdict = "COVER_LIFT_NECESSARY"
    else:
        verdict = "INDETERMINATE"

    diagnostics: dict[str, Any] = {
        "per_node_phase_velocity_entropy_nats": per_node_entropy.tolist(),
        "per_node_phase_variance": per_node_phase_variance.tolist(),
        "per_node_winding_count": windings.tolist(),
        "scalar_threshold": float(scalar_threshold),
        "cover_threshold": float(cover_threshold),
        "winding_atol": float(winding_atol),
        "max_entropy_nats": float(max_entropy),
        "seed": int(seed),
    }

    return PhiTypeSignatureCertificate(
        signature=signature,
        winding_fraction=float(winding_fraction),
        winding_count=int(winding_count),
        n_nodes=int(actual_n_nodes),
        max_unwrap_delta_rad=float(max_unwrap),
        mean_winding_count=float(mean_winding_count),
        mean_spectral_entropy_nats=float(mean_entropy),
        effective_modes=float(effective_modes),
        n_steps=int(actual_n_steps),
        n_bins=int(n_bins),
        verdict=verdict,
        diagnostics=diagnostics,
    )
