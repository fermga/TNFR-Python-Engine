"""REMESH-Window-Type Signature — Diagnostic for the T-REMESH-window Conjecture (§13quadraginta-tertia).

This module implements a purely diagnostic quantity, the **REMESH-
Window-Type Signature** :math:`\\mathcal{S}_{\\tau}`, that quantifies
on canonical TNFR network evolutions whether the canonical REMESH
memory window
:math:`(\\tau_l, \\tau_g) \\in \\mathbb{N} \\times \\mathbb{N}`
admits an *irreducible* continuous-time or fractional-order lift
(e.g. a continuous kernel :math:`K(t, s)` with :math:`s \\in [0, t]`,
or a fractional-order discrete window
:math:`\\tau_l, \\tau_g \\in \\mathbb{R}^+ \\setminus \\mathbb{N}`),
or whether the integer-indexed history vector read by
:func:`tnfr.operators.remesh.apply_network_remesh` is structurally
sufficient.

Methodological scope (mandatory honesty)
----------------------------------------
This module is a *diagnostic only*.  It does **not** construct,
promote, or modify any canonical operator.  It does **not** advance
G4 = RH.  It does **not** by itself decide the T-REMESH-window
Conjecture (which requires the forcing-axiom reduction of
§13quadraginta-quarta and the final verdict of
§13quadraginta-quinta — both deferred to B4b/B4c).

The diagnostic probes two orthogonal axes:

1. **Integer-index storage axis** — the fraction of REMESH-bearing
   parameter reads at which the canonical window slot
   ``G.graph["REMESH_TAU_LOCAL"]`` / ``G.graph["REMESH_TAU_GLOBAL"]``
   stores a non-integer-coercible value (float with non-zero
   fractional part, NumPy float, mapping, tensor, callable, or
   otherwise non-``int`` payload).  Under the canonical
   implementation
   :func:`tnfr.operators.remesh.apply_network_remesh`, both slots
   are coerced via ``int(get_param(...))`` at every read, so this
   fraction is structurally ``0`` — exactly mirroring
   ``w_frac = 0`` (B2a), ``bepi_frac = 0`` (B1a), and
   ``T_frac = 0`` (B3a).
2. **Window-refinement sensitivity axis** — variance of the
   post-REMESH per-node EPI snapshot across a small bracket of
   adjacent integer windows
   :math:`\\{(\\tau_l + j, \\tau_g + j) : j = 0, 1, 2\\}`,
   normalised to :math:`[0, 1]`.  Low variance is a *necessary*
   condition for the canonical integer-indexed REMESH to be
   structurally adequate: it says that the canonical step-function
   :math:`\\tau \\mapsto (\\text{EPI history slot})` is already
   *Lipschitz-smooth* in :math:`\\tau` at the resolution at which
   the dynamics evolves, so no continuous kernel
   :math:`K(t, s)` is forced by canonical evolution.

A high :math:`\\mathcal{S}_{\\tau}` is a *necessary-condition*
check: it says only that adjacent integer windows produce
substantially different post-REMESH states, so a continuous-time
kernel *might* be required to disambiguate the true asymptotic
target.  It does **not** prove that the canonical type of the
REMESH window is a non-trivial continuous kernel.

A low :math:`\\mathcal{S}_{\\tau}` plus a zero integer-storage
fraction is the empirically expected outcome — structurally
consistent with the catalog row 1 typing
:math:`(\\tau_l, \\tau_g) \\in \\mathbb{N} \\times \\mathbb{N}`,
with the N15 REMESH-∞ closure
(``theory/REMESH_INFINITY_DERIVATION.md``, §§1–8) supplying the
asymptotic-limit discharge mechanism (mean ergodic theorem
applied to the contractive transfer matrix at integer
:math:`\\tau_g \\to \\infty`), and with the integer-indexed
history read at
``src/tnfr/operators/remesh.py:1218–1228`` (``int(get_param(...))``
+ ``hist[-(tau_g + 1)]``).

References
----------
- ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` §13quadraginta-tertia
- ``theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`` §4 row B4
- ``theory/REMESH_INFINITY_DERIVATION.md`` §§1–8 (N15 closure;
  canonical asymptotic-limit discharge mechanism)
- ``src/tnfr/operators/remesh.py:1212::apply_network_remesh``
  (canonical integer-indexed implementation)
- ``src/tnfr/config/defaults_core.py:221–222`` (canonical defaults
  ``REMESH_TAU_GLOBAL: int = 8``, ``REMESH_TAU_LOCAL: int = 4``)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "RemeshWindowTypeSignatureCertificate",
    "compute_remesh_window_type_signature",
]


def _wrap_to_pi(angle: float) -> float:
    """Wrap ``angle`` to the canonical fundamental domain ``[-π, π]``."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _is_integer_payload(value: Any) -> bool:
    """Return ``True`` iff ``value`` is structurally a non-negative integer.

    Accepts: Python ``int`` (excluding ``bool``), NumPy integer scalar,
    and Python ``float`` whose fractional part is exactly zero (after
    safe conversion).  Rejects: floats with non-zero fractional part,
    NumPy ``float`` arrays of dimension > 0, mappings, sequences,
    callables, None.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, np.integer):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, np.floating):
        return float(value).is_integer()
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return False
        try:
            return float(value).is_integer()
        except (TypeError, ValueError):
            return False
    return False


def _build_canonical_demo_graph(n_nodes: int, seed: int) -> Any:
    """Build a small canonical ring graph for the REMESH-window probe.

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
        current_vf = float(G.nodes[node].get("nu_f", 1.0))
        G.nodes[node]["nu_f"] = max(0.05, current_vf + 0.05 * (rng.random() - 0.5))
    return G


def _read_epi_snapshot(G: Any, nodes: list[Any]) -> np.ndarray:
    """Return per-node EPI vector in deterministic ``nodes`` order."""
    out = np.zeros(len(nodes), dtype=float)
    for i, node in enumerate(nodes):
        out[i] = float(G.nodes[node].get("EPI", 0.0))
    return out


def _inspect_tau_storage(G: Any) -> tuple[int, int]:
    """Inspect raw ``REMESH_TAU_*`` payloads stored on ``G.graph``.

    Returns
    -------
    n_integer : int
        Number of canonical window slots (out of 2) whose raw payload
        is structurally an integer.
    n_total : int
        Total number of canonical window slots inspected (2 per call).
    """
    n_total = 2
    n_integer = 0
    raw_l = G.graph.get("REMESH_TAU_LOCAL")
    raw_g = G.graph.get("REMESH_TAU_GLOBAL")
    if raw_l is not None and _is_integer_payload(raw_l):
        n_integer += 1
    if raw_g is not None and _is_integer_payload(raw_g):
        n_integer += 1
    return n_integer, n_total


def _evolve_to_steady_history(G: Any, warmup_steps: int) -> None:
    """Run ``warmup_steps`` canonical steps so ``_epi_hist`` is populated."""
    from ..constants import inject_defaults
    from ..dynamics import step

    inject_defaults(G)
    for _ in range(int(warmup_steps)):
        step(G)


def _run_remesh_bracket(
    n_nodes: int,
    seed: int,
    warmup_steps: int,
    tau_l_base: int,
    tau_g_base: int,
    n_events: int,
) -> tuple[np.ndarray, list[Any], int, int]:
    """Run the canonical REMESH event ``n_events`` times for each window in the bracket.

    For each integer offset ``j ∈ {0, 1, 2}``, rebuild a fresh
    canonical demo graph from ``seed`` (deterministically identical
    pre-REMESH state across bracket entries), warm up ``_epi_hist``
    with ``warmup_steps`` canonical steps, set
    ``(τ_l + j, τ_g + j)`` on the freshly-built graph, fire
    :func:`apply_network_remesh` ``n_events`` times, and record the
    final per-node EPI snapshot.  The rebuild-per-bracket strategy
    is used because the canonical TNFR graph carries non-picklable
    runtime locks (``_thread.RLock``) and cannot be ``deepcopy``-d.

    Returns
    -------
    epi_bracket : np.ndarray of shape ``(3, n_nodes)``
        Final EPI vectors for the three bracket windows.
    nodes : list[Any]
        Canonical node order used to interpret the EPI columns.
    n_integer_samples : int
        Cumulative number of integer-coerced window-slot reads across
        the bracket.
    n_total_samples : int
        Cumulative number of window-slot reads across the bracket.
    """
    from ..operators.remesh import apply_network_remesh

    epi_bracket = np.zeros((3, int(n_nodes)), dtype=float)
    n_integer_samples = 0
    n_total_samples = 0
    nodes_canonical: list[Any] | None = None
    for j in range(3):
        G_j = _build_canonical_demo_graph(n_nodes, seed)
        _evolve_to_steady_history(G_j, warmup_steps)
        nodes_j = list(G_j.nodes())
        if nodes_canonical is None:
            nodes_canonical = nodes_j
        G_j.graph["REMESH_TAU_LOCAL"] = int(tau_l_base + j)
        G_j.graph["REMESH_TAU_GLOBAL"] = int(tau_g_base + j)
        for _ in range(int(n_events)):
            n_int, n_tot = _inspect_tau_storage(G_j)
            n_integer_samples += n_int
            n_total_samples += n_tot
            apply_network_remesh(G_j)
        epi_bracket[j, :] = _read_epi_snapshot(G_j, nodes_j)
    assert nodes_canonical is not None
    return epi_bracket, nodes_canonical, n_integer_samples, n_total_samples


def _window_refinement_signature(
    epi_bracket: np.ndarray, *, eps: float = 1e-12
) -> tuple[float, float]:
    """Compute the normalised per-node EPI variance across the integer-window bracket.

    For each node, compute the variance of EPI across the three
    bracket windows, divide by the per-node mean absolute EPI plus
    ``eps`` (scale invariance), and average across nodes.  The result
    is clipped to ``[0, 1]`` by squashing through
    :math:`\\tanh(\\cdot)` so it is comparable to the B0/B1/B2/B3
    signatures.

    Returns
    -------
    signature : float
        Squashed signature in ``[0, 1]``.
    raw_relative_variance : float
        Raw per-node mean of variance/mean-abs-EPI (before squashing).
    """
    if epi_bracket.shape[0] < 2:
        return 0.0, 0.0
    per_node_var = np.var(epi_bracket, axis=0, ddof=0)
    per_node_scale = np.mean(np.abs(epi_bracket), axis=0) + eps
    rel_var = per_node_var / per_node_scale
    raw_mean = float(np.mean(rel_var))
    signature = float(math.tanh(raw_mean))
    return signature, raw_mean


@dataclass(frozen=True)
class RemeshWindowTypeSignatureCertificate:
    """Result of the REMESH-Window-Type Signature diagnostic on a canonical network.

    Attributes
    ----------
    signature : float
        :math:`\\mathcal{S}_{\\tau} \\in [0, 1]`.  ``0`` means
        integer-window-adequate (the three bracket windows produce
        essentially identical post-REMESH EPI snapshots); ``1`` means
        adjacent integer windows produce maximally different
        snapshots (continuous kernel may be necessary).
    integer_storage_fraction : float
        Fraction of REMESH-bearing window-slot reads in which the
        canonical slot stored an *integer-coercible* payload.  ``1.0``
        is the empirically expected value under the canonical
        implementation ``int(get_param(...))`` at every read; any value
        below ``1.0`` flags that canonical evolution stores a
        non-integer that the integer reader necessarily truncates.
    noninteger_storage_count : int
        Absolute number of window-slot reads that stored a non-integer
        payload (equals ``n_remesh_events * 2 - integer_storage_count``).
    n_remesh_events : int
        Total number of :func:`apply_network_remesh` events fired across
        the bracket (== ``3 * remesh_events_per_window``).
    n_nodes : int
        Number of nodes in the diagnostic graph.
    tau_l : int
        Base canonical local window τ_l used as bracket anchor.
    tau_g : int
        Base canonical global window τ_g used as bracket anchor.
    bracket : tuple[int, int, int]
        The three integer offsets applied to ``(τ_l, τ_g)``: ``(0, 1, 2)``.
    raw_relative_variance : float
        Pre-squash per-node mean of variance/mean-abs-EPI across the
        bracket (in EPI units).
    bracket_mean_l2 : float
        Mean across nodes of :math:`\\ell^2` distance between the
        baseline window's EPI snapshot and the ``+1`` and ``+2``
        windows' snapshots (advisory; complements
        ``raw_relative_variance``).
    verdict : str
        One of ``"INTEGER_WINDOW_ADEQUATE"`` (signature <
        ``scalar_threshold`` AND integer storage fraction == 1.0),
        ``"CONTINUOUS_KERNEL_NECESSARY"`` (signature >
        ``continuous_threshold`` OR integer storage fraction < 1.0),
        or ``"INDETERMINATE"``.
    diagnostics : dict
        Auxiliary fields (bracket EPI matrices, thresholds, seed,
        warmup steps, raw counters).
    """

    signature: float
    integer_storage_fraction: float
    noninteger_storage_count: int
    n_remesh_events: int
    n_nodes: int
    tau_l: int
    tau_g: int
    bracket: tuple[int, int, int]
    raw_relative_variance: float
    bracket_mean_l2: float
    verdict: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "REMESH-Window-Type Signature certificate (diagnostic only — §13quadraginta-tertia.5)",
            f"  signature S_tau          : {self.signature:.6f}   (0 = bracket-flat, 1 = bracket-saturated)",
            f"  integer storage fraction : {self.integer_storage_fraction:.4f}"
            f"  ({self.noninteger_storage_count} non-integer reads / "
            f"{self.n_remesh_events * 2} total reads)",
            f"  raw relative variance    : {self.raw_relative_variance:.6e}"
            f" (per-node Var(EPI)/<|EPI|> across bracket)",
            f"  bracket mean L2          : {self.bracket_mean_l2:.6f}"
            f" (baseline vs +1, +2 EPI snapshots)",
            f"  bracket windows          : (tau_l, tau_g) in "
            f"{{({self.tau_l}, {self.tau_g}),"
            f" ({self.tau_l + 1}, {self.tau_g + 1}),"
            f" ({self.tau_l + 2}, {self.tau_g + 2})}}",
            f"  graph                    : {self.n_nodes} nodes,"
            f" {self.n_remesh_events} REMESH events ({self.n_remesh_events // 3} per window)",
            f"  verdict                  : {self.verdict}",
            "  scope: necessary-condition diagnostic; does NOT advance G4 = RH",
        ]
        return "\n".join(lines)


def compute_remesh_window_type_signature(
    *,
    n_nodes: int = 24,
    warmup_steps: int = 16,
    tau_l: int = 4,
    tau_g: int = 8,
    remesh_events_per_window: int = 8,
    seed: int = 17,
    scalar_threshold: float = 0.15,
    continuous_threshold: float = 0.5,
) -> RemeshWindowTypeSignatureCertificate:
    """Compute the REMESH-Window-Type Signature on a canonical TNFR ring evolution.

    Parameters
    ----------
    n_nodes : int, default 24
        Size of the ring graph used as the canonical probe.
    warmup_steps : int, default 16
        Number of canonical evolution steps before the REMESH bracket
        starts firing.  Must be large enough that the canonical
        ``_epi_hist`` deque has accumulated ``> max(τ_l, τ_g) + 2``
        snapshots (else :func:`apply_network_remesh` early-returns).
        Default ``16`` covers the canonical defaults ``τ_l = 4``,
        ``τ_g = 8`` plus the ``+2`` bracket headroom.
    tau_l : int, default 4 (canonical default of
        ``REMESH_TAU_LOCAL``)
        Base local memory window τ_l used as bracket anchor.
    tau_g : int, default 8 (canonical default of
        ``REMESH_TAU_GLOBAL``)
        Base global memory window τ_g used as bracket anchor.
    remesh_events_per_window : int, default 8
        Number of :func:`apply_network_remesh` events fired per
        window in the bracket; total events == ``3 *
        remesh_events_per_window``.
    seed : int, default 17
        Deterministic seed for the initial phase / EPI / νf
        perturbation.
    scalar_threshold : float, default 0.15
        Below this signature value AND with integer storage fraction
        equal to ``1.0``, the verdict is ``"INTEGER_WINDOW_ADEQUATE"``.
    continuous_threshold : float, default 0.5
        Above this signature value OR with integer storage fraction
        below ``1.0``, the verdict is
        ``"CONTINUOUS_KERNEL_NECESSARY"``.

    Returns
    -------
    RemeshWindowTypeSignatureCertificate
        Diagnostic certificate.

    Notes
    -----
    The diagnostic uses two orthogonal axes:

    - **Integer-index storage axis**: per REMESH event, inspect the
      raw payloads stored at ``G.graph["REMESH_TAU_LOCAL"]`` and
      ``G.graph["REMESH_TAU_GLOBAL"]`` for non-integer-coercible
      values.  Under the canonical implementation
      :func:`tnfr.operators.remesh.apply_network_remesh`, every read
      is coerced via ``int(get_param(...))``; the integer-storage
      fraction is therefore structurally ``1.0`` by construction —
      exactly mirroring the :math:`w_{\\mathrm{frac}} = 0`,
      :math:`\\mathrm{bepi\\_frac} = 0`, and
      :math:`T_{\\mathrm{frac}} = 0` outcomes of the B2a/B1a/B3a
      diagnostics (inverted polarity: here "1.0 integer" plays the
      role of "0.0 non-canonical").
    - **Window-refinement sensitivity axis**: for each of three
      adjacent integer windows
      :math:`\\{(\\tau_l + j, \\tau_g + j) : j = 0, 1, 2\\}`,
      deep-copy the warmed-up template graph, fire
      :func:`apply_network_remesh` ``remesh_events_per_window``
      times, and record the final per-node EPI snapshot.  Compute
      the per-node variance across the bracket, normalise by per-
      node mean absolute EPI (scale invariance), and squash through
      :math:`\\tanh` to ``[0, 1]``.  A low signature means the three
      adjacent integer windows produce essentially identical post-
      REMESH states — the canonical integer-indexed REMESH is
      structurally smooth in :math:`\\tau` and no continuous kernel
      :math:`K(t, s)` is forced by canonical evolution.

    The diagnostic preserves the canonical implementation entirely
    (no monkey-patching, no operator modification, no parameter
    coercion bypass).  It is a *read-only probe* of canonical
    canonical TNFR evolution at three canonical integer windows.

    Empirical baseline
    ------------------
    Under canonical defaults (``τ_l = 4``, ``τ_g = 8``,
    ``α = 0.5``) on a ring graph with mild deterministic initial
    perturbation, the expected outcome is
    ``integer_storage_fraction == 1.0`` (structural) and
    ``signature ∈ [0, scalar_threshold)`` (empirical), yielding
    verdict ``"INTEGER_WINDOW_ADEQUATE"``.
    """
    epi_bracket, nodes, n_int, n_tot = _run_remesh_bracket(
        n_nodes=int(n_nodes),
        seed=int(seed),
        warmup_steps=int(warmup_steps),
        tau_l_base=int(tau_l),
        tau_g_base=int(tau_g),
        n_events=int(remesh_events_per_window),
    )
    signature, raw_var = _window_refinement_signature(epi_bracket)
    integer_storage_fraction = float(n_int) / float(n_tot) if n_tot > 0 else 0.0
    noninteger_storage_count = int(n_tot - n_int)
    n_remesh_events = 3 * int(remesh_events_per_window)

    # Advisory L2 distance: baseline vs (+1, +2) windows.
    baseline = epi_bracket[0]
    l2_p1 = float(np.linalg.norm(epi_bracket[1] - baseline)) / max(
        1, baseline.size
    )
    l2_p2 = float(np.linalg.norm(epi_bracket[2] - baseline)) / max(
        1, baseline.size
    )
    bracket_mean_l2 = 0.5 * (l2_p1 + l2_p2)

    if (
        signature < scalar_threshold
        and integer_storage_fraction >= 1.0 - 1e-12
    ):
        verdict = "INTEGER_WINDOW_ADEQUATE"
    elif (
        signature > continuous_threshold
        or integer_storage_fraction < 1.0 - 1e-12
    ):
        verdict = "CONTINUOUS_KERNEL_NECESSARY"
    else:
        verdict = "INDETERMINATE"

    diagnostics: dict[str, Any] = {
        "epi_bracket": epi_bracket.tolist(),
        "bracket_l2_per_offset": (l2_p1, l2_p2),
        "scalar_threshold": float(scalar_threshold),
        "continuous_threshold": float(continuous_threshold),
        "seed": int(seed),
        "warmup_steps": int(warmup_steps),
        "remesh_events_per_window": int(remesh_events_per_window),
        "integer_storage_count": int(n_int),
        "total_window_reads": int(n_tot),
    }

    return RemeshWindowTypeSignatureCertificate(
        signature=signature,
        integer_storage_fraction=integer_storage_fraction,
        noninteger_storage_count=noninteger_storage_count,
        n_remesh_events=n_remesh_events,
        n_nodes=int(n_nodes),
        tau_l=int(tau_l),
        tau_g=int(tau_g),
        bracket=(0, 1, 2),
        raw_relative_variance=raw_var,
        bracket_mean_l2=bracket_mean_l2,
        verdict=verdict,
        diagnostics=diagnostics,
    )
