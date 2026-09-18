"""Data-agnostic confrontation of a real signal with canonical TNFR magnitudes.

The empirical arm as a first-class engine capability: point :func:`confront_signal`
at **any** real multichannel signal (EEG, grid telemetry, coupled oscillators) and
read the TNFR magnitudes -- the emergent phase-locking graph, the structural
tetrad (``|∇φ|``, ``K_φ``, ``Φ_s``, ``ξ_C``), the pulse (``ω_k = √λ_k``), the
static pressure-coherence snapshot ``C_static`` and the Kuramoto order ``R`` --
plus a descriptive modal AR(2) diagnostic, not a physical regime certificate.
The signal
window supplies no TNFR EPI-rate channel, so ``C_static`` evaluates the shared
coherence map with the explicit assumption ``dEPI = 0``; it is not dynamic total
``C(t)`` or an attractor certificate.  Bring your own signal; **no bundled data,
no ML dependencies**.

This composes the existing engine pipeline
(:mod:`tnfr.validation.multichannel_interface` for the phase-locking graph and
tetrad, :mod:`tnfr.physics.structural_diffusion` for the pulse and the face
certificate, :func:`tnfr.metrics.common.structural_coherence` for the shared
scalar map) into a single confrontation entry point.

The PLV graph and amplitude-pressure proxy are observational constructions,
not admitted canonical wiring or a physical state map. All these descriptors
use the full input window. The legacy nodal skill also fits and evaluates
that same window; independent calibration and reserved forecasts live in
:mod:`tnfr.validation.nodal_prediction`. Neither API by itself supplies a
physical measurement admission. Historical context is recorded separately in
``docs/EMPIRICAL_CONFRONTATION_EEG.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..metrics.common import is_structural_equilibrium, structural_coherence
from ..physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
from ..physics.structural_diffusion import (
    compute_emergent_pulse,
    structural_diffusion_operator,
    symmetric_normalized_laplacian,
)
from .multichannel_interface import (
    build_coupling_graph,
    kuramoto_order_parameter,
    phase_amplitude_matrices,
)

__all__ = [
    "SignalConfrontation",
    "ModalRootDiagnostic",
    "diagnose_modal_roots",
    "confront_signal",
    "estimate_quality_factor",
    "emergent_wave_fraction",
    "NodalPredictionSkill",
    "nodal_prediction_skill",
]


def _mean_abs(values: Any) -> float:
    vals = list(values.values())
    return float(np.mean([abs(v) for v in vals])) if vals else float("nan")


def _signal_array(signals: Any, *, min_channels: int = 3) -> np.ndarray:
    """Validate real finite observations before any statistical transform."""
    if np.iscomplexobj(signals):
        raise ValueError("signals must be real, not complex")
    data = np.asarray(signals, dtype=float)
    if data.ndim != 2 or data.shape[0] < min_channels:
        raise ValueError(
            "signals must be (n_channels, n_samples) with "
            f">= {min_channels} channels"
        )
    if data.shape[1] == 0:
        raise ValueError("signals must contain time samples")
    if not np.all(np.isfinite(data)):
        raise ValueError("signals must contain only finite observations")
    return data


def estimate_quality_factor(signals: Any) -> float:
    """Dimensionless quality factor ``Q`` of a multichannel signal (fs-free).

    ``Q = k_peak / FWHM`` of the dominant spectral peak, measured in FFT-bin
    units so it needs **no** sampling rate.  For the damped substrate wave
    ``q̈ + γq̇ + Lq = 0`` a single mode has ``Q = ω₀/γ`` and ``Q = 1/2`` is
    critical damping in that specified model. This is a **secondary spectral
    read-out** only; measured Q does not identify that model or its damping.
    Aggregated as the median across channels. Zero denotes no measured peak
    (including a constant or short window), not a diffusion certificate.
    """
    data = _signal_array(signals, min_channels=1)
    if data.shape[1] < 8:
        return 0.0
    qs: list[float] = []
    for row in data:
        scale = float(np.max(np.abs(row)))
        x = row / scale if scale else row
        x = x - float(np.mean(x))
        power = np.abs(np.fft.rfft(x)) ** 2
        if power.size < 3:
            continue
        power[0] = 0.0  # drop DC
        k_peak = int(np.argmax(power))
        peak = float(power[k_peak])
        if k_peak < 1 or peak <= 0.0:
            continue
        half = peak / 2.0
        lo = k_peak
        while lo > 0 and power[lo] >= half:
            lo -= 1
        hi = k_peak
        while hi < power.size - 1 and power[hi] >= half:
            hi += 1
        qs.append(k_peak / max(hi - lo, 1))
    return float(np.median(qs)) if qs else 0.0


@dataclass(frozen=True)
class ModalRootDiagnostic:
    """Retrospective AR(2) root statistics, with explicit abstention.

    Each centered mode fits ``a[t]=b+phi1*a[t-1]+phi2*a[t-2]``. The intercept
    avoids manufacturing damping from the mean of an incomplete period; it
    is a statistical nuisance term, not a TNFR source or pressure law.
    ``resolved`` means the selected numerical fits were identified; it does
    not certify diffusion, a conservative wave, physical stability or a TNFR
    pressure law. ``stability`` describes the fitted discrete roots only.
    Eigenvalues below 1e-9 and relative modal variance below 1e-20 are excluded;
    the root/discriminant tolerance is 1e-8. These are diagnostic policies.
    """

    status: str
    reason: str
    complex_root_fraction: float | None = None
    root_classification: str = "unresolved"
    stability: str = "unresolved"
    fitted_modes: int = 0
    unresolved_modes: int = 0
    growing_modes: int = 0
    decaying_modes: int = 0
    boundary_modes: int = 0

    @property
    def legacy_diffusive_face(self) -> bool | None:
        """Deprecated root-majority alias; never a physical certificate.

        Growing fits abstain even when their roots are real. Callers must
        handle None explicitly and read ``root_classification`` instead.
        """
        if self.status != "resolved" or self.growing_modes:
            return None
        if self.complex_root_fraction is None:
            return None
        return self.complex_root_fraction <= 0.5


def _fit_modal_roots(graph: Any, data: np.ndarray) -> ModalRootDiagnostic:
    """Fit modes on one graph; failures are handled by its diagnostic caller."""
    if data.shape[1] < 8:
        return ModalRootDiagnostic("unresolved", "fewer than 8 time samples")
    nodes, lsym = symmetric_normalized_laplacian(graph)
    # Graph nodes are channel identities, not an implicit positional fallback.
    if len(nodes) != data.shape[0] or set(nodes) != set(range(data.shape[0])):
        raise ValueError("graph nodes must equal signal channel indices")
    x = data[np.asarray(nodes, dtype=int)]
    scale = float(np.max(np.abs(x)))
    if scale == 0.0:
        return ModalRootDiagnostic("unresolved", "constant signal")
    x = x / scale  # protects finite inputs from variance/AR-product overflow
    x = x - x.mean(axis=1, keepdims=True)
    if not np.any(x):
        return ModalRootDiagnostic("unresolved", "constant signal")
    w, vecs = np.linalg.eigh(lsym)
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(vecs)):
        raise ValueError("nonfinite graph eigensystem")
    modal = vecs.T @ x
    total_energy = float(np.sum(x * x) / x.shape[1])
    num = den = 0.0
    fitted = unresolved = growing = decaying = boundary = 0
    reasons: list[str] = []
    tolerance = 1e-8
    for k, eigenvalue in enumerate(w):
        if eigenvalue < 1e-9:
            continue
        a = modal[k]
        energy = float(np.mean(a * a))
        if energy <= total_energy * 1e-20:
            continue
        xd = np.column_stack([a[1:-1], a[:-2], np.ones(a.size - 2)])
        parameters, _, rank, _ = np.linalg.lstsq(xd, a[2:], rcond=None)
        if not np.all(np.isfinite(parameters)):
            raise ValueError("nonfinite AR(2) coefficients")
        phi = parameters[:2]
        discriminant = float(phi[0] ** 2 + 4.0 * phi[1])
        discriminant_scale = max(1.0, float(phi[0] ** 2), abs(4 * phi[1]))
        if rank != 3 or abs(discriminant) <= tolerance * discriminant_scale:
            unresolved += 1
            reasons.append(f"mode {k}: rank-deficient or repeated-root fit")
            continue
        roots = np.roots([1.0, -phi[0], -phi[1]])
        if not np.all(np.isfinite(roots)):
            raise ValueError("nonfinite AR(2) roots")
        radius = float(np.max(np.abs(roots)))
        growing += int(radius > 1.0 + tolerance)
        decaying += int(radius < 1.0 - tolerance)
        boundary += int(abs(radius - 1.0) <= tolerance)
        fitted += 1
        den += energy
        if discriminant < 0.0:
            num += energy
    counts = dict(
        fitted_modes=fitted, unresolved_modes=unresolved,
        growing_modes=growing, decaying_modes=decaying,
        boundary_modes=boundary,
    )
    if unresolved:
        return ModalRootDiagnostic("unresolved", "; ".join(reasons), **counts)
    if not fitted:
        return ModalRootDiagnostic(
            "unresolved", "no energetic nontrivial graph modes", **counts
        )
    fraction = num / den
    return ModalRootDiagnostic(
        status="resolved", reason="descriptive affine AR(2) fits only",
        complex_root_fraction=fraction,
        root_classification=(
            "complex_dominated" if fraction > 0.5 else "real_dominated"
        ),
        stability=("growing" if growing else "unit_boundary" if boundary
                   else "decaying"),
        **counts,
    )


def _modal_roots_from_graph(graph: Any, data: np.ndarray) -> ModalRootDiagnostic:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _fit_modal_roots(graph, data)
    except Exception as error:
        return ModalRootDiagnostic(
            "failure", f"{type(error).__name__}: {error}"
        )


def diagnose_modal_roots(
    signals: Any, *, k_neighbours: int = 4
) -> ModalRootDiagnostic:
    """Read graph-mode AR roots without equating them with physical regimes.

    Malformed/nonfinite observations raise ValueError. Short, constant or
    degenerate valid windows return unresolved; computational errors return
    failure with a reason. The graph and fits use the entire supplied window.
    """
    data = _signal_array(signals)
    if data.shape[1] < 8:
        return ModalRootDiagnostic("unresolved", "fewer than 8 time samples")
    try:
        phase, amp = phase_amplitude_matrices(data)
        graph = build_coupling_graph(phase, amp, k_neighbours=k_neighbours)
    except Exception as error:
        return ModalRootDiagnostic(
            "failure", f"{type(error).__name__}: {error}"
        )
    return _modal_roots_from_graph(graph, data)


def _wave_fraction_from_graph(graph: Any, data: np.ndarray) -> float | None:
    """Compatibility scalar; None means the modal diagnostic is unavailable."""
    return _modal_roots_from_graph(graph, data).complex_root_fraction


def emergent_wave_fraction(
    signals: Any, *, k_neighbours: int = 4
) -> float | None:
    """Legacy name for the descriptive complex-root energy fraction.

    Returns None on failure or unresolved fits, never a fabricated zero.
    A real root need not decay; a complex root need not conserve energy.
    Use :func:`diagnose_modal_roots` for the status, reason and root stability.
    """
    return diagnose_modal_roots(
        signals, k_neighbours=k_neighbours
    ).complex_root_fraction


@dataclass(frozen=True)
class SignalConfrontation:
    """TNFR read-out of one multichannel signal window.

    ``coherence`` is the static pressure-only value obtained with ``dEPI = 0``.
    ``at_equilibrium`` tests only whether its mean pressure magnitude lies
    within the selected tolerance; neither field establishes dynamic
    convergence of the underlying signal.

    ``wave_fraction`` is a legacy name for a complex-root energy fraction.
    ``diffusive_face_valid`` is a deprecated descriptive majority alias, not
    a diffusion certificate; None means unavailable or growing fitted roots.
    Use ``modal_diagnostic`` for the actual classification and its scope.
    """

    n_channels: int
    n_samples: int
    kuramoto_R: float
    grad_phi: float
    k_phi: float
    phi_s: float
    xi_c: float
    coherence: float
    at_equilibrium: bool
    pulse_fundamental: float
    dominant_beat: float
    vibration_energy: float
    quality_factor: float
    wave_fraction: float | None
    diffusive_face_valid: bool | None
    modal_diagnostic: ModalRootDiagnostic | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready report, retaining null abstentions and nonfinite fields.

        Infinite spectral length is encoded as null and its field name is
        listed explicitly; strict JSON need not manufacture a finite value.
        """
        result = asdict(self)
        nonfinite = [
            name for name, value in result.items()
            if isinstance(value, float) and not np.isfinite(value)
        ]
        for name in nonfinite:
            result[name] = None
        result["nonfinite_readouts"] = nonfinite
        return result

    def summary(self) -> str:
        """Return a one-line summary with the static score labelled explicitly."""
        eq = (
            "mean |ΔNFR| within tolerance"
            if self.at_equilibrium
            else "mean |ΔNFR| outside tolerance"
        )
        modal = self.modal_diagnostic
        if modal is None:
            diagnosis = "unresolved (legacy report lacks modal provenance)"
        else:
            diagnosis = (
                f"{modal.status}: {modal.root_classification}, "
                f"fitted stability={modal.stability} ({modal.reason})"
            )
        fraction = (
            "unavailable" if self.wave_fraction is None
            else f"{self.wave_fraction:.2f}"
        )
        return (
            f"SignalConfrontation[{self.n_channels}ch × {self.n_samples}]: "
            f"R={self.kuramoto_R:.3f}, C_static={self.coherence:.3f} ({eq}); "
            f"tetrad |∇φ|={self.grad_phi:.3f} |K_φ|={self.k_phi:.3f} "
            f"Φ_s={self.phi_s:.3f} ξ_C={self.xi_c:.3f}; pulse "
            f"ω₀={self.pulse_fundamental:.3f} beat={self.dominant_beat:.3f}; "
            f"modal={diagnosis}; complex_root_fraction={fraction}, "
            f"Q={self.quality_factor:.2f} (descriptive, not a regime certificate)"
        )


def confront_signal(
    signals: Any, *, k_neighbours: int = 4
) -> SignalConfrontation:
    """Confront a real multichannel signal with scoped TNFR magnitudes.

    The returned ``coherence`` is a static pressure snapshot with ``dEPI = 0``
    because one signal window does not materialize a TNFR EPI-rate channel.

    Parameters
    ----------
    signals : array-like, shape ``(n_channels, n_samples)``
        Any real multichannel signal (EEG, telemetry, coupled oscillators).
    k_neighbours : int
        Neighbours per channel in the emergent phase-locking coupling graph.

    Returns
    -------
    SignalConfrontation
        Scoped read-outs and a retrospective modal-root diagnostic.
    """
    data = _signal_array(signals)
    n_channels, n_samples = data.shape

    phase, amp = phase_amplitude_matrices(data)
    graph = build_coupling_graph(phase, amp, k_neighbours=k_neighbours)

    r_order = kuramoto_order_parameter(phase)
    grad = _mean_abs(compute_phase_gradient(graph))
    kphi = _mean_abs(compute_phase_curvature(graph))
    phis = _mean_abs(compute_structural_potential(graph))

    dnfr_vals = [abs(float(graph.nodes[n].get("dnfr", 0.0))) for n in graph]
    mean_dnfr = float(np.mean(dnfr_vals)) if dnfr_vals else 0.0
    coherence = structural_coherence(mean_dnfr, 0.0)
    at_eq = is_structural_equilibrium(mean_dnfr, 0.0, eps_dnfr=1e-2)

    pulse = compute_emergent_pulse(graph)
    omega0 = float(pulse["fundamental"])
    # Topology-only 1/sqrt(lambda_2) fallback. The state-dependent
    # autocorrelation fit is unavailable on these coupling graphs, so retain
    # the spectral provenance rather than claiming an exact fit identity.
    xi_c = (1.0 / omega0) if omega0 > 0.0 else float("inf")

    # Numerical AR roots describe this fitted window, not a nodal-law proof.
    q_factor = estimate_quality_factor(data)
    modal = _modal_roots_from_graph(graph, data)

    return SignalConfrontation(
        n_channels=int(n_channels),
        n_samples=int(n_samples),
        kuramoto_R=float(r_order),
        grad_phi=grad,
        k_phi=kphi,
        phi_s=phis,
        xi_c=xi_c,
        coherence=float(coherence),
        at_equilibrium=bool(at_eq),
        pulse_fundamental=float(pulse["fundamental"]),
        dominant_beat=float(pulse["dominant_beat"]),
        vibration_energy=float(pulse["vibration_energy"]),
        quality_factor=float(q_factor),
        wave_fraction=modal.complex_root_fraction,
        diffusive_face_valid=modal.legacy_diffusive_face,
        modal_diagnostic=modal,
    )


@dataclass(frozen=True)
class NodalPredictionSkill:
    """Legacy same-window descriptive fit, not held-out predictive evidence.

    Confronts the **evolution law** (not just the static read-outs): the EPI
    channel of ``dEPI/dt = ν_f·ΔNFR`` is graph diffusion, so the one-step
    predictor is ``x̂(t+1) = x(t) − c·L_rw·x(t)`` with a single diffusion step
    ``c = ν_f·dt``. Graph, centering, scales and coefficients all use the
    scored window. Unconstrained least squares guarantees nonnegative nodal
    skill up to roundoff; positive skill is not evidence of forecasting skill.
    Negative c is outside the positive-capacity diffusion model and is retained
    descriptively, without clipping. A per-channel AR-1 is fitted in-window too.
    """

    n_channels: int
    n_samples: int
    diffusivity: float  # fitted c = ν_f·dt (the structural diffusion step)
    nodal_skill: float  # 1 − MSE(nodal) / MSE(persistence)
    ar1_skill: float    # 1 − MSE(per-channel AR-1) / MSE(persistence)
    evaluation_scope: str = "same_window_descriptive_fit"

    @property
    def capacity_domain(self) -> str:
        """Sign admissibility only; positive fit is not physical calibration."""
        if not np.isfinite(self.diffusivity):
            return "unresolved"
        if self.diffusivity == 0.0:
            return "inactive_boundary"
        return "positive" if self.diffusivity > 0.0 else "negative_outside_model"

    @property
    def beats_persistence(self) -> bool:
        """Whether the in-window residual improves on persistence (no holdout)."""
        return self.nodal_skill > 0.0

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        return (
            f"NodalPredictionSkill[{self.n_channels}ch × {self.n_samples}]: "
            f"nodal_skill={self.nodal_skill:+.3f} vs AR-1 "
            f"{self.ar1_skill:+.3f} (c=ν_f·dt={self.diffusivity:+.3f}); "
            f"in-window improvement={self.beats_persistence}; "
            f"capacity_domain={self.capacity_domain}; "
            f"scope={self.evaluation_scope}, not held-out forecasting"
        )


def _nodal_skill_from_graph(
    graph: Any, data: "np.ndarray"
) -> tuple[float, float, float]:
    """Same-window descriptive (c, nodal_skill, ar1_skill) on a fixed graph."""
    nodes, lrw = structural_diffusion_operator(graph)
    try:
        idx = np.asarray([int(nd) for nd in nodes])
        x = np.asarray(data, dtype=float)[idx]
    except (ValueError, IndexError, TypeError):
        x = np.asarray(data, dtype=float)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    xc = (x - mu) / sd
    r = xc[:, 1:] - xc[:, :-1]           # actual one-step increment
    d = -(np.asarray(lrw, dtype=float) @ xc[:, :-1])  # diffusion direction
    den = float(np.sum(d * d))
    c = float(np.sum(r * d) / den) if den > 0.0 else 0.0
    mse_persist = float(np.mean(r * r))
    if mse_persist <= 0.0:
        return c, 0.0, 0.0
    mse_nodal = float(np.mean((r - c * d) ** 2))
    ar_res = []
    for i in range(xc.shape[0]):
        xi, xn = xc[i, :-1], xc[i, 1:]
        denom = float(np.dot(xi, xi))
        a = float(np.dot(xn, xi) / denom) if denom > 0.0 else 0.0
        ar_res.append(xn - a * xi)
    mse_ar = float(np.mean(np.concatenate(ar_res) ** 2))
    return c, 1.0 - mse_nodal / mse_persist, 1.0 - mse_ar / mse_persist


def nodal_prediction_skill(
    signals: Any, *, k_neighbours: int = 4
) -> NodalPredictionSkill:
    """Fit a legacy descriptive nodal model and score the same signal window.

    The EPI channel of ``dEPI/dt = ν_f·ΔNFR`` is graph diffusion
    ``dEPI/dt = −ν_f·L_rw·EPI``, so its one-step (Euler) predictor on the
    emergent coupling graph is ``x̂(t+1) = x(t) − c·L_rw·x(t)`` with a single
    diffusion step ``c = ν_f·dt`` fitted by least squares.  Returns the fraction
    of one-step increment variance explained beyond persistence, and the
    per-channel AR-1 baseline. All preprocessing and fits use the scored data;
    positive skill is algebraically expected, not held-out evidence. This API
    retains its unconstrained fitted coefficient for compatibility and cannot
    admit a positive-capacity physical model or establish causal forecasts.
    """
    data = _signal_array(signals)
    if data.shape[1] < 8:
        raise ValueError("signals must have >= 8 time samples")
    phase, amp = phase_amplitude_matrices(data)
    graph = build_coupling_graph(phase, amp, k_neighbours=k_neighbours)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        c, nodal, ar1 = _nodal_skill_from_graph(graph, data)
    if not np.all(np.isfinite((c, nodal, ar1))):
        raise ValueError("same-window descriptive fit is nonfinite")
    return NodalPredictionSkill(
        n_channels=int(data.shape[0]),
        n_samples=int(data.shape[1]),
        diffusivity=float(c),
        nodal_skill=float(nodal),
        ar1_skill=float(ar1),
    )
