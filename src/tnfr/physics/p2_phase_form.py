"""Explicit conditional P2 composition of existing phase and pressure kernels.

Fixed positive capacities supply the configured free angular rates. The U3
sine proposal drives phase; freshly computed canonical multichannel pressure
drives form through DefaultIntegrator. This opt-in composition is separate
from the isolated-EPI optimizers and the independent-pressure extension.
It derives neither a clock nor the selection of capacity/support from EPI.

The real-model theorem is in theory/FORCED_SUPPORT_BALANCE.md section 26.
Numerical lock targets are estimates; finite detached steps retain their
pressure and Euler defects and do not certify infinite binary64 stability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction

import networkx as nx

from .._exact_time import finite_represented_real
from ..alias import set_theta
from ..config import DEFAULTS
from ..constants.operational import NODAL_OPT_COUPLING_CANONICAL
from ..dynamics._euler_kernel import euler_update
from ..mathematics.unified_numerical import np
from ..utils import angle_diff
from ._cycle_algebra import Vector
from .forcing_realization import (
    NonEpiForcingObservation,
    _runtime_weights,
    capture_non_epi_forcing,
)
from .structural_diffusion import _fraction_sqrt_upper_float
from .support_transport import (
    SupportTransportEuler,
    observe_support_transport,
    observe_support_transport_euler,
)

__all__ = [
    "P2PhaseFormModel",
    "P2PhaseFormStep",
    "derive_p2_phase_form_model",
    "propose_p2_phase_form_step",
]

_CHANNELS = ("phase", "epi", "vf", "topo")
_SCOPE = (
    "conditional fixed unit P2; supplied constant capacities and omega=nu "
    "sine-coupling phase law; fresh canonical pressure; no substrate-generation "
    "or infinite binary64 stability claim"
)


def _pair(values, label) -> Vector:
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered pair")
    items = tuple(values)
    if len(items) != 2:
        raise ValueError(f"{label} must have two coordinates")
    return tuple(finite_represented_real(item, label)[1] for item in items)


@dataclass(frozen=True)
class P2PhaseFormModel:
    """Represented coefficients and predictions for the ideal half-pi chart.

    The strict-lock classification compares represented d and K exactly for
    the ideal real gate pi/2. Trigonometric targets use binary64 math, not
    transcendental interval certificates. ``phase_decay_rate_estimate`` is
    the least binary64 upper enclosure of sqrt((2K-d)(2K+d)), not a lower
    contraction bound. Its rational radicand avoids intermediate overflow and
    underflow near the locking boundary.
    Each step separately checks the production's represented strict gate.
    Configured weights are the recipe; normalized weights are the actual
    shared reader's coefficients. No measured EPI derivative sets pressure.
    """

    capacities: Vector
    coupling_strength: Fraction
    configured_weights: tuple[tuple[str, Fraction], ...]
    normalized_weights: tuple[tuple[str, Fraction], ...]
    locking_ratio: Fraction | None
    lock_status: str
    locked_phase_estimate: float | None
    locked_contrast_estimate: float | None
    phase_decay_rate_estimate: float | None
    form_decay_rate: Fraction
    monotone_step_ceiling: Fraction
    scope: str


def derive_p2_phase_form_model(
    capacities,
    *,
    pressure_weights=None,
    coupling_strength=NODAL_OPT_COUPLING_CANONICAL,
) -> P2PhaseFormModel:
    """Declare one composition and predict its relative lock before execution.

    Capacities and all coefficients are finite binary64 materializations.
    Positive EPI weight is required; zero phase/capacity/topology weights and
    K=0 are valid controls. Equal capacities with K=0 give a neutral family of
    constant phase gaps, not a unique attracting lock. Topology pressure is
    zero on P2 but its weight still participates in the standard normalization.
    """
    nu = _pair(capacities, "capacities")
    if any(value <= 0 for value in nu):
        raise ValueError("capacities must be positive")
    _, coupling = finite_represented_real(coupling_strength, "coupling_strength")
    if coupling < 0:
        raise ValueError("coupling_strength must be nonnegative")
    raw = DEFAULTS["DNFR_WEIGHTS"] if pressure_weights is None else pressure_weights
    if not isinstance(raw, Mapping) or set(raw) - set(_CHANNELS):
        raise ValueError("pressure_weights must map phase/epi/vf/topo channels")
    configured = tuple(
        (name, finite_represented_real(raw.get(name, 0.0), f"{name} weight")[1])
        for name in _CHANNELS
    )
    if any(value < 0 for _, value in configured) or not any(
        value for _, value in configured
    ):
        raise ValueError("pressure weights must be nonnegative with a positive sum")
    weight_graph = nx.Graph()
    weight_graph.graph["DNFR_WEIGHTS"] = {
        name: float(value) for name, value in configured
    }
    weights = _runtime_weights(weight_graph)
    effective = dict(weights)
    e, w, v = (effective[name] for name in ("epi", "phase", "vf"))
    if e <= 0:
        raise ValueError("effective EPI weight must be positive")
    sigma, difference = sum(nu), nu[1] - nu[0]
    rate = e * sigma
    finite_represented_real(rate, "form decay rate")
    ratio = difference / (2 * coupling) if coupling else None
    phase_target = contrast_target = phase_rate = None
    status = (
        "neutral_phase_family"
        if coupling == 0 and difference == 0
        else "no_strict_lock"
    )
    if ratio is not None and abs(ratio) < 1:
        status = "strict_attracting"
        phase_target = math.asin(float(ratio))
        # Combine represented target terms before materializing their ratio;
        # an intermediate w/e can overflow even when the final contrast is 0.
        target = (
            w * Fraction.from_float(phase_target) / Fraction.from_float(math.pi)
            + v * difference
        ) / e
        try:
            contrast_target = float(target)
        except OverflowError as exc:
            raise ValueError("lock estimates must remain finite") from exc
        phase_rate = _fraction_sqrt_upper_float(
            (2 * coupling - difference) * (2 * coupling + difference)
        )
        if not all(
            math.isfinite(value)
            for value in (phase_target, contrast_target, phase_rate)
        ):
            raise ValueError("lock estimates must remain finite")
    ceiling = min(1 / rate, 1 / (2 * coupling)) if coupling else 1 / rate
    return P2PhaseFormModel(
        capacities=nu,
        coupling_strength=coupling,
        configured_weights=configured,
        normalized_weights=weights,
        locking_ratio=ratio,
        lock_status=status,
        locked_phase_estimate=phase_target,
        locked_contrast_estimate=contrast_target,
        phase_decay_rate_estimate=phase_rate,
        form_decay_rate=rate,
        monotone_step_ceiling=ceiling,
        scope=_SCOPE,
    )


@dataclass(frozen=True)
class P2PhaseFormStep:
    """One detached shared-kernel step, including measured numerical defects.

    ``pressure_before`` is refreshed and consumed by this step;
    ``pressure_after`` is freshly evaluated at both new form and new phase.
    ``forcing_before`` uses represented nonlinear channel values, keeping its
    kernel defect separate from the exact held-rate Euler budget. Its stored
    residual compares fresh pressure to the fixture's initial zero buffer.
    No caller graph, operator history or external runtime invocation is sealed.
    """

    model: P2PhaseFormModel
    dt: Fraction
    before_epi: Vector
    after_epi: Vector
    before_phase: Vector
    after_phase: Vector
    pressure_before: Vector
    pressure_after: Vector
    forcing_before: NonEpiForcingObservation
    euler: SupportTransportEuler
    before_contrast: Fraction
    after_contrast: Fraction
    before_phase_gap: float
    after_phase_gap: float
    weighted_mean_drift: Fraction
    invocation_owners: tuple[str, ...]
    scope: str


def propose_p2_phase_form_step(model, epi, phases, *, dt) -> P2PhaseFormStep:
    """Execute one opt-in composition on a newly owned detached P2 fixture.

    All phase/pressure inputs come from the same initial state. DefaultIntegrator
    executes one Euler segment (DT_MIN=0, Gamma disabled); the shared phase
    proposal is committed only afterwards, then endpoint pressure is refreshed.
    Only canonical [0,2*pi) phase coordinates strictly within the represented
    half-pi gate are accepted, before and after. EPI and its unclipped proposal
    must lie in [-1,1]. The sufficient monotone timestep restriction is checked
    exactly on represented coefficients; it is not a global runtime certificate.
    """
    from ..dynamics.dnfr import default_compute_delta_nfr
    from ..dynamics.integrators import DefaultIntegrator, prepare_integration_params
    from ..dynamics.phase_evolution import propose_u3_gated_phase_step

    if type(model) is not P2PhaseFormModel:
        raise TypeError("model must be a P2PhaseFormModel")
    rebuilt = derive_p2_phase_form_model(
        model.capacities,
        pressure_weights=dict(model.configured_weights),
        coupling_strength=model.coupling_strength,
    )
    if model != rebuilt:
        raise ValueError("model differs from its declared recipe")
    # Admission must use reconstructed coefficients even if a caller supplied
    # a derived field with custom comparison behavior.
    model = rebuilt
    before_epi, before_phase = _pair(epi, "EPI"), _pair(phases, "phases")
    h_float, h = finite_represented_real(dt, "dt")
    if not 0 < h <= model.monotone_step_ceiling:
        raise ValueError("dt must be positive and satisfy the monotone step ceiling")
    if any(not -1 <= value <= 1 for value in before_epi):
        raise ValueError("initial EPI must lie in [-1,1]")
    if any(not 0 <= value < Fraction.from_float(math.tau) for value in before_phase):
        raise ValueError("phases must lie in the canonical [0,2*pi) chart")
    phase = tuple(map(float, before_phase))
    gap = angle_diff(phase[1], phase[0])
    if abs(gap) >= math.pi / 2:
        raise ValueError("initial phase gap must be strictly inside the half-pi gate")
    graph = nx.path_graph(2)
    graph.edges[0, 1].update(weight=1.0, length=1.0)
    graph.graph.update(
        DNFR_WEIGHTS={name: float(value) for name, value in model.configured_weights},
        DELTA_PHI_MAX=math.pi / 2,
        UM_MAX_PHASE_DIFF=math.pi / 2,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        vectorized_dnfr=True,
        _t=0.0,
    )
    for i in graph:
        graph.nodes[i].update(
            EPI=float(before_epi[i]),
            nu_f=float(model.capacities[i]),
            theta=phase[i],
            delta_nfr=0.0,
            dEPI=0.0,
        )
    if prepare_integration_params(graph, h_float, 0.0, "euler") != (
        h_float,
        1,
        0.0,
        "euler",
    ):
        raise RuntimeError("P2 composition requires one unsplit Euler segment")
    forcing = capture_non_epi_forcing(graph)
    if forcing.normalized_weights != model.normalized_weights:
        raise RuntimeError("pressure coefficients differ from the model recipe")
    phase_proposal = propose_u3_gated_phase_step(
        graph,
        (0, 1),
        phase,
        tuple(map(float, model.capacities)),
        dt=h_float,
        coupling_strength=float(model.coupling_strength),
    )
    new_gap = angle_diff(float(phase_proposal[1]), float(phase_proposal[0]))
    if abs(new_gap) >= math.pi / 2:
        raise ValueError("proposed phase gap leaves the strict half-pi gate")
    default_compute_delta_nfr(graph)
    before = observe_support_transport(graph)
    if before.stored_pressure != forcing.full_kernel_pressure:
        raise RuntimeError("refreshed pressure differs from captured production kernel")
    # Preflight the same arithmetic owner used by the NumPy integrator. This
    # is an unclipped proposal check, not a second numerical evolution method.
    raw_epi = euler_update(
        np.asarray(tuple(map(float, before_epi))),
        h_float,
        np.asarray(tuple(map(float, model.capacities)))
        * np.asarray(tuple(map(float, before.stored_pressure))),
    )
    if (
        not np.all(np.isfinite(raw_epi))
        or np.any(raw_epi < -1.0)
        or np.any(raw_epi > 1.0)
    ):
        raise ValueError("P2 proposal would require EPI clipping")
    DefaultIntegrator().integrate(graph, dt=h_float, t=0.0, method="euler", n_jobs=1)
    after = observe_support_transport(graph)
    if tuple(map(float, after.epi)) != tuple(map(float, raw_epi)):
        raise RuntimeError(
            "integrated endpoint differs from the unclipped Euler proposal"
        )
    budget = observe_support_transport_euler(before, after, h)
    for i in graph:
        set_theta(graph, i, float(phase_proposal[i]))
    after_phase = _pair((graph.nodes[i]["theta"] for i in graph), "endpoint phases")
    default_compute_delta_nfr(graph)
    endpoint = observe_support_transport(graph)
    nu0, nu1 = model.capacities
    mean_drift = (
        nu1 * (after.epi[0] - before.epi[0]) + nu0 * (after.epi[1] - before.epi[1])
    ) / (nu0 + nu1)
    return P2PhaseFormStep(
        model=model,
        dt=h,
        before_epi=before.epi,
        after_epi=after.epi,
        before_phase=before_phase,
        after_phase=after_phase,
        pressure_before=before.stored_pressure,
        pressure_after=endpoint.stored_pressure,
        forcing_before=forcing,
        euler=budget,
        before_contrast=before.epi[0] - before.epi[1],
        after_contrast=after.epi[0] - after.epi[1],
        before_phase_gap=gap,
        after_phase_gap=new_gap,
        weighted_mean_drift=mean_drift,
        invocation_owners=(
            "propose_u3_gated_phase_step",
            "default_compute_delta_nfr",
            "DefaultIntegrator.integrate:euler",
            "default_compute_delta_nfr:endpoint",
        ),
        scope=_SCOPE
        + "; detached single-step observation with separate represented defects",
    )
