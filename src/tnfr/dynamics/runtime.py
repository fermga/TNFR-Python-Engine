"""Runtime orchestration for TNFR dynamics."""

from __future__ import annotations

import inspect
import sys
from copy import deepcopy
from collections import deque
from collections.abc import Iterable, Mapping, MutableMapping
from numbers import Real
from typing import Any, cast

from ..alias import get_attr
from ..utils import CallbackEvent, callback_manager
from ..constants import get_graph_param, get_param
from ..glyph_history import ensure_history
from ..metrics.sense_index import compute_Si
from ..operators import apply_remesh_if_globally_stable
from ..telemetry import publish_graph_cache_metrics
from ..types import HistoryState, NodeId, TNFRGraph
from ..utils import normalize_optional_int
from ..validation import apply_canonical_clamps
from . import adaptation, coordination, integrators, selectors
from .aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_THETA, ALIAS_VF

try:  # pragma: no cover - optional NumPy dependency
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency missing
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional math extras
    from ..mathematics.dynamics import MathematicalDynamicsEngine
    from ..mathematics.projection import BasicStateProjector
    from ..mathematics.runtime import (
        coherence as runtime_coherence,
        frequency_positive as runtime_frequency_positive,
        normalized as runtime_normalized,
    )
except Exception:  # pragma: no cover - fallback when extras not available
    MathematicalDynamicsEngine = None  # type: ignore[assignment]
    BasicStateProjector = None  # type: ignore[assignment]
    runtime_coherence = None  # type: ignore[assignment]
    runtime_frequency_positive = None  # type: ignore[assignment]
    runtime_normalized = None  # type: ignore[assignment]
from .dnfr import default_compute_delta_nfr
from .sampling import update_node_sample as _update_node_sample

__all__ = (
    "ALIAS_VF",
    "ALIAS_DNFR",
    "ALIAS_EPI",
    "ALIAS_SI",
    "_normalize_job_overrides",
    "_resolve_jobs_override",
    "_prepare_dnfr",
    "_update_nodes",
    "_update_epi_hist",
    "_maybe_remesh",
    "_run_validators",
    "_run_before_callbacks",
    "_run_after_callbacks",
    "step",
    "run",
)


def _normalize_job_overrides(
    job_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Canonicalise job override keys for ΔNFR, νf and phase orchestration.

    Parameters
    ----------
    job_overrides : Mapping[str, Any] | None
        User-provided mapping whose keys may use legacy ``*_N_JOBS`` forms or
        mixed casing. The values tune the parallel workloads that update ΔNFR,
        νf adaptation and global phase coordination.

    Returns
    -------
    dict[str, Any]
        A dictionary where keys are upper-cased without the ``_N_JOBS`` suffix,
        ready for downstream lookup in the runtime schedulers.

    Notes
    -----
    ``None`` keys are silently skipped to preserve resiliency when
    orchestrating ΔNFR workers.

    Examples
    --------
    >>> _normalize_job_overrides({"dnfr_n_jobs": 2, "vf_adapt": 4})
    {'DNFR': 2, 'VF_ADAPT': 4}
    >>> _normalize_job_overrides(None)
    {}
    """
    if not job_overrides:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in job_overrides.items():
        if key is None:
            continue
        key_str = str(key).upper()
        if key_str.endswith("_N_JOBS"):
            key_str = key_str[: -len("_N_JOBS")]
        normalized[key_str] = value
    return normalized


def _resolve_jobs_override(
    overrides: Mapping[str, Any],
    key: str,
    graph_value: Any,
    *,
    allow_non_positive: bool,
) -> int | None:
    """Resolve job overrides prioritising user hints over graph defaults.

    Parameters
    ----------
    overrides : Mapping[str, Any]
        Normalised overrides produced by :func:`_normalize_job_overrides` that
        steer the ΔNFR computation, νf adaptation or phase coupling workers.
    key : str
        Logical subsystem key such as ``"DNFR"`` or ``"VF_ADAPT"``.
    graph_value : Any
        Baseline job count stored in the graph configuration.
    allow_non_positive : bool
        Propagated policy describing whether zero or negative values are valid
        for the subsystem.

    Returns
    -------
    int | None
        Final job count that each scheduler will use, or ``None`` when no
        explicit override or valid fallback exists.

    Notes
    -----
    Preference resolution is pure and returns ``None`` instead of raising when
    overrides cannot be coerced into valid integers.

    Examples
    --------
    >>> overrides = _normalize_job_overrides({"phase": 0})
    >>> _resolve_jobs_override(overrides, "phase", 2, allow_non_positive=True)
    0
    >>> _resolve_jobs_override({}, "vf_adapt", 4, allow_non_positive=False)
    4
    """
    norm_key = key.upper()
    if overrides and norm_key in overrides:
        return normalize_optional_int(
            overrides.get(norm_key),
            allow_non_positive=allow_non_positive,
            strict=False,
            sentinels=None,
        )

    return normalize_optional_int(
        graph_value,
        allow_non_positive=allow_non_positive,
        strict=False,
        sentinels=None,
    )


_INTEGRATOR_CACHE_KEY = "_integrator_cache"


def _call_integrator_factory(factory: Any, G: TNFRGraph) -> Any:
    """Invoke an integrator factory respecting optional graph injection."""

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()

    params = list(signature.parameters.values())
    required = [
        p
        for p in params
        if p.default is inspect._empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    if any(p.kind is inspect.Parameter.KEYWORD_ONLY for p in required):
        raise TypeError(
            "Integrator factory cannot require keyword-only arguments",
        )

    positional_required = [
        p
        for p in required
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional_required) > 1:
        raise TypeError(
            "Integrator factory must accept at most one positional argument",
        )

    if positional_required:
        return factory(G)

    positional = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) == 1:
        return factory(G)
    elif len(positional) > 1:
        raise TypeError(
            "Integrator factory must accept at most one positional argument",
        )

    # Check for any remaining required positional or keyword-only arguments
    remaining_required = [
        p
        for p in params
        if p.default is inspect._empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if remaining_required:
        raise TypeError(
            f"Integrator factory requires arguments: {', '.join(p.name for p in remaining_required)}"
        )
    return factory()


def _resolve_integrator_instance(G: TNFRGraph) -> integrators.AbstractIntegrator:
    """Return an integrator instance configured on ``G`` or a default."""

    cache_entry = G.graph.get(_INTEGRATOR_CACHE_KEY)
    candidate = G.graph.get("integrator")
    if (
        isinstance(cache_entry, tuple)
        and len(cache_entry) == 2
        and cache_entry[0] is candidate
        and isinstance(cache_entry[1], integrators.AbstractIntegrator)
    ):
        return cache_entry[1]

    if isinstance(candidate, integrators.AbstractIntegrator):
        instance = candidate
    elif inspect.isclass(candidate) and issubclass(candidate, integrators.AbstractIntegrator):
        instance = candidate()
    elif callable(candidate):
        instance = cast(
            integrators.AbstractIntegrator,
            _call_integrator_factory(candidate, G),
        )
    elif candidate is None:
        instance = integrators.DefaultIntegrator()
    else:
        raise TypeError(
            "Graph integrator must be an AbstractIntegrator, subclass or callable",
        )

    if not isinstance(instance, integrators.AbstractIntegrator):
        raise TypeError(
            "Configured integrator must implement AbstractIntegrator.integrate",
        )

    G.graph[_INTEGRATOR_CACHE_KEY] = (candidate, instance)
    return instance


def _run_before_callbacks(
    G: TNFRGraph,
    *,
    step_idx: int,
    dt: float | None,
    use_Si: bool,
    apply_glyphs: bool,
) -> None:
    """Notify ``BEFORE_STEP`` observers with execution context."""

    callback_manager.invoke_callbacks(
        G,
        CallbackEvent.BEFORE_STEP.value,
        {
            "step": step_idx,
            "dt": dt,
            "use_Si": use_Si,
            "apply_glyphs": apply_glyphs,
        },
    )


def _prepare_dnfr(
    G: TNFRGraph,
    *,
    use_Si: bool,
    job_overrides: Mapping[str, Any] | None = None,
) -> None:
    """Recompute ΔNFR (and optionally Si) ahead of an integration step."""

    compute_dnfr_cb = G.graph.get("compute_delta_nfr", default_compute_delta_nfr)
    overrides = job_overrides or {}
    n_jobs = _resolve_jobs_override(
        overrides,
        "DNFR",
        G.graph.get("DNFR_N_JOBS"),
        allow_non_positive=False,
    )

    supports_n_jobs = False
    try:
        signature = inspect.signature(compute_dnfr_cb)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        params = signature.parameters
        if "n_jobs" in params:
            kind = params["n_jobs"].kind
            supports_n_jobs = kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            supports_n_jobs = True

    if supports_n_jobs:
        compute_dnfr_cb(G, n_jobs=n_jobs)
    else:
        try:
            compute_dnfr_cb(G, n_jobs=n_jobs)
        except TypeError as exc:
            if "n_jobs" in str(exc):
                compute_dnfr_cb(G)
            else:
                raise
    G.graph.pop("_sel_norms", None)
    if use_Si:
        si_jobs = _resolve_jobs_override(
            overrides,
            "SI",
            G.graph.get("SI_N_JOBS"),
            allow_non_positive=False,
        )
        dynamics_module = sys.modules.get("tnfr.dynamics")
        compute_si_fn = (
            getattr(dynamics_module, "compute_Si", None) if dynamics_module is not None else None
        )
        if compute_si_fn is None:
            compute_si_fn = compute_Si
        compute_si_fn(G, inplace=True, n_jobs=si_jobs)


def _update_nodes(
    G: TNFRGraph,
    *,
    dt: float | None,
    use_Si: bool,
    apply_glyphs: bool,
    step_idx: int,
    hist: HistoryState,
    job_overrides: Mapping[str, Any] | None = None,
) -> None:
    """Apply glyphs, integrate ΔNFR and refresh derived nodal state."""

    _update_node_sample(G, step=step_idx)
    overrides = job_overrides or {}
    _prepare_dnfr(G, use_Si=use_Si, job_overrides=overrides)
    selector = selectors._apply_selector(G)
    if apply_glyphs:
        selectors._apply_glyphs(G, selector, hist)
    _dt = get_graph_param(G, "DT") if dt is None else float(dt)
    method = get_graph_param(G, "INTEGRATOR_METHOD", str)
    n_jobs = _resolve_jobs_override(
        overrides,
        "INTEGRATOR",
        G.graph.get("INTEGRATOR_N_JOBS"),
        allow_non_positive=True,
    )
    integrator = _resolve_integrator_instance(G)
    integrator.integrate(
        G,
        dt=_dt,
        t=None,
        method=cast(str | None, method),
        n_jobs=n_jobs,
    )
    for n, nd in G.nodes(data=True):
        apply_canonical_clamps(cast(MutableMapping[str, Any], nd), G, cast(NodeId, n))
    phase_jobs = _resolve_jobs_override(
        overrides,
        "PHASE",
        G.graph.get("PHASE_N_JOBS"),
        allow_non_positive=True,
    )
    coordination.coordinate_global_local_phase(G, None, None, n_jobs=phase_jobs)
    vf_jobs = _resolve_jobs_override(
        overrides,
        "VF_ADAPT",
        G.graph.get("VF_ADAPT_N_JOBS"),
        allow_non_positive=False,
    )
    adaptation.adapt_vf_by_coherence(G, n_jobs=vf_jobs)


def _update_epi_hist(G: TNFRGraph) -> None:
    """Maintain the rolling EPI history used by remeshing heuristics."""

    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    tau = max(tau_g, tau_l)
    maxlen = max(2 * tau + 5, 64)
    epi_hist = G.graph.get("_epi_hist")
    if not isinstance(epi_hist, deque) or epi_hist.maxlen != maxlen:
        epi_hist = deque(list(epi_hist or [])[-maxlen:], maxlen=maxlen)
        G.graph["_epi_hist"] = epi_hist
    epi_hist.append({n: get_attr(nd, ALIAS_EPI, 0.0) for n, nd in G.nodes(data=True)})


def _maybe_remesh(G: TNFRGraph) -> None:
    """Trigger remeshing when stability thresholds are satisfied."""

    apply_remesh_if_globally_stable(G)


def _run_validators(G: TNFRGraph) -> None:
    """Execute registered validators ensuring canonical invariants hold."""

    from ..validation import run_validators

    run_validators(G)


def _run_after_callbacks(G, *, step_idx: int) -> None:
    """Notify ``AFTER_STEP`` observers with the latest structural metrics."""

    h = ensure_history(G)
    ctx = {"step": step_idx}
    metric_pairs = [
        ("C", "C_steps"),
        ("stable_frac", "stable_frac"),
        ("phase_sync", "phase_sync"),
        ("glyph_disr", "glyph_load_disr"),
        ("Si_mean", "Si_mean"),
    ]
    for dst, src in metric_pairs:
        values = h.get(src)
        if values:
            ctx[dst] = values[-1]
    callback_manager.invoke_callbacks(G, CallbackEvent.AFTER_STEP.value, ctx)

    telemetry = G.graph.get("telemetry")
    if isinstance(telemetry, MutableMapping):
        history = telemetry.get("nu_f")
        history_key = "nu_f_history"
        if isinstance(history, list) and history_key not in telemetry:
            telemetry[history_key] = history
        payload = telemetry.get("nu_f_snapshot")
        if isinstance(payload, Mapping):
            bridge_raw = telemetry.get("nu_f_bridge")
            try:
                bridge = float(bridge_raw) if bridge_raw is not None else None
            except (TypeError, ValueError):
                bridge = None
            nu_f_summary = {
                "total_reorganisations": payload.get("total_reorganisations"),
                "total_duration": payload.get("total_duration"),
                "rate_hz_str": payload.get("rate_hz_str"),
                "rate_hz": payload.get("rate_hz"),
                "variance_hz_str": payload.get("variance_hz_str"),
                "variance_hz": payload.get("variance_hz"),
                "confidence_level": payload.get("confidence_level"),
                "ci_hz_str": {
                    "lower": payload.get("ci_lower_hz_str"),
                    "upper": payload.get("ci_upper_hz_str"),
                },
                "ci_hz": {
                    "lower": payload.get("ci_lower_hz"),
                    "upper": payload.get("ci_upper_hz"),
                },
                "bridge": bridge,
            }
            telemetry["nu_f"] = nu_f_summary
            math_summary = telemetry.get("math_engine")
            if isinstance(math_summary, MutableMapping):
                math_summary["nu_f"] = dict(nu_f_summary)


def _get_math_engine_config(G: TNFRGraph) -> MutableMapping[str, Any] | None:
    """Return the mutable math-engine configuration stored on ``G``."""

    cfg_raw = G.graph.get("MATH_ENGINE")
    if not isinstance(cfg_raw, Mapping) or not cfg_raw.get("enabled", False):
        return None
    if isinstance(cfg_raw, MutableMapping):
        return cfg_raw
    cfg_mutable: MutableMapping[str, Any] = dict(cfg_raw)
    G.graph["MATH_ENGINE"] = cfg_mutable
    return cfg_mutable


def _initialise_math_state(
    G: TNFRGraph,
    cfg: MutableMapping[str, Any],
    *,
    hilbert_space: Any,
    projector: BasicStateProjector,
) -> np.ndarray | None:
    """Project graph nodes into the Hilbert space to seed the math engine."""

    dimension = getattr(hilbert_space, "dimension", None)
    if dimension is None:
        raise AttributeError("Hilbert space configuration is missing 'dimension'.")

    vectors: list[np.ndarray] = []
    for _, nd in G.nodes(data=True):
        epi = float(get_attr(nd, ALIAS_EPI, 0.0) or 0.0)
        nu_f = float(get_attr(nd, ALIAS_VF, 0.0) or 0.0)
        theta_val = float(get_attr(nd, ALIAS_THETA, 0.0) or 0.0)
        try:
            vector = projector(epi=epi, nu_f=nu_f, theta=theta_val, dim=int(dimension))
        except ValueError:
            continue
        vectors.append(np.asarray(vector, dtype=np.complex128))

    if not vectors:
        return None

    stacked = np.vstack(vectors)
    averaged = np.mean(stacked, axis=0)
    atol = float(getattr(projector, "atol", 1e-9))
    norm = float(getattr(hilbert_space, "norm")(averaged))
    if np.isclose(norm, 0.0, atol=atol):
        averaged = vectors[0]
        norm = float(getattr(hilbert_space, "norm")(averaged))
    if np.isclose(norm, 0.0, atol=atol):
        return None
    normalised = averaged / norm
    cfg.setdefault("_state_origin", "projected")
    return normalised


def _advance_math_engine(
    G: TNFRGraph,
    *,
    dt: float,
    step_idx: int,
    hist: HistoryState,
) -> None:
    """Advance the optional math engine and record spectral telemetry."""

    cfg = _get_math_engine_config(G)
    if cfg is None:
        return

    if (
        np is None
        or MathematicalDynamicsEngine is None
        or runtime_normalized is None
        or runtime_coherence is None
    ):
        raise RuntimeError(
            "Mathematical dynamics require NumPy and tnfr.mathematics extras to be installed."
        )

    hilbert_space = cfg.get("hilbert_space")
    coherence_operator = cfg.get("coherence_operator")
    coherence_threshold = cfg.get("coherence_threshold")
    if hilbert_space is None or coherence_operator is None or coherence_threshold is None:
        raise ValueError(
            "MATH_ENGINE requires 'hilbert_space', 'coherence_operator' and "
            "'coherence_threshold' entries."
        )

    if BasicStateProjector is None:  # pragma: no cover - guarded by import above
        raise RuntimeError("Mathematical dynamics require the BasicStateProjector helper.")

    projector = cfg.get("state_projector")
    if not isinstance(projector, BasicStateProjector):
        projector = BasicStateProjector()
        cfg["state_projector"] = projector

    engine = cfg.get("dynamics_engine")
    if not isinstance(engine, MathematicalDynamicsEngine):
        generator = cfg.get("generator_matrix")
        if generator is None:
            raise ValueError(
                "MATH_ENGINE requires either a 'dynamics_engine' instance or a "
                "'generator_matrix'."
            )
        generator_matrix = np.asarray(generator, dtype=np.complex128)
        engine = MathematicalDynamicsEngine(generator_matrix, hilbert_space=hilbert_space)
        cfg["dynamics_engine"] = engine

    state_vector = cfg.get("_state_vector")
    if state_vector is None:
        state_vector = _initialise_math_state(
            G,
            cfg,
            hilbert_space=hilbert_space,
            projector=projector,
        )
        if state_vector is None:
            return
    else:
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        dimension = getattr(hilbert_space, "dimension", state_vector.shape[0])
        if state_vector.shape != (int(dimension),):
            state_vector = _initialise_math_state(
                G,
                cfg,
                hilbert_space=hilbert_space,
                projector=projector,
            )
            if state_vector is None:
                return

    advanced = engine.step(state_vector, dt=float(dt), normalize=True)
    cfg["_state_vector"] = advanced

    atol = float(cfg.get("atol", getattr(engine, "atol", 1e-9)))
    label = f"step[{step_idx}]"

    normalized_passed, norm_value = runtime_normalized(
        advanced,
        hilbert_space,
        atol=atol,
        label=label,
    )
    coherence_passed, coherence_value = runtime_coherence(
        advanced,
        coherence_operator,
        float(coherence_threshold),
        normalise=False,
        atol=atol,
        label=label,
    )

    frequency_operator = cfg.get("frequency_operator")
    frequency_summary: dict[str, Any] | None = None
    if frequency_operator is not None:
        if runtime_frequency_positive is None:  # pragma: no cover - guarded above
            raise RuntimeError("Frequency positivity checks require tnfr.mathematics extras.")
        freq_raw = runtime_frequency_positive(
            advanced,
            frequency_operator,
            normalise=False,
            enforce=True,
            atol=atol,
            label=label,
        )
        frequency_summary = {
            "passed": bool(freq_raw.get("passed", False)),
            "value": float(freq_raw.get("value", 0.0)),
            "projection_passed": bool(freq_raw.get("projection_passed", False)),
            "spectrum_psd": bool(freq_raw.get("spectrum_psd", False)),
            "enforced": bool(freq_raw.get("enforce", True)),
        }
        if "spectrum_min" in freq_raw:
            frequency_summary["spectrum_min"] = float(freq_raw.get("spectrum_min", 0.0))

    summary = {
        "step": step_idx,
        "normalized": bool(normalized_passed),
        "norm": float(norm_value),
        "coherence": {
            "passed": bool(coherence_passed),
            "value": float(coherence_value),
            "threshold": float(coherence_threshold),
        },
        "frequency": frequency_summary,
    }

    hist.setdefault("math_engine_summary", []).append(summary)
    hist.setdefault("math_engine_norm", []).append(summary["norm"])
    hist.setdefault("math_engine_normalized", []).append(summary["normalized"])
    hist.setdefault("math_engine_coherence", []).append(summary["coherence"]["value"])
    hist.setdefault("math_engine_coherence_passed", []).append(summary["coherence"]["passed"])

    if frequency_summary is None:
        hist.setdefault("math_engine_frequency", []).append(None)
        hist.setdefault("math_engine_frequency_passed", []).append(None)
        hist.setdefault("math_engine_frequency_projection_passed", []).append(None)
    else:
        hist.setdefault("math_engine_frequency", []).append(frequency_summary["value"])
        hist.setdefault("math_engine_frequency_passed", []).append(frequency_summary["passed"])
        hist.setdefault("math_engine_frequency_projection_passed", []).append(
            frequency_summary["projection_passed"]
        )

    cfg["last_summary"] = summary
    telemetry = G.graph.setdefault("telemetry", {})
    telemetry["math_engine"] = deepcopy(summary)


def step(
    G: TNFRGraph,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
    n_jobs: Mapping[str, Any] | None = None,
) -> None:
    """Advance the runtime one ΔNFR step updating νf, phase and glyphs.

    Parameters
    ----------
    G : TNFRGraph
        Graph whose nodes store EPI, νf and phase metadata. The graph must
        expose a ΔNFR hook under ``G.graph['compute_delta_nfr']`` and optional
        selector or callback registrations.
    dt : float | None, optional
        Time increment injected into the integrator. ``None`` falls back to the
        ``DT`` attribute stored in ``G.graph`` which keeps ΔNFR integration
        aligned with the nodal equation.
    use_Si : bool, default True
        When ``True`` the Sense Index (Si) is recomputed to modulate ΔNFR and
        νf adaptation heuristics.
    apply_glyphs : bool, default True
        Enables canonical glyph selection so that phase and coherence glyphs
        continue to modulate ΔNFR.
    n_jobs : Mapping[str, Any] | None, optional
        Optional overrides that tune the parallel workers used for ΔNFR, phase
        coordination and νf adaptation. The mapping is processed by
        :func:`_normalize_job_overrides`.

    Returns
    -------
    None
        Mutates ``G`` in place by recomputing ΔNFR, νf and phase metrics.

    Notes
    -----
    Registered callbacks execute within :func:`step` and any exceptions they
    raise propagate according to the callback manager configuration.

    Examples
    --------
    Register a hook that records phase synchrony while using the parametric
    selector to choose glyphs before advancing one runtime step.

    >>> from tnfr.utils import CallbackEvent, callback_manager
    >>> from tnfr.dynamics import selectors
    >>> from tnfr.dynamics.runtime import ALIAS_VF
    >>> from tnfr.structural import create_nfr
    >>> G, node = create_nfr("seed", epi=0.2, vf=1.5)
    >>> callback_manager.register_callback(
    ...     G,
    ...     CallbackEvent.AFTER_STEP,
    ...     lambda graph, ctx: graph.graph.setdefault("phase_log", []).append(ctx.get("phase_sync")),
    ... )
    >>> G.graph["glyph_selector"] = selectors.ParametricGlyphSelector()
    >>> step(G, dt=0.05, n_jobs={"dnfr_n_jobs": 1})
    >>> ALIAS_VF in G.nodes[node]
    True
    """
    job_overrides = _normalize_job_overrides(n_jobs)
    hist = ensure_history(G)
    step_idx = len(hist.setdefault("C_steps", []))
    _run_before_callbacks(G, step_idx=step_idx, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)
    _update_nodes(
        G,
        dt=dt,
        use_Si=use_Si,
        apply_glyphs=apply_glyphs,
        step_idx=step_idx,
        hist=hist,
        job_overrides=job_overrides,
    )
    resolved_dt = get_graph_param(G, "DT") if dt is None else float(dt)
    _advance_math_engine(
        G,
        dt=resolved_dt,
        step_idx=step_idx,
        hist=hist,
    )
    _update_epi_hist(G)
    _maybe_remesh(G)
    _run_validators(G)
    _run_after_callbacks(G, step_idx=step_idx)
    publish_graph_cache_metrics(G)


def run(
    G: TNFRGraph,
    steps: int,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
    n_jobs: Mapping[str, Any] | None = None,
) -> None:
    """Iterate :func:`step` to evolve ΔNFR, νf and phase trajectories.

    Parameters
    ----------
    G : TNFRGraph
        Graph that stores the coherent structures. Callbacks and selectors
        configured on ``G.graph`` orchestrate glyph application and telemetry.
    steps : int
        Number of times :func:`step` is invoked. Each iteration integrates ΔNFR
        and νf according to ``dt`` and the configured selector.
    dt : float | None, optional
        Time increment for each step. ``None`` uses the graph's default ``DT``.
    use_Si : bool, default True
        Recompute the Sense Index during each iteration to keep ΔNFR feedback
        loops tied to νf adjustments.
    apply_glyphs : bool, default True
        Enables glyph selection and application per step.
    n_jobs : Mapping[str, Any] | None, optional
        Shared overrides forwarded to each :func:`step` call.

    Returns
    -------
    None
        The graph ``G`` is updated in place.

    Raises
    ------
    ValueError
        Raised when ``steps`` is negative because the runtime cannot evolve a
        negative number of ΔNFR updates.

    Examples
    --------
    Install a before-step callback and use the default glyph selector while
    running two iterations that synchronise phase and νf.

    >>> from tnfr.utils import CallbackEvent, callback_manager
    >>> from tnfr.dynamics import selectors
    >>> from tnfr.structural import create_nfr
    >>> G, node = create_nfr("seed", epi=0.3, vf=1.2)
    >>> callback_manager.register_callback(
    ...     G,
    ...     CallbackEvent.BEFORE_STEP,
    ...     lambda graph, ctx: graph.graph.setdefault("dt_trace", []).append(ctx["dt"]),
    ... )
    >>> G.graph["glyph_selector"] = selectors.default_glyph_selector
    >>> run(G, 2, dt=0.1)
    >>> len(G.graph["dt_trace"])
    2
    """
    steps_int = int(steps)
    if steps_int < 0:
        raise ValueError("'steps' must be non-negative")
    stop_cfg = get_graph_param(G, "STOP_EARLY", dict)
    stop_enabled = False
    if stop_cfg and stop_cfg.get("enabled", False):
        w = max(1, int(stop_cfg.get("window", 25)))
        frac = float(stop_cfg.get("fraction", 0.90))
        stop_enabled = True
    job_overrides = _normalize_job_overrides(n_jobs)
    for _ in range(steps_int):
        step(
            G,
            dt=dt,
            use_Si=use_Si,
            apply_glyphs=apply_glyphs,
            n_jobs=job_overrides,
        )
        if stop_enabled:
            history = ensure_history(G)
            raw_series = dict.get(history, "stable_frac", [])
            if not isinstance(raw_series, Iterable):
                series = []
            elif isinstance(raw_series, list):
                series = raw_series
            else:
                series = list(raw_series)
            numeric_series = [v for v in series if isinstance(v, Real)]
            if len(numeric_series) >= w and all(v >= frac for v in numeric_series[-w:]):
                break
