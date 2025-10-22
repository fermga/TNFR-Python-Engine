"""Runtime orchestration for TNFR dynamics."""

from __future__ import annotations

import inspect
import math
import sys
from collections import deque
from collections.abc import Mapping, MutableMapping, MutableSequence
from typing import Any, cast

from ..alias import (
    get_attr,
    get_theta_attr,
    set_attr,
    set_theta,
    set_theta_attr,
    set_vf,
    multi_recompute_abs_max,
)
from ..callback_utils import CallbackEvent, callback_manager
from ..constants import DEFAULTS, get_graph_param, get_param
from ..glyph_history import ensure_history
from ..helpers.numeric import clamp
from ..metrics.sense_index import compute_Si
from ..types import HistoryState, NodeId, TNFRGraph
from .aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_VF
from . import adaptation, coordination, integrators, selectors
from .dnfr import default_compute_delta_nfr
from .sampling import update_node_sample as _update_node_sample
from ..operators import apply_remesh_if_globally_stable

HistoryLog = MutableSequence[MutableMapping[str, object]]

__all__ = (
    "ALIAS_VF",
    "ALIAS_DNFR",
    "ALIAS_EPI",
    "ALIAS_SI",
    "apply_canonical_clamps",
    "validate_canon",
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


def _log_clamp(
    hist: HistoryLog,
    node: NodeId | None,
    attr: str,
    value: float,
    lo: float,
    hi: float,
) -> None:
    if value < lo or value > hi:
        hist.append({"node": node, "attr": attr, "value": float(value)})


def _normalize_job_overrides(
    job_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
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


def _coerce_jobs_value(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _sanitize_jobs(value: int | None, *, allow_non_positive: bool) -> int | None:
    if value is None:
        return None
    if not allow_non_positive and value <= 0:
        return None
    return value


def _resolve_jobs_override(
    overrides: Mapping[str, Any],
    key: str,
    graph_value: Any,
    *,
    allow_non_positive: bool,
) -> int | None:
    norm_key = key.upper()
    if overrides and norm_key in overrides:
        return _sanitize_jobs(
            _coerce_jobs_value(overrides.get(norm_key)),
            allow_non_positive=allow_non_positive,
        )

    return _sanitize_jobs(
        _coerce_jobs_value(graph_value),
        allow_non_positive=allow_non_positive,
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
    if positional:
        return factory(G)

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
    elif inspect.isclass(candidate) and issubclass(
        candidate, integrators.AbstractIntegrator
    ):
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


def apply_canonical_clamps(
    nd: MutableMapping[str, Any],
    G: TNFRGraph | None = None,
    node: NodeId | None = None,
) -> None:
    if G is not None:
        graph_dict = cast(MutableMapping[str, Any], G.graph)
        graph_data: Mapping[str, Any] = graph_dict
    else:
        graph_dict = None
        graph_data = DEFAULTS
    eps_min = float(graph_data.get("EPI_MIN", DEFAULTS["EPI_MIN"]))
    eps_max = float(graph_data.get("EPI_MAX", DEFAULTS["EPI_MAX"]))
    vf_min = float(graph_data.get("VF_MIN", DEFAULTS["VF_MIN"]))
    vf_max = float(graph_data.get("VF_MAX", DEFAULTS["VF_MAX"]))
    theta_wrap = bool(graph_data.get("THETA_WRAP", DEFAULTS["THETA_WRAP"]))

    epi = cast(float, get_attr(nd, ALIAS_EPI, 0.0))
    vf = get_attr(nd, ALIAS_VF, 0.0)
    th_val = get_theta_attr(nd, 0.0)
    th = 0.0 if th_val is None else float(th_val)

    strict = bool(
        graph_data.get("VALIDATORS_STRICT", DEFAULTS.get("VALIDATORS_STRICT", False))
    )
    if strict and graph_dict is not None:
        history = cast(MutableMapping[str, Any], graph_dict.setdefault("history", {}))
        hist = cast(
            HistoryLog,
            history.setdefault("clamp_alerts", []),
        )
        _log_clamp(hist, node, "EPI", float(epi), eps_min, eps_max)
        _log_clamp(hist, node, "VF", float(vf), vf_min, vf_max)

    set_attr(nd, ALIAS_EPI, clamp(epi, eps_min, eps_max))

    vf_val = float(clamp(vf, vf_min, vf_max))
    if G is not None and node is not None:
        set_vf(G, node, vf_val, update_max=False)
    else:
        set_attr(nd, ALIAS_VF, vf_val)

    if theta_wrap:
        new_th = (th + math.pi) % (2 * math.pi) - math.pi
        if G is not None and node is not None:
            set_theta(G, node, new_th)
        else:
            set_theta_attr(nd, new_th)


def validate_canon(G: TNFRGraph) -> TNFRGraph:
    for n, nd in G.nodes(data=True):
        apply_canonical_clamps(cast(MutableMapping[str, Any], nd), G, cast(NodeId, n))
    maxes = multi_recompute_abs_max(G, {"_vfmax": ALIAS_VF})
    G.graph.update(maxes)
    return G


def _run_before_callbacks(
    G: TNFRGraph,
    *,
    step_idx: int,
    dt: float | None,
    use_Si: bool,
    apply_glyphs: bool,
) -> None:
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
    compute_dnfr_cb = G.graph.get(
        "compute_delta_nfr", default_compute_delta_nfr
    )
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
        elif any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
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
            getattr(dynamics_module, "compute_Si", None)
            if dynamics_module is not None
            else None
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
        apply_canonical_clamps(
            cast(MutableMapping[str, Any], nd), G, cast(NodeId, n)
        )
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
    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    tau = max(tau_g, tau_l)
    maxlen = max(2 * tau + 5, 64)
    epi_hist = G.graph.get("_epi_hist")
    if not isinstance(epi_hist, deque) or epi_hist.maxlen != maxlen:
        epi_hist = deque(list(epi_hist or [])[-maxlen:], maxlen=maxlen)
        G.graph["_epi_hist"] = epi_hist
    epi_hist.append(
        {n: get_attr(nd, ALIAS_EPI, 0.0) for n, nd in G.nodes(data=True)}
    )


def _maybe_remesh(G: TNFRGraph) -> None:
    apply_remesh_if_globally_stable(G)


def _run_validators(G: TNFRGraph) -> None:
    from ..utils import run_validators

    run_validators(G)


def _run_after_callbacks(G, *, step_idx: int) -> None:
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


def step(
    G: TNFRGraph,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
    n_jobs: Mapping[str, Any] | None = None,
) -> None:
    job_overrides = _normalize_job_overrides(n_jobs)
    hist = ensure_history(G)
    step_idx = len(hist.setdefault("C_steps", []))
    _run_before_callbacks(
        G, step_idx=step_idx, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs
    )
    _update_nodes(
        G,
        dt=dt,
        use_Si=use_Si,
        apply_glyphs=apply_glyphs,
        step_idx=step_idx,
        hist=hist,
        job_overrides=job_overrides,
    )
    _update_epi_hist(G)
    _maybe_remesh(G)
    _run_validators(G)
    _run_after_callbacks(G, step_idx=step_idx)


def run(
    G: TNFRGraph,
    steps: int,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
    n_jobs: Mapping[str, Any] | None = None,
) -> None:
    steps_int = int(steps)
    if steps_int < 0:
        raise ValueError("'steps' must be non-negative")
    stop_cfg = get_graph_param(G, "STOP_EARLY", dict)
    stop_enabled = False
    if stop_cfg and stop_cfg.get("enabled", False):
        w = int(stop_cfg.get("window", 25))
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
            series = history.get("stable_frac", [])
            if not isinstance(series, list):
                series = list(series)
            if len(series) >= w and all(v >= frac for v in series[-w:]):
                break

