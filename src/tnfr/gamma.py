"""Registry of optional additive EPI-rate sources.

These configured Kuramoto-based and harmonic terms extend the unforced nodal
law. Their units must match dEPI/dt; they need not vanish at zero capacity.
Using them is a declared model choice, not evidence that they emerge from TNFR.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, NamedTuple

from ._exact_time import finite_represented_real
from .alias import get_theta_attr
from .constants import DEFAULTS
from .metrics.trig_cache import get_trig_cache
from .types import GammaSpec, NodeId, TNFRGraph
from .utils import (
    edge_version_cache,
    get_graph_mapping,
    get_logger,
    json_dumps,
    node_set_checksum,
)

logger = get_logger(__name__)

DEFAULT_GAMMA: Mapping[str, Any] = MappingProxyType(dict(DEFAULTS["GAMMA"]))

__all__ = (
    "kuramoto_R_psi",
    "gamma_none",
    "gamma_kuramoto_linear",
    "gamma_kuramoto_bandpass",
    "gamma_kuramoto_tanh",
    "gamma_harmonic",
    "GammaEntry",
    "GAMMA_REGISTRY",
    "eval_gamma",
    "eval_gamma_vectorized",
)


@lru_cache(maxsize=1)
def _default_gamma_spec() -> tuple[bytes, str]:
    dumped = json_dumps(dict(DEFAULT_GAMMA), sort_keys=True, to_bytes=True)
    hash_ = hashlib.blake2b(dumped, digest_size=16).hexdigest()
    return dumped, hash_


def _ensure_kuramoto_cache(G: TNFRGraph, t: float | int) -> None:
    """Cache order for the actual phases, even when the requested time repeats."""
    checksum = G.graph.get("_dnfr_nodes_checksum")
    if checksum is None:
        # reuse checksum from cached_nodes_and_A when available
        checksum = node_set_checksum(G)
    nodes_sig = (len(G), checksum)
    max_steps = int(G.graph.get("KURAMOTO_CACHE_STEPS", 1))
    trig = get_trig_cache(G, cache_size=max_steps)
    phase_signature = tuple(trig.theta_checksums[node] for node in trig.order)

    def builder() -> dict[str, float]:
        R, psi = _kuramoto_from_trig(trig)
        return {"R": R, "psi": psi}

    key = (t, nodes_sig, phase_signature)
    entry = edge_version_cache(G, key, builder, max_entries=max_steps)
    G.graph["_kuramoto_cache"] = entry


def kuramoto_R_psi(G: TNFRGraph) -> tuple[float, float]:
    """Return ``(R, ψ)`` for Kuramoto order using θ from all nodes."""
    max_steps = int(G.graph.get("KURAMOTO_CACHE_STEPS", 1))
    trig = get_trig_cache(G, cache_size=max_steps)
    return _kuramoto_from_trig(trig)


def _kuramoto_from_trig(trig: Any) -> tuple[float, float]:
    """Reduce one phase-aware trigonometric snapshot to global order."""
    n = len(trig.theta)
    if n == 0:
        return 0.0, 0.0

    cos_sum = sum(trig.cos.values())
    sin_sum = sum(trig.sin.values())
    R = math.hypot(cos_sum, sin_sum) / n
    psi = math.atan2(sin_sum, cos_sum)
    return R, psi


def _kuramoto_common(
    G: TNFRGraph, node: NodeId, _cfg: GammaSpec
) -> tuple[float, float, float]:
    """Return ``(θ_i, R, ψ)`` for Kuramoto-based Γ functions.

    Reads cached global order ``R`` and mean phase ``ψ`` and obtains node
    phase ``θ_i``. ``_cfg`` is accepted only to keep a homogeneous signature
    with Γ evaluators.
    """
    cache = G.graph.get("_kuramoto_cache", {})
    R = float(cache.get("R", 0.0))
    psi = float(cache.get("psi", 0.0))
    th_i = get_theta_attr(
        G.nodes[node],
        0.0,
        strict=True,
        conv=lambda raw: _gamma_real(raw, "Gamma phase"),
    )
    return th_i, R, psi


def _read_gamma_raw(G: TNFRGraph) -> GammaSpec | None:
    """Return raw Γ specification from ``G.graph['GAMMA']``.

    The returned value is the mapping in ``G.graph['GAMMA']`` or the fallback
    from :func:`get_graph_mapping`. This compatibility loader does not
    load a filesystem path. Strict execution rejects invalid containers.
    """

    raw = G.graph.get("GAMMA")
    if raw is None or isinstance(raw, Mapping):
        return raw
    return get_graph_mapping(
        G,
        "GAMMA",
        "G.graph['GAMMA'] is not a mapping; using {'type': 'none'}",
    )


def _get_gamma_spec(G: TNFRGraph, *, strict: bool = False) -> GammaSpec:
    """Return the loaded Γ specification with content-aware caching.

    The raw value from ``G.graph['GAMMA']`` is cached together with the
    loaded specification and its hash. When the raw value is unchanged,
    the cached spec is returned without re-reading or re-validating,
    preventing repeated warnings or costly hashing.
    """

    raw = G.graph.get("GAMMA")
    if strict and raw is not None and not isinstance(raw, Mapping):
        raise ValueError("GAMMA must be a mapping or None")
    cached_raw = G.graph.get("_gamma_raw")
    cached_spec = G.graph.get("_gamma_spec")
    cached_hash = G.graph.get("_gamma_spec_hash")

    def _hash_mapping(mapping: GammaSpec) -> str:
        dumped = json_dumps(dict(mapping), sort_keys=True, to_bytes=True)
        return hashlib.blake2b(dumped, digest_size=16).hexdigest()

    mapping_hash: str | None = None
    if isinstance(raw, Mapping):
        mapping_hash = _hash_mapping(raw)
        if (
            raw is cached_raw
            and cached_spec is not None
            and cached_hash == mapping_hash
        ):
            return cached_spec
    elif raw is cached_raw and cached_spec is not None and cached_hash is not None:
        return cached_spec

    if raw is None:
        spec = DEFAULT_GAMMA
        _, cur_hash = _default_gamma_spec()
    elif isinstance(raw, Mapping):
        spec = raw
        cur_hash = mapping_hash if mapping_hash is not None else _hash_mapping(spec)
    else:
        spec_raw = _read_gamma_raw(G)
        if isinstance(spec_raw, Mapping) and spec_raw is not None:
            spec = spec_raw
            cur_hash = _hash_mapping(spec)
        else:
            spec = DEFAULT_GAMMA
            _, cur_hash = _default_gamma_spec()

    # Loading does not validate the registry entry or its numerical parameters.
    G.graph["_gamma_raw"] = raw
    G.graph["_gamma_spec"] = spec
    G.graph["_gamma_spec_hash"] = cur_hash
    return spec


# -----------------
# Helpers
# -----------------


def _gamma_params(cfg: GammaSpec, **defaults: float) -> tuple[float, ...]:
    """Return normalized Γ parameters from ``cfg``.

    Parameters are retrieved from ``cfg`` using the keys in ``defaults`` and
    validated as finite represented real scalars. If a key is missing, its
    value from ``defaults`` is used. Text and booleans are not rate parameters.

    Example
    -------
    >>> beta, R0 = _gamma_params(cfg, beta=0.0, R0=0.0)
    """

    return tuple(
        _gamma_real(cfg.get(name, default), f"Gamma {name}")
        for name, default in defaults.items()
    )


def _gamma_real(value: Any, label: str) -> float:
    """Use the shared represented-real boundary before any coercion."""
    try:
        return finite_represented_real(value, label)[0]
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


# -----------------
# Canonical Γi(R)
# -----------------


def gamma_none(G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec) -> float:
    """Return ``0.0`` to disable Γ forcing for the given node."""

    return 0.0


def _gamma_kuramoto(
    G: TNFRGraph,
    node: NodeId,
    cfg: GammaSpec,
    builder: Callable[..., float],
    **defaults: float,
) -> float:
    """Construct a Kuramoto-based Γ function.

    ``builder`` receives ``(θ_i, R, ψ, *params)`` where ``params`` are
    extracted from ``cfg`` according to ``defaults``.
    """

    params = _gamma_params(cfg, **defaults)
    th_i, R, psi = _kuramoto_common(G, node, cfg)
    return builder(th_i, R, psi, *params)


def _builder_linear(th_i: float, R: float, psi: float, beta: float, R0: float) -> float:
    return beta * (R - R0) * math.cos(th_i - psi)


def _builder_bandpass(th_i: float, R: float, psi: float, beta: float) -> float:
    sgn = 1.0 if math.cos(th_i - psi) >= 0.0 else -1.0
    return beta * R * (1.0 - R) * sgn


def _builder_tanh(
    th_i: float, R: float, psi: float, beta: float, k: float, R0: float
) -> float:
    return beta * math.tanh(k * (R - R0)) * math.cos(th_i - psi)


def gamma_kuramoto_linear(
    G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec
) -> float:
    """Linear Kuramoto coupling for Γi(R).

    Formula: Γ = β · (R - R0) · cos(θ_i - ψ)
      - R ∈ [0,1] is the global phase order.
      - ψ is the mean phase (coordination direction).
      - β, R0 are parameters (gain/threshold).

    Use: reinforces integration when the network already shows phase
    coherence (R>R0).
    """

    return _gamma_kuramoto(G, node, cfg, _builder_linear, beta=0.0, R0=0.0)


def gamma_kuramoto_bandpass(
    G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec
) -> float:
    """Compute Γ = β · R(1-R) · sign(cos(θ_i - ψ))."""

    return _gamma_kuramoto(G, node, cfg, _builder_bandpass, beta=0.0)


def gamma_kuramoto_tanh(
    G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec
) -> float:
    """Saturating tanh coupling for Γi(R).

    Formula: Γ = β · tanh(k·(R - R0)) · cos(θ_i - ψ)
      - β: coupling gain
      - k: tanh slope (how fast it saturates)
      - R0: activation threshold
    """

    return _gamma_kuramoto(G, node, cfg, _builder_tanh, beta=0.0, k=1.0, R0=0.0)


def gamma_harmonic(G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec) -> float:
    """Harmonic forcing aligned with the global phase field.

    Formula: Γ = β · sin(ω·t + φ) · cos(θ_i - ψ)
      - β: coupling gain
      - ω: angular frequency of the forcing
      - φ: initial phase of the forcing
    """
    beta, omega, phi = _gamma_params(cfg, beta=0.0, omega=1.0, phi=0.0)
    th_i, _, psi = _kuramoto_common(G, node, cfg)
    return beta * math.sin(omega * t + phi) * math.cos(th_i - psi)


class GammaEntry(NamedTuple):
    """Lookup entry linking Γ evaluators with their preconditions."""

    fn: Callable[[TNFRGraph, NodeId, float | int, GammaSpec], float]
    needs_kuramoto: bool


# ``GAMMA_REGISTRY`` associates each coupling name with a ``GammaEntry`` where
# ``fn`` is the evaluation function and ``needs_kuramoto`` indicates whether
# the global phase order must be precomputed.
GAMMA_REGISTRY: dict[str, GammaEntry] = {
    "none": GammaEntry(gamma_none, False),
    "kuramoto_linear": GammaEntry(gamma_kuramoto_linear, True),
    "kuramoto_bandpass": GammaEntry(gamma_kuramoto_bandpass, True),
    "kuramoto_tanh": GammaEntry(gamma_kuramoto_tanh, True),
    "harmonic": GammaEntry(gamma_harmonic, True),
}

# Keep the original implementations distinct from user replacements under a
# built-in name: only these entries have the array formulas implemented below.
_BUILTIN_GAMMA_ENTRIES = GAMMA_REGISTRY.copy()


def _resolve_gamma_entry(spec: GammaSpec) -> GammaEntry:
    """Resolve the live registry entry without silently disabling a source."""
    spec_type = spec.get("type", "none")
    if not isinstance(spec_type, str) or spec_type not in GAMMA_REGISTRY:
        raise ValueError(f"Unknown GAMMA type: {spec_type!r}")
    return GAMMA_REGISTRY[spec_type]


def _uses_builtin_gamma(spec: GammaSpec) -> bool:
    """Whether detached array evaluation implements the live registered entry."""
    entry = _resolve_gamma_entry(spec)
    return entry == _BUILTIN_GAMMA_ENTRIES.get(spec.get("type", "none"))


def eval_gamma(
    G: TNFRGraph,
    node: NodeId,
    t: float | int,
    *,
    strict: bool = False,
    log_level: int | None = None,
) -> float:
    """Evaluate Γi for ``node`` using ``G.graph['GAMMA']`` specification.

    If ``strict`` is ``True`` exceptions raised during evaluation are
    propagated instead of returning ``0.0``. Likewise, if the specified
    Γ type is not registered, permissive evaluation logs the failure and
    returns zero. Runtime execution uses strict evaluation. Neither mode
    admits booleans/text as rates or built-in numerical parameters.

    ``log_level`` controls the logging level for captured errors when
    ``strict`` is ``False``. If omitted, ``logging.ERROR`` is used in
    strict mode and ``logging.DEBUG`` otherwise.
    """
    try:
        spec = _get_gamma_spec(G, strict=strict)
        entry = _resolve_gamma_entry(spec)
        t = _gamma_real(t, "Gamma time")
        if entry.needs_kuramoto:
            _ensure_kuramoto_cache(G, t)
        return _gamma_real(entry.fn(G, node, t, spec), "Gamma rate")
    except (ValueError, TypeError, ArithmeticError) as exc:
        level = (
            log_level
            if log_level is not None
            else (logging.ERROR if strict else logging.DEBUG)
        )
        logger.log(
            level,
            "Failed to evaluate Γi for node %s at t=%s: %s: %s",
            node,
            t,
            exc.__class__.__name__,
            exc,
        )
        if strict:
            raise
        return 0.0


def eval_gamma_vectorized(
    G: TNFRGraph,
    theta_arr: Any,
    t: float,
    np_mod: Any,
    *,
    strict: bool = False,
) -> Any:
    """Evaluate Γi for all nodes using vectorized operations.

    Args:
        G: The graph (for gamma spec).
        theta_arr: NumPy array of node phases (θ).
        t: Current time.
        np_mod: The NumPy module.
        strict: Propagate invalid source configuration/evaluation instead of
            returning zero, matching :func:`eval_gamma`. Integrators use True.

    Returns:
        NumPy array of Γ values.

    Registered custom entries are evaluated once per node in graph order via
    the scalar owner. Their graph reads are not simulated array-stage states.
    """
    try:
        spec = _get_gamma_spec(G, strict=strict)
        t = _gamma_real(t, "Gamma time")
        if not _uses_builtin_gamma(spec):
            return np_mod.asarray(
                [eval_gamma(G, node, t, strict=strict) for node in G], dtype=float
            )
        if spec.get("type", "none") == "none":
            return np_mod.zeros(len(G), dtype=float)
        raw_phases = np_mod.asarray(theta_arr, dtype=object)
        if raw_phases.ndim != 1 or len(raw_phases) != len(G):
            raise ValueError("Gamma phases must have one scalar per graph node")
        theta_arr = np_mod.asarray(
            [_gamma_real(value, "Gamma phase") for value in raw_phases], dtype=float
        )
        result = _eval_builtin_gamma_vectorized(spec, theta_arr, t, np_mod)
        if not np_mod.isfinite(result).all():
            raise ValueError("Gamma rate must be finite")
        return result
    except (ValueError, TypeError, ArithmeticError):
        if strict:
            raise
        logger.debug("Failed to evaluate vectorized Gamma", exc_info=True)
        return np_mod.zeros(len(G), dtype=float)


def _eval_builtin_gamma_vectorized(
    spec: GammaSpec, theta_arr: Any, t: float, np_mod: Any
) -> Any:
    """Array formulas for the unchanged built-in entries only."""
    spec_type = spec.get("type", "none")

    # Precompute Kuramoto order if needed
    # For vectorized, we assume theta_arr contains all nodes in order
    # so we can compute R and psi directly from it.

    R = 0.0
    psi = 0.0
    needs_kuramoto = spec_type.startswith("kuramoto") or spec_type == "harmonic"

    if needs_kuramoto:
        # Using real arithmetic for safety/speed
        cos_sum = np_mod.sum(np_mod.cos(theta_arr))
        sin_sum = np_mod.sum(np_mod.sin(theta_arr))
        n = theta_arr.size
        if n > 0:
            R = np_mod.hypot(cos_sum, sin_sum) / n
            psi = np_mod.arctan2(sin_sum, cos_sum)

    # Dispatch
    if spec_type == "kuramoto_linear":
        beta, R0 = _gamma_params(spec, beta=0.0, R0=0.0)
        # beta * (R - R0) * cos(theta - psi)
        return beta * (R - R0) * np_mod.cos(theta_arr - psi)

    elif spec_type == "kuramoto_bandpass":
        (beta,) = _gamma_params(spec, beta=0.0)
        # beta * R * (1 - R) * sign(cos(theta - psi))
        cos_diff = np_mod.cos(theta_arr - psi)
        # Match scalar behavior: sgn=1 if cos>=0 else -1.
        sgn = np_mod.where(cos_diff >= 0.0, 1.0, -1.0)
        return beta * R * (1.0 - R) * sgn

    elif spec_type == "kuramoto_tanh":
        beta, k, R0 = _gamma_params(spec, beta=0.0, k=1.0, R0=0.0)
        # beta * tanh(k * (R - R0)) * cos(theta - psi)
        return beta * np_mod.tanh(k * (R - R0)) * np_mod.cos(theta_arr - psi)

    elif spec_type == "harmonic":
        beta, omega, phi = _gamma_params(spec, beta=0.0, omega=1.0, phi=0.0)
        # beta * sin(omega * t + phi) * cos(theta - psi)
        return beta * np_mod.sin(omega * t + phi) * np_mod.cos(theta_arr - psi)

    raise ValueError(f"No array formula for Gamma type {spec_type!r}")
