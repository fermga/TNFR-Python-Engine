"""Gamma registry."""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, NamedTuple

from .alias import get_theta_attr
from .constants import DEFAULTS
from .utils import json_dumps
from .metrics.trig_cache import get_trig_cache
from .types import GammaSpec, NodeId, TNFRGraph
from .utils import (
    edge_version_cache,
    get_graph_mapping,
    get_logger,
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
)


@lru_cache(maxsize=1)
def _default_gamma_spec() -> tuple[bytes, str]:
    dumped = json_dumps(dict(DEFAULT_GAMMA), sort_keys=True, to_bytes=True)
    hash_ = hashlib.blake2b(dumped, digest_size=16).hexdigest()
    return dumped, hash_


def _ensure_kuramoto_cache(G: TNFRGraph, t: float | int) -> None:
    """Cache ``(R, ψ)`` for the current step ``t`` using ``edge_version_cache``."""
    checksum = G.graph.get("_dnfr_nodes_checksum")
    if checksum is None:
        # reuse checksum from cached_nodes_and_A when available
        checksum = node_set_checksum(G)
    nodes_sig = (len(G), checksum)
    max_steps = int(G.graph.get("KURAMOTO_CACHE_STEPS", 1))

    def builder() -> dict[str, float]:
        R, psi = kuramoto_R_psi(G)
        return {"R": R, "psi": psi}

    key = (t, nodes_sig)
    entry = edge_version_cache(G, key, builder, max_entries=max_steps)
    G.graph["_kuramoto_cache"] = entry


def kuramoto_R_psi(G: TNFRGraph) -> tuple[float, float]:
    """Return ``(R, ψ)`` for Kuramoto order using θ from all nodes."""
    max_steps = int(G.graph.get("KURAMOTO_CACHE_STEPS", 1))
    trig = get_trig_cache(G, cache_size=max_steps)
    n = len(trig.theta)
    if n == 0:
        return 0.0, 0.0

    cos_sum = sum(trig.cos.values())
    sin_sum = sum(trig.sin.values())
    R = math.hypot(cos_sum, sin_sum) / n
    psi = math.atan2(sin_sum, cos_sum)
    return R, psi


def _kuramoto_common(G: TNFRGraph, node: NodeId, _cfg: GammaSpec) -> tuple[float, float, float]:
    """Return ``(θ_i, R, ψ)`` for Kuramoto-based Γ functions.

    Reads cached global order ``R`` and mean phase ``ψ`` and obtains node
    phase ``θ_i``. ``_cfg`` is accepted only to keep a homogeneous signature
    with Γ evaluators.
    """
    cache = G.graph.get("_kuramoto_cache", {})
    R = float(cache.get("R", 0.0))
    psi = float(cache.get("psi", 0.0))
    th_val = get_theta_attr(G.nodes[node], 0.0)
    th_i = float(th_val if th_val is not None else 0.0)
    return th_i, R, psi


def _read_gamma_raw(G: TNFRGraph) -> GammaSpec | None:
    """Return raw Γ specification from ``G.graph['GAMMA']``.

    The returned value is the direct contents of ``G.graph['GAMMA']`` when
    it is a mapping or the result of :func:`get_graph_mapping` if a path is
    provided.  Final validation and caching are handled elsewhere.
    """

    raw = G.graph.get("GAMMA")
    if raw is None or isinstance(raw, Mapping):
        return raw
    return get_graph_mapping(
        G,
        "GAMMA",
        "G.graph['GAMMA'] is not a mapping; using {'type': 'none'}",
    )


def _get_gamma_spec(G: TNFRGraph) -> GammaSpec:
    """Return validated Γ specification caching results.

    The raw value from ``G.graph['GAMMA']`` is cached together with the
    normalized specification and its hash. When the raw value is unchanged,
    the cached spec is returned without re-reading or re-validating,
    preventing repeated warnings or costly hashing.
    """

    raw = G.graph.get("GAMMA")
    cached_raw = G.graph.get("_gamma_raw")
    cached_spec = G.graph.get("_gamma_spec")
    cached_hash = G.graph.get("_gamma_spec_hash")

    def _hash_mapping(mapping: GammaSpec) -> str:
        dumped = json_dumps(mapping, sort_keys=True, to_bytes=True)
        return hashlib.blake2b(dumped, digest_size=16).hexdigest()

    mapping_hash: str | None = None
    if isinstance(raw, Mapping):
        mapping_hash = _hash_mapping(raw)
        if raw is cached_raw and cached_spec is not None and cached_hash == mapping_hash:
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

    # Store raw input, validated spec and its hash for future calls
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
    converted to ``float``. If a key is missing, its value from ``defaults`` is
    used. Values convertible to ``float`` (e.g. strings) are accepted.

    Example
    -------
    >>> beta, R0 = _gamma_params(cfg, beta=0.0, R0=0.0)
    """

    return tuple(float(cfg.get(name, default)) for name, default in defaults.items())


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


def _builder_tanh(th_i: float, R: float, psi: float, beta: float, k: float, R0: float) -> float:
    return beta * math.tanh(k * (R - R0)) * math.cos(th_i - psi)


def gamma_kuramoto_linear(G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec) -> float:
    """Linear Kuramoto coupling for Γi(R).

    Formula: Γ = β · (R - R0) · cos(θ_i - ψ)
      - R ∈ [0,1] is the global phase order.
      - ψ is the mean phase (coordination direction).
      - β, R0 are parameters (gain/threshold).

    Use: reinforces integration when the network already shows phase
    coherence (R>R0).
    """

    return _gamma_kuramoto(G, node, cfg, _builder_linear, beta=0.0, R0=0.0)


def gamma_kuramoto_bandpass(G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec) -> float:
    """Compute Γ = β · R(1-R) · sign(cos(θ_i - ψ))."""

    return _gamma_kuramoto(G, node, cfg, _builder_bandpass, beta=0.0)


def gamma_kuramoto_tanh(G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec) -> float:
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
    Γ type is not registered a warning is emitted (or ``ValueError`` in
    strict mode) and ``gamma_none`` is used.

    ``log_level`` controls the logging level for captured errors when
    ``strict`` is ``False``. If omitted, ``logging.ERROR`` is used in
    strict mode and ``logging.DEBUG`` otherwise.
    """
    spec = _get_gamma_spec(G)
    spec_type = spec.get("type", "none")
    reg_entry = GAMMA_REGISTRY.get(spec_type)
    if reg_entry is None:
        msg = f"Unknown GAMMA type: {spec_type}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        entry = GammaEntry(gamma_none, False)
    else:
        entry = reg_entry
    if entry.needs_kuramoto:
        _ensure_kuramoto_cache(G, t)
    try:
        return float(entry.fn(G, node, t, spec))
    except (ValueError, TypeError, ArithmeticError) as exc:
        level = log_level if log_level is not None else (logging.ERROR if strict else logging.DEBUG)
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
