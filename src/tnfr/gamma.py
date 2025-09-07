"""Gamma registry."""

from __future__ import annotations
from typing import Dict, Any, Tuple
import math
import cmath
import logging
import warnings
import hashlib
import json
from collections.abc import Mapping

from .constants import ALIAS_THETA
from .alias import get_attr
from .helpers import node_set_checksum, edge_version_cache
from .logging_utils import get_logger


logger = get_logger(__name__)


def _ensure_kuramoto_cache(G, t) -> None:
    """Cache ``(R, ψ)`` for the current step ``t`` using ``edge_version_cache``."""
    checksum = G.graph.get("_dnfr_nodes_checksum")
    if checksum is None:
        # reuse checksum from cached_nodes_and_A when available
        checksum = node_set_checksum(G)
    nodes_sig = (len(G), checksum)
    max_steps = int(G.graph.get("KURAMOTO_CACHE_STEPS", 1))

    def builder() -> Dict[str, float]:
        R, psi = kuramoto_R_psi(G)
        return {"R": R, "psi": psi}

    key = (t, nodes_sig)
    entry = edge_version_cache(G, key, builder, max_entries=max_steps)
    G.graph["_kuramoto_cache"] = entry


def kuramoto_R_psi(G) -> Tuple[float, float]:
    """Return ``(R, ψ)`` for Kuramoto order using θ from all nodes."""
    acc = 0 + 0j
    n = 0
    for _, nd in G.nodes(data=True):
        th = get_attr(nd, ALIAS_THETA, 0.0)
        acc += cmath.exp(1j * th)
        n += 1
    if n == 0:
        return 0.0, 0.0
    z = acc / n
    return abs(z), math.atan2(z.imag, z.real)


def _kuramoto_common(G, node, _cfg):
    """Return ``(θ_i, R, ψ)`` for Kuramoto-based Γ functions.

    Reads cached global order ``R`` and mean phase ``ψ`` and obtains node
    phase ``θ_i``. ``_cfg`` is accepted only to keep a homogeneous signature
    with Γ evaluators.
    """
    cache = G.graph.get("_kuramoto_cache", {})
    R = float(cache.get("R", 0.0))
    psi = float(cache.get("psi", 0.0))
    th_i = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
    return th_i, R, psi


def _get_gamma_spec(G) -> Mapping[str, Any]:
    raw = G.graph.get("GAMMA")
    cached = G.graph.get("_gamma_spec")
    prev_hash = G.graph.get("_gamma_spec_hash")

    invalid = False
    if raw is None:
        spec: Mapping[str, Any] = {"type": "none"}
    elif not isinstance(raw, Mapping):
        spec = {"type": "none"}
        invalid = True
    else:
        spec = raw

    dumped = json.dumps(spec, sort_keys=True).encode("utf-8")
    cur_hash = hashlib.blake2b(dumped, digest_size=16).hexdigest()
    if cached is not None and prev_hash == cur_hash:
        return cached
    if invalid:
        warnings.warn(
            "G.graph['GAMMA'] no es un mapeo; se usa {'type': 'none'}",
            UserWarning,
            stacklevel=2,
        )
    G.graph["_gamma_spec"] = spec
    G.graph["_gamma_spec_hash"] = cur_hash
    return spec


# -----------------
# Helpers
# -----------------


def _gamma_params(
    cfg: Mapping[str, Any], **defaults: float
) -> tuple[float, ...]:
    """Return normalized Γ parameters from ``cfg``.

    Parameters are retrieved from ``cfg`` using the keys in ``defaults`` and
    converted to ``float``. If a key is missing, its value from ``defaults`` is
    used. Values convertible to ``float`` (e.g. strings) are accepted.

    Example
    -------
    >>> beta, R0 = _gamma_params(cfg, beta=0.0, R0=0.0)
    """

    return tuple(
        float(cfg.get(name, default)) for name, default in defaults.items()
    )


# -----------------
# Γi(R) canónicos
# -----------------


def gamma_none(G, node, t, cfg: Dict[str, Any]) -> float:
    return 0.0


def gamma_kuramoto_linear(G, node, t, cfg: Dict[str, Any]) -> float:
    """Linear Kuramoto coupling for Γi(R).

    Formula: Γ = β · (R - R0) · cos(θ_i - ψ)
      - R ∈ [0,1] is the global phase order.
      - ψ is the mean phase (coordination direction).
      - β, R0 are parameters (gain/threshold).

    Use: reinforces integration when the network already shows phase
    coherence (R>R0).
    """
    beta, R0 = _gamma_params(cfg, beta=0.0, R0=0.0)
    th_i, R, psi = _kuramoto_common(G, node, cfg)
    return beta * (R - R0) * math.cos(th_i - psi)


def gamma_kuramoto_bandpass(G, node, t, cfg: Dict[str, Any]) -> float:
    """Γ = β · R(1-R) · sign(cos(θ_i - ψ))"""
    (beta,) = _gamma_params(cfg, beta=0.0)
    th_i, R, psi = _kuramoto_common(G, node, cfg)
    sgn = 1.0 if math.cos(th_i - psi) >= 0.0 else -1.0
    return beta * R * (1.0 - R) * sgn


def gamma_kuramoto_tanh(G, node, t, cfg: Dict[str, Any]) -> float:
    """Saturating tanh coupling for Γi(R).

    Formula: Γ = β · tanh(k·(R - R0)) · cos(θ_i - ψ)
      - β: coupling gain
      - k: tanh slope (how fast it saturates)
      - R0: activation threshold
    """
    beta, k, R0 = _gamma_params(cfg, beta=0.0, k=1.0, R0=0.0)
    th_i, R, psi = _kuramoto_common(G, node, cfg)
    return beta * math.tanh(k * (R - R0)) * math.cos(th_i - psi)


def gamma_harmonic(G, node, t, cfg: Dict[str, Any]) -> float:
    """Harmonic forcing aligned with the global phase field.

    Formula: Γ = β · sin(ω·t + φ) · cos(θ_i - ψ)
      - β: coupling gain
      - ω: angular frequency of the forcing
      - φ: initial phase of the forcing
    """
    beta, omega, phi = _gamma_params(cfg, beta=0.0, omega=1.0, phi=0.0)
    th_i, _, psi = _kuramoto_common(G, node, cfg)
    return beta * math.sin(omega * t + phi) * math.cos(th_i - psi)


# ``GAMMA_REGISTRY`` asocia el nombre del acoplamiento con un par
# ``(fn, needs_kuramoto)`` donde ``fn`` es la función evaluadora y
# ``needs_kuramoto`` indica si requiere precomputar el orden global de fase.
GAMMA_REGISTRY = {
    "none": (gamma_none, False),
    "kuramoto_linear": (gamma_kuramoto_linear, True),
    "kuramoto_bandpass": (gamma_kuramoto_bandpass, True),
    "kuramoto_tanh": (gamma_kuramoto_tanh, True),
    "harmonic": (gamma_harmonic, True),
}


def eval_gamma(
    G,
    node,
    t,
    *,
    strict: bool = False,
    log_level: int | None = None,
) -> float:
    """Evaluate Γi for ``node`` according to ``G.graph['GAMMA']``
    specification.

    If ``strict`` is ``True`` exceptions raised during evaluation are
    propagated instead of returning ``0.0``. Likewise, if the specified
    Γ type is not registered a warning is emitted (o ``ValueError`` en
    modo estricto) y se usa ``gamma_none``.

    ``log_level`` controls the logging level for captured errors when
    ``strict`` is ``False``. If omitted, ``logging.ERROR`` is used in
    strict mode and ``logging.DEBUG`` otherwise.
    """
    spec = _get_gamma_spec(G)
    spec_type = spec.get("type", "none")
    reg_entry = GAMMA_REGISTRY.get(spec_type)
    if reg_entry is None:
        msg = f"Tipo GAMMA desconocido: {spec_type}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        fn, needs_kuramoto = gamma_none, False
    else:
        fn, needs_kuramoto = reg_entry
    if needs_kuramoto:
        _ensure_kuramoto_cache(G, t)
    try:
        return float(fn(G, node, t, spec))
    except (ValueError, TypeError, ArithmeticError) as exc:
        level = (
            log_level
            if log_level is not None
            else (logging.ERROR if strict else logging.DEBUG)
        )
        logger.log(
            level,
            "Fallo al evaluar Γi para nodo %s en t=%s: %s: %s",
            node,
            t,
            exc.__class__.__name__,
            exc,
        )
        if strict:
            raise
        return 0.0
