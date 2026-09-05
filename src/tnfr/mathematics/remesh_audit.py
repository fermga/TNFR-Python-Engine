r"""REMESH contract audit — is the p-adic transport REMESH? (R4b, N09).

The p-adic tower (R4) gives an exact projective transport, and N08 classifies its
lift/scale maps as **morphisms** (LIFT, COARSE_GRAINING, and the same-scale
projection ``Lift·R_e``).  A morphism transports structure *instantaneously*; the
canonical **REMESH** operator (Recursivity, glyph REMESH) instead echoes the form
**across time**,

    ``EPI_new = (1-α)² EPI(t) + α(1-α) EPI(t-τ_l) + α EPI(t-τ_g)``

(:mod:`tnfr.operators.operator_contracts`), a genuine temporal memory.  This
module audits the four REMESH-contract conditions and runs one predefined
campaign contrasting the static p-adic projection with the temporal recurrence.

**Honest result.**  The static tower map satisfies three conditions (NETWORK
scale, preserved identity, U5 multiscale coherence) but **fails the temporal
echo** — it is a projection morphism, not REMESH (``NT-P04b`` stays negative for
the tower).  The temporal recurrence passes all four, so the audit is a real
discriminator, not a vacuous gate.  The only ingredient the lift lacks to be
REMESH is the ``EPI(t) ← EPI(t-τ)`` recursion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .padic_tower import (
    RemeshContractAudit,
    padic_lift_map,
    projective_scale_map,
)

__all__ = [
    "remesh_coefficients",
    "remesh_recurrence",
    "remesh_recurrence_update",
    "scale_projection_update",
    "temporal_echo_residual",
    "audit_remesh_candidate",
    "RemeshCampaign",
    "remesh_campaign",
]


def _frac(matrix) -> np.ndarray:
    return np.array([[float(x) for x in row] for row in matrix], dtype=float)


def _coherence(x) -> float:
    r"""Coherence proxy ``1/(1+std)`` — higher means a more uniform (coherent)
    per-node field."""
    return 1.0 / (1.0 + float(np.std(np.asarray(x, dtype=float))))


def remesh_coefficients(alpha: float) -> tuple[float, float, float]:
    r"""``(c_now, c_local, c_global) = ((1-α)², α(1-α), α)`` — a partition of unity.

    Their sum is ``(1-α)² + α(1-α) + α = 1``, so the recurrence is a convex
    combination for ``0 ≤ α ≤ 1`` (identity- and coherence-preserving).
    """
    a = float(alpha)
    return ((1.0 - a) ** 2, a * (1.0 - a), a)


def remesh_recurrence(now, past_local, past_global, *, alpha: float = 0.5):
    r"""The canonical REMESH temporal echo across two scales (local/global)."""
    c0, cl, cg = remesh_coefficients(alpha)
    return (c0 * np.asarray(now, dtype=float)
            + cl * np.asarray(past_local, dtype=float)
            + cg * np.asarray(past_global, dtype=float))


def remesh_recurrence_update(*, alpha: float = 0.5):
    r"""A candidate REMESH update ``(now, past_l, past_g) → EPI_new`` with **real
    temporal memory**."""
    def update(now, past_local, past_global):
        return remesh_recurrence(now, past_local, past_global, alpha=alpha)
    return update


def scale_projection_update(p: int, e: int):
    r"""The static p-adic candidate ``P = Lift·R_e`` on the fine level.

    It regenerates the fine field from its fiber averages (a same-scale
    idempotent PROJECTION morphism, N08); crucially it **ignores** the delayed
    inputs, so it has no temporal echo.
    """
    proj = _frac(padic_lift_map(p, e)) @ _frac(projective_scale_map(p, e))

    def update(now, past_local, past_global):  # delayed inputs unused (no echo)
        return proj @ np.asarray(now, dtype=float)
    return update


def temporal_echo_residual(update, now, past_local, past_global, *,
                           delta: float = 1e-2) -> float:
    r"""Sensitivity of ``update`` to the **delayed** inputs — the temporal echo.

    Perturbs ``past_local`` and ``past_global`` and measures the induced change
    in the output.  ``0`` means the update ignores history (a static morphism);
    ``> 0`` means a genuine ``EPI(t) ← EPI(t-τ)`` recursion.
    """
    now = np.asarray(now, dtype=float)
    pl = np.asarray(past_local, dtype=float)
    pg = np.asarray(past_global, dtype=float)
    base = update(now, pl, pg)
    step = delta * np.ones_like(pl)
    d_local = np.linalg.norm(update(now, pl + step, pg) - base)
    d_global = np.linalg.norm(update(now, pl, pg + step) - base)
    return float(max(d_local, d_global) / delta)


def audit_remesh_candidate(update, *, network_size: int = 9,
                           tol: float = 1e-9) -> RemeshContractAudit:
    r"""Audit the four REMESH-contract conditions for a candidate ``update``.

    (1) EPI recursion — a non-zero temporal echo; (2) NETWORK scale — acts on the
    whole ``EPI`` field; (3) identity preserved — a coherent (fiber-constant)
    state is a fixed point; (4) U5 multiscale — coherence is not lost.  ``REMESH``
    may be named only when all four hold.
    """
    rng = np.random.default_rng(0)
    n = network_size
    now = rng.standard_normal(n)
    past_local = rng.standard_normal(n)
    past_global = rng.standard_normal(n)
    # a coherent (fiber-constant on residues mod 3) probe for identity / U5
    coherent = np.tile(np.array([1.0, -1.0, 0.5]), n // 3)[:n]

    echo = temporal_echo_residual(update, now, past_local, past_global)
    out = np.asarray(update(now, past_local, past_global), dtype=float)
    fixed = update(coherent, coherent, coherent)

    epi_recursion = echo > 1e-6
    network_scale = out.shape == (n,) and n > 1
    identity_preserved = bool(
        np.linalg.norm(np.asarray(fixed, dtype=float) - coherent) < 1e-6
    )
    u5_multiscale = _coherence(fixed) >= _coherence(coherent) - tol
    return RemeshContractAudit(
        epi_recursion_verified=epi_recursion,
        network_scale_verified=network_scale,
        identity_preserved_verified=identity_preserved,
        u5_multiscale_verified=u5_multiscale,
    )


@dataclass(frozen=True)
class RemeshCampaign:
    """One predefined campaign: static p-adic transport vs temporal recurrence."""

    static_lift: RemeshContractAudit
    temporal_recurrence: RemeshContractAudit

    @property
    def tower_realizes_remesh(self) -> bool:
        """Whether the p-adic tower transport is REMESH (it is not)."""
        return self.static_lift.realizes_remesh

    @property
    def audit_discriminates(self) -> bool:
        """The gate rejects the static morphism yet accepts genuine REMESH."""
        return (not self.static_lift.realizes_remesh
                and self.temporal_recurrence.realizes_remesh)


def remesh_campaign(*, p: int = 3, e: int = 1,
                    alpha: float = 0.5) -> RemeshCampaign:
    r"""Run the predefined R4b campaign on the ``p^{e+1}``-node fine level."""
    n = p ** (e + 1)
    return RemeshCampaign(
        static_lift=audit_remesh_candidate(
            scale_projection_update(p, e), network_size=n),
        temporal_recurrence=audit_remesh_candidate(
            remesh_recurrence_update(alpha=alpha), network_size=n),
    )
