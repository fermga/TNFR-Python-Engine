from __future__ import annotations

import math
import networkx as nx
import pytest

from tnfr.physics import canonical as _canonical  # type: ignore
from tnfr.structural import (  # type: ignore
    create_nfr,
    run_sequence,
    Coherence,
    Silence,
    validate_sequence,
)

 
compute_structural_potential = _canonical.compute_structural_potential


def _make_small_graph(n: int = 10) -> nx.Graph:
    G = nx.path_graph(n)
    # Seed delta_nfr on nodes using public alias name
    for i in G.nodes:
        G.nodes[i]["delta_nfr"] = 1.0 if i % 2 == 0 else -0.5
    return G


def _make_medium_graph(n: int = 120, p: float = 0.05) -> nx.Graph:
    G = nx.erdos_renyi_graph(n=n, p=p, seed=42)
    for i in G.nodes:
        G.nodes[i]["delta_nfr"] = (i % 7) - 3  # balanced positive/negative
    return G


def test_phi_s_cache_key_includes_alpha_distinguishes_results() -> None:
    G = _make_small_graph(10)

    phi_2 = compute_structural_potential(G, alpha=2.0)
    phi_3 = compute_structural_potential(G, alpha=3.0)

    # Dicts should differ for different alpha; if cache key ignored alpha,
    # the second call could incorrectly reuse the first result.
    assert phi_2 != phi_3, "Cache key must include alpha to avoid reuse"


def test_phi_s_landmark_ratio_metadata_with_validation() -> None:
    G = _make_medium_graph(120)

    # Request validation so metadata is embedded
    phi_r2 = compute_structural_potential(
        G, alpha=2.0, landmark_ratio=0.02, validate=True
    )
    phi_r5 = compute_structural_potential(
        G, alpha=2.0, landmark_ratio=0.05, validate=True
    )

    # Metadata must reflect effective ratios used after refinement and
    # demonstrate that landmark_ratio participates in cache keys.
    r2 = float(phi_r2.get("__phi_s_landmark_ratio__", math.nan))
    r5 = float(phi_r5.get("__phi_s_landmark_ratio__", math.nan))

    assert not math.isnan(r2) and not math.isnan(r5)
    # Refinement only increases; ensure final ratios respect requested minima
    assert r2 >= 0.02 and r2 <= 0.5
    assert r5 >= 0.05 and r5 <= 0.5
    # Show caching distinguishes different requested ratios either via
    # distinct effective ratios or differing potentials
    assert (r2 != r5) or (phi_r2 != phi_r5)


def test_diagnostic_oz_zhir_requires_context_gating() -> None:
    # Without diagnostic context, the ephemeral pattern must fail (U1b)
    res_no_ctx = validate_sequence(["dissonance", "mutation"])
    assert not res_no_ctx.passed

    # With diagnostic=True and initial_epi_nonzero=True, it should pass
    res_diag = validate_sequence(
        ["dissonance", "mutation"],
        context={"diagnostic": True, "initial_epi_nonzero": True},
    )
    assert res_diag.passed


def test_run_sequence_logs_u1a_override_when_epi_nonzero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Create a minimal NFR with non-zero EPI
    G, node = create_nfr("seed", epi=1.0, vf=1.0, theta=0.0)

    # Capture logs from tnfr.structural
    caplog.set_level("INFO", logger="tnfr.structural")

    # Start with non-generator operator to trigger the override
    run_sequence(G, node, [Coherence(), Silence()])

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("U1a override (EPI≠0)" in m for m in messages), (
        "Expected U1a override log when starting with non-generator on EPI≠0"
    )
