r"""R1 cache-isolation tests (report R1-T01 / N01).

The operator-equivariance audit reuses the seed graphs across all 13 operators.
TNFR content-keyed caches live in ``G.graph`` and survive ``G.copy()`` by shared
reference; keyed on a label-independent node-set checksum, they collided between
the isomorphic copies ``A`` and ``B = σ(A)`` and leaked ΔNFR prep across
operators (spurious ``~1e-3`` residuals). :func:`_isolate_graph_caches` drops
those caches per copy, making every measurement an independent experiment.

Acceptance (report R1-T01): ``max ‖F_cache(x) − F_nocache(x)‖ ≤ τ`` for every
operator — i.e. the residual is independent of any warm engine cache, of the
operator order, and of repeated runs.
"""

from __future__ import annotations

import networkx as nx

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics import operator_equivariance as oe
from tnfr.physics.operator_equivariance import (
    audit_operator_equivariance,
    operator_equivariance_residual,
    _isolate_graph_caches,
)
from tnfr.operators.definitions import Resonance, Silence, Transition

# The operator-equivariance tolerance (the audit's own default). Residuals on a
# Fix(Γ) seed cancel exactly, so this is a generous bound, not a fitted constant.
TOL = 1e-6

_CACHE_HINTS = ("cache", "checksum", "dirty")


def _residuals():
    return {r.name: r.residual for r in audit_operator_equivariance()}


def _leaks(results):
    return [r.name for r in results if not r.is_equivariant]


# --------------------------------------------------------------------------- #
# Root result: the batch audit no longer leaks
# --------------------------------------------------------------------------- #
def test_batch_audit_has_no_operator_leak():
    results = audit_operator_equivariance()
    assert _leaks(results) == []
    assert max(r.residual for r in results) <= TOL


# --------------------------------------------------------------------------- #
# Report R1-T01 test 1: cache enabled/disabled parity
# --------------------------------------------------------------------------- #
def test_equivariance_cache_enabled_disabled_parity():
    # First run leaves the process-level engine caches warm; a second run must
    # give bit-identical residuals (the per-copy isolation makes the result
    # independent of any warm cache).
    cold = _residuals()
    warm = _residuals()  # engine caches are now warm from the first run
    assert cold == warm
    assert all(v <= TOL for v in warm.values())


# --------------------------------------------------------------------------- #
# Report R1-T01 test 2: operator-audit order independence
# --------------------------------------------------------------------------- #
def test_operator_audit_order_independent():
    baseline = _residuals()
    # run a different subset/order on the shared seeds; must not perturb anything
    cases = oe._test_cases()
    for cls in (Transition, Silence, Resonance):
        for G, sigma, node in cases:
            assert operator_equivariance_residual(cls(), G, sigma, node) <= TOL
    assert _residuals() == baseline  # no accumulated state


# --------------------------------------------------------------------------- #
# Report R1-T01 test 3: identical content, distinct context, no leak
# --------------------------------------------------------------------------- #
def test_identical_content_distinct_context_no_leak():
    cases = oe._test_cases()
    G, sigma, node = cases[0]
    # two content-identical experiments in sequence must give the same clean 0;
    # the isomorphic sibling B = σ(A) must not leak its ΔNFR prep into A.
    r1 = operator_equivariance_residual(Resonance(), G, sigma, node)
    r2 = operator_equivariance_residual(Resonance(), G, sigma, node)
    assert r1 <= TOL
    assert r2 == r1


# --------------------------------------------------------------------------- #
# Report R1-T01 test 4: parallel-experiment state isolation
# --------------------------------------------------------------------------- #
def test_parallel_experiment_state_isolation():
    # two independent audit runs must be bit-identical (no shared mutable state)
    run1 = _residuals()
    run2 = _residuals()
    assert run1 == run2


# --------------------------------------------------------------------------- #
# Report R1-T01 test 5: cache invalidation after graph mutation
# --------------------------------------------------------------------------- #
def test_cache_invalidation_after_graph_mutation():
    cases = oe._test_cases()
    G, sigma, node = cases[0]
    default_compute_delta_nfr(G)  # warm G's per-graph caches
    assert any(h in k for k in G.graph for h in _CACHE_HINTS)
    # a copy, after isolation, carries none of the inherited content caches
    H = G.copy()
    _isolate_graph_caches(H)
    assert not any(h in k for k in H.graph for h in _CACHE_HINTS)
    # config keys survive (isolation drops caches, not the ΔNFR configuration)
    assert "_DNFR_META" in H.graph or "_dnfr_weights" in H.graph
    # and the residual on the warm G is still the clean 0
    assert operator_equivariance_residual(Silence(), G, sigma, node) <= TOL


# --------------------------------------------------------------------------- #
# Isolation is exact: warm vs cold graph give the same residual
# --------------------------------------------------------------------------- #
def test_warm_and_cold_graph_give_same_residual():
    # F_cache(x): residual on a graph whose caches are warm.
    # F_nocache(x): residual on a freshly built, cold graph.
    warm_cases = oe._test_cases()
    G, sigma, node = warm_cases[0]
    default_compute_delta_nfr(G)
    f_cache = operator_equivariance_residual(Resonance(), G, sigma, node)

    cold_cases = oe._test_cases()
    Gc, sigmac, nodec = cold_cases[0]
    f_nocache = operator_equivariance_residual(Resonance(), Gc, sigmac, nodec)
    assert abs(f_cache - f_nocache) <= TOL


def test_isolate_graph_caches_is_a_noop_on_a_clean_graph():
    G = nx.cycle_graph(4)
    _isolate_graph_caches(G)  # nothing to drop; must not raise
    assert G.number_of_nodes() == 4
