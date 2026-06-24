r"""Y3 non-Abelian derivability audit for TNFR structural gauges.

The current canonical TNFR gauge sector is the local U(1) symmetry of the
complex geometric field Ψ = K_φ + i·J_φ.  This module audits whether a
non-Abelian / multi-channel gauge sector can be derived from TNFR-internal
structures without importing external group labels, hand-selected generators,
or non-canonical per-node parameters.

The expected conservative verdict is ``OPEN_DERIVABILITY_GAP`` unless a route
simultaneously supplies:

1. a TNFR-native multiplet;
2. a canonical connection mixing multiplet components;
3. non-commuting generators derived from nodal dynamics;
4. U1–U6 compatibility without external labels.

No such route is currently canonical in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:  # pragma: no cover - available in the test environment
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from ..physics.conservation_gauge_unification import compute_grammar_symmetry_mapping
from ..physics.gauge import compute_gauge_connection, compute_gauge_curvature
from ..physics.unified import compute_complex_geometric_field
from .structural_gap import build_structural_gauge_graph

DEFAULT_NONABELIAN_ROUTES = (
    "u5_nested_epi_multiplet",
    "thol_remesh_internal_space",
    "cycle_basis_bundle",
)


@dataclass(frozen=True)
class NonAbelianCandidateAudit:
    """Audit result for one possible non-Abelian derivability route."""

    route: str
    status: str
    obstruction: str
    nodal_derivable: bool
    grammar_compatible: bool
    requires_external_labels: bool
    has_multiplet: bool
    has_canonical_connection: bool
    has_noncommuting_generators: bool
    evidence: dict[str, Any]


@dataclass(frozen=True)
class NonAbelianDerivabilityReport:
    """Y3 report for non-Abelian derivability from TNFR data only."""

    canonical_gauge_group: str
    u1_baseline_confirmed: bool
    nonabelian_derived: bool
    verdict: str
    candidates: tuple[NonAbelianCandidateAudit, ...]
    summary: dict[str, Any]


def audit_nonabelian_derivability(
    G: Any | None = None,
    *,
    routes: Iterable[str] = DEFAULT_NONABELIAN_ROUTES,
    seed: int = 42,
) -> NonAbelianDerivabilityReport:
    """Audit non-Abelian gauge derivability from TNFR-internal structures.

    Parameters
    ----------
    G : graph, optional
        TNFR-ready graph to inspect.  If omitted, a reproducible finite Y1
        graph is built for baseline auditing.
    routes : iterable[str]
        Candidate derivability routes.  Supported values are listed in
        ``DEFAULT_NONABELIAN_ROUTES``.
    seed : int
        Seed used only when ``G`` is omitted.

    Returns
    -------
    NonAbelianDerivabilityReport
        Conservative audit report.  The current expected verdict is
        ``OPEN_DERIVABILITY_GAP``.
    """
    if G is None:
        G = build_structural_gauge_graph(12, topology="complete", seed=seed)

    route_tuple = tuple(routes)
    unknown_routes = sorted(set(route_tuple).difference(DEFAULT_NONABELIAN_ROUTES))
    if unknown_routes:
        raise ValueError(
            "unsupported non-Abelian derivability route(s): "
            + ", ".join(unknown_routes)
        )

    evidence = _collect_baseline_evidence(G)
    candidates = tuple(_audit_route(route, evidence) for route in route_tuple)
    nonabelian_derived = any(
        candidate.nodal_derivable
        and candidate.grammar_compatible
        and not candidate.requires_external_labels
        and candidate.has_multiplet
        and candidate.has_canonical_connection
        and candidate.has_noncommuting_generators
        for candidate in candidates
    )

    verdict = (
        "NONABELIAN_CANDIDATE_DERIVED"
        if nonabelian_derived
        else "OPEN_DERIVABILITY_GAP"
    )
    summary = {
        "canonical_gauge_group": "U(1)",
        "internal_field_rank": evidence["internal_field_rank"],
        "connection_scalar": evidence["connection_scalar"],
        "curvature_scalar": evidence["curvature_scalar"],
        "cycle_rank": evidence["cycle_rank"],
        "nested_epi_nodes": evidence["nested_epi_nodes"],
        "operator_history_events": evidence["operator_history_events"],
        "candidate_count": len(candidates),
        "derived_candidate_count": sum(
            1 for candidate in candidates if candidate.status == "DERIVED"
        ),
        "scope": "Y3_derivability_audit_not_nonabelian_promotion",
    }

    return NonAbelianDerivabilityReport(
        canonical_gauge_group="U(1)",
        u1_baseline_confirmed=bool(
            evidence["connection_scalar"] and evidence["curvature_scalar"]
        ),
        nonabelian_derived=nonabelian_derived,
        verdict=verdict,
        candidates=candidates,
        summary=summary,
    )


def _collect_baseline_evidence(G: Any) -> dict[str, Any]:
    psi = compute_complex_geometric_field(G)
    connection = compute_gauge_connection(G)
    curvature = compute_gauge_curvature(G)
    grammar = compute_grammar_symmetry_mapping(G)

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "internal_field_rank": _internal_field_rank(psi),
        "connection_scalar": all(
            _is_real_scalar(value) for value in connection.values()
        ),
        "curvature_scalar": all(_is_real_scalar(value) for value in curvature.values()),
        "cycle_rank": _cycle_rank(G),
        "cycle_count_detected": len(curvature),
        "nested_epi_nodes": _count_nested_epi_nodes(G),
        "operator_history_events": _count_operator_history_events(G),
        "grammar_rules_satisfied": sum(1 for item in grammar if item.is_satisfied),
        "grammar_rules_total": len(grammar),
    }


def _audit_route(
    route: str,
    evidence: dict[str, Any],
) -> NonAbelianCandidateAudit:
    if route == "u5_nested_epi_multiplet":
        return _audit_u5_nested_epi_multiplet(evidence)
    if route == "thol_remesh_internal_space":
        return _audit_thol_remesh_internal_space(evidence)
    if route == "cycle_basis_bundle":
        return _audit_cycle_basis_bundle(evidence)
    raise ValueError(f"unsupported non-Abelian derivability route: {route}")


def _audit_u5_nested_epi_multiplet(
    evidence: dict[str, Any],
) -> NonAbelianCandidateAudit:
    has_multiplet = evidence["nested_epi_nodes"] > 0
    status = (
        "OPEN_MULTIPLET_WITHOUT_CANONICAL_CONNECTION"
        if has_multiplet
        else "FAILED_NO_TNFR_MULTIPLET"
    )
    obstruction = (
        "Nested EPI data can supply multiple components, but the current "
        "canonical gauge connection remains scalar A_ij and does not derive "
        "component-mixing parallel transport or non-commuting generators."
        if has_multiplet
        else ("No nested EPI multiplet is present; Ψ is a single complex " "scalar.")
    )
    return NonAbelianCandidateAudit(
        route="u5_nested_epi_multiplet",
        status=status,
        obstruction=obstruction,
        nodal_derivable=has_multiplet,
        grammar_compatible=True,
        requires_external_labels=False,
        has_multiplet=has_multiplet,
        has_canonical_connection=False,
        has_noncommuting_generators=False,
        evidence={
            "nested_epi_nodes": evidence["nested_epi_nodes"],
            "internal_field_rank": evidence["internal_field_rank"],
            "connection_scalar": evidence["connection_scalar"],
        },
    )


def _audit_thol_remesh_internal_space(
    evidence: dict[str, Any],
) -> NonAbelianCandidateAudit:
    has_history = evidence["operator_history_events"] > 0
    status = (
        "OPEN_HISTORY_WITHOUT_CANONICAL_GENERATORS"
        if has_history
        else "FAILED_NO_OPERATOR_INTERNAL_SPACE"
    )
    obstruction = (
        "Operator history exists, but the 13 canonical operators do not "
        "expose "
        "a derived non-commuting generator algebra for gauge transport."
        if has_history
        else (
            "No THOL/REMESH-derived internal state space is recorded on " "the graph."
        )
    )
    return NonAbelianCandidateAudit(
        route="thol_remesh_internal_space",
        status=status,
        obstruction=obstruction,
        nodal_derivable=has_history,
        grammar_compatible=True,
        requires_external_labels=False,
        has_multiplet=has_history,
        has_canonical_connection=False,
        has_noncommuting_generators=False,
        evidence={
            "operator_history_events": evidence["operator_history_events"],
            "grammar_rules_satisfied": evidence["grammar_rules_satisfied"],
            "grammar_rules_total": evidence["grammar_rules_total"],
        },
    )


def _audit_cycle_basis_bundle(evidence: dict[str, Any]) -> NonAbelianCandidateAudit:
    has_cycle_basis = evidence["cycle_rank"] > 1
    status = (
        "FAILED_BASIS_DEPENDENT_EXTERNAL_SELECTION"
        if has_cycle_basis
        else "FAILED_INSUFFICIENT_CYCLE_RANK"
    )
    obstruction = (
        "A multi-cycle basis exists, but choosing generators/orientations as "
        "a gauge algebra is basis-dependent and not derived from nodal "
        "dynamics."
        if has_cycle_basis
        else (
            "The graph does not contain enough independent cycles for a "
            "cycle-bundle route."
        )
    )
    return NonAbelianCandidateAudit(
        route="cycle_basis_bundle",
        status=status,
        obstruction=obstruction,
        nodal_derivable=False,
        grammar_compatible=True,
        requires_external_labels=has_cycle_basis,
        has_multiplet=has_cycle_basis,
        has_canonical_connection=False,
        has_noncommuting_generators=False,
        evidence={
            "cycle_rank": evidence["cycle_rank"],
            "cycle_count_detected": evidence["cycle_count_detected"],
            "curvature_scalar": evidence["curvature_scalar"],
        },
    )


def _internal_field_rank(psi: dict[Any, complex]) -> int:
    # The canonical Ψ field is a scalar complex value at each node.
    return 1 if psi else 0


def _is_real_scalar(value: Any) -> bool:
    return isinstance(value, (int, float))


def _cycle_rank(G: Any) -> int:
    if nx is None:  # pragma: no cover
        return 0
    if G.is_directed():
        undirected = G.to_undirected()
    else:
        undirected = G
    components = nx.number_connected_components(undirected)
    return int(undirected.number_of_edges() - undirected.number_of_nodes() + components)


def _count_nested_epi_nodes(G: Any) -> int:
    count = 0
    for node in G.nodes():
        epi = G.nodes[node].get("EPI")
        if isinstance(epi, dict) and len(epi) > 1:
            count += 1
        elif isinstance(epi, (list, tuple)) and len(epi) > 1:
            count += 1
    return count


def _count_operator_history_events(G: Any) -> int:
    total = 0
    graph_history = G.graph.get("operator_history", ())
    if isinstance(graph_history, (list, tuple)):
        total += len(graph_history)
    for node in G.nodes():
        history = G.nodes[node].get("operator_history", ())
        if isinstance(history, (list, tuple)):
            total += len(history)
    return total
