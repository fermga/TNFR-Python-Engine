"""Compatible-support capacity diffusion and local antipodal phase response.

The compatible graph belongs to the capacity proposal and can differ from
the full pressure graph. Exact detached balances, current readouts and actual
operator admission are separate. No function below evolves EPI or a graph.
"""

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_THETA
from ..types import Glyph
from ._cycle_algebra import Matrix, Vector, dot, ordered_vector
from ._exact_linear_algebra import exact_square_matrix_product
from .support_transport import (
    SupportTransportSnapshot,
    _energy,
    _laplacian,
    observe_support_transport,
)

__all__ = [
    "CompatibleCapacityBalance",
    "CouplingSupportObservation",
    "derive_compatible_capacity_balance",
    "observe_coupling_support",
    "AntipodalRegionPhaseBalance",
    "AntipodalRegionPhaseResponse",
    "derive_antipodal_region_phase_balance",
    "observe_antipodal_region_phase_response",
]


def _ordered(values, label):
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered sequence")
    try:
        return tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered sequence") from exc


def _node_order(nodes):
    result = _ordered(nodes, "nodes")
    if not result:
        raise ValueError(
            "compatible-support observation requires a nonempty node order"
        )
    if len(set(result)) != len(result):
        raise ValueError("node IDs must be unique")
    return result


def _neighbor_rows(neighbors, size, *, allow_empty):
    rows = tuple(
        _ordered(row, "neighbor row") for row in _ordered(neighbors, "neighbors")
    )
    if len(rows) != size:
        raise ValueError("neighbor rows must match the node order")
    for i, row in enumerate(rows):
        if not allow_empty and not row:
            raise ValueError("every target requires a nonempty compatible-neighbor row")
        if any(type(j) is not int or not 0 <= j < size or j == i for j in row) or len(
            set(row)
        ) != len(row):
            raise ValueError(
                "neighbor rows must contain distinct loop-free valid indices"
            )
    if any(i not in rows[j] for i, row in enumerate(rows) for j in row):
        raise ValueError("compatible-neighbor rows must be reciprocal")
    return rows


def _components(rows):
    remaining, result = set(range(len(rows))), []
    while remaining:
        first = min(remaining)
        reached, pending = {first}, [first]
        while pending:
            i = pending.pop()
            new = set(rows[i]) - reached
            reached.update(new)
            pending.extend(new)
        remaining.difference_update(reached)
        result.append(tuple(sorted(reached)))
    return tuple(result)


@dataclass(frozen=True)
class CompatibleCapacityBalance:
    """One exact Jacobi capacity map on a reciprocal nonempty-row graph."""

    nodes: tuple
    neighbors: tuple
    capacity: Vector
    coupling_factor: Fraction
    degrees: tuple
    components: tuple
    component_means_before: Vector
    component_means_after: Vector
    component_mean_identity_residuals: Vector
    capacity_gradient: Vector
    capacity_after: Vector
    is_fixed: bool
    component_constant: bool
    energy_before: Fraction
    energy_after: Fraction
    drift_term: Fraction
    quadratic_term: Fraction
    energy_change: Fraction
    identity_residual: Fraction
    gradient_norm_squared: Fraction
    strict_drop_upper_bound: Fraction


def derive_compatible_capacity_balance(
    *,
    nodes,
    neighbors,
    capacity,
    coupling_factor,
) -> CompatibleCapacityBalance:
    """Derive component means, fixed points and a capacity Dirichlet balance.

    For the supplied simple reciprocal support C, D is its unweighted degree
    matrix and B its unit-conductance Laplacian. Simultaneous all-target UM has
    nu'=(I-gamma*D^-1*B)nu in exact arithmetic. With 0<gamma<1 it fixes exactly
    the fields constant on each connected compatible component. Every such
    component preserves its degree-weighted capacity mean.

    For E_C=.5*nu^T*B*nu and S=(B*nu)^T*D^-1*(B*nu),
    E_C'-E_C=-gamma*S+.5*gamma^2*(D^-1*B*nu)^T*B*(D^-1*B*nu)
    <=-gamma*(1-gamma)*S. The last inequality uses the normalized Laplacian's
    upper spectral bound two. The energy kernels are shared with nodal transport,
    but the observed vector here is capacity, not an assigned graph EPI field.

    This is not sequential UM, binary64 arithmetic, a phase-invariance proof or
    an admission certificate. Empty compatible rows are rejected, rather than
    inventing admissible identity updates for isolated targets.
    """
    ordered = _node_order(nodes)
    rows = _neighbor_rows(neighbors, len(ordered), allow_empty=False)
    nu = ordered_vector(capacity, "capacity")
    gamma = exact_or_represented_real(coupling_factor, "coupling_factor")
    if len(nu) != len(ordered) or any(value < 0 for value in nu):
        raise ValueError("nonnegative capacities must match the node order")
    if not 0 < gamma < 1:
        raise ValueError("coupling factor must be strictly between zero and one")
    degrees = tuple(len(row) for row in rows)
    conductance = tuple((i, j, Fraction(1)) for i, row in enumerate(rows) for j in row)
    bnu = _laplacian(conductance, nu)
    gradient = tuple(
        -value / degree for value, degree in zip(bnu, degrees, strict=True)
    )
    after = tuple(
        value + gamma * delta for value, delta in zip(nu, gradient, strict=True)
    )
    components = _components(rows)

    def means(values):
        return tuple(
            sum((degrees[i] * values[i] for i in component), Fraction(0))
            / sum(degrees[i] for i in component)
            for component in components
        )

    means_before, means_after = means(nu), means(after)
    mean_residuals = tuple(
        right - left for left, right in zip(means_before, means_after)
    )
    fixed = after == nu
    constant = all(
        all(nu[i] == nu[component[0]] for i in component) for component in components
    )
    before_energy, after_energy = _energy(conductance, nu), _energy(conductance, after)
    norm_squared = sum(
        (value**2 / degree for value, degree in zip(bnu, degrees)), Fraction(0)
    )
    drift = gamma * dot(bnu, gradient)
    quadratic = gamma**2 * _energy(conductance, gradient)
    change = after_energy - before_energy
    identity = change - drift - quadratic
    upper = -gamma * (1 - gamma) * norm_squared
    if (
        fixed != constant
        or identity
        or any(mean_residuals)
        or change > upper
        or (not fixed and change >= 0)
        or any(value < 0 for value in after)
    ):
        raise RuntimeError(
            "exact compatible-capacity balance lost a component or energy identity"
        )
    return CompatibleCapacityBalance(
        ordered,
        rows,
        nu,
        gamma,
        degrees,
        components,
        means_before,
        means_after,
        mean_residuals,
        gradient,
        after,
        fixed,
        constant,
        before_energy,
        after_energy,
        drift,
        quadratic,
        change,
        identity,
        norm_squared,
        upper,
    )


@dataclass(frozen=True)
class CouplingSupportObservation:
    """Current materialized U3 rows; no execution or full-admission seal.

    Rows, components and excluded edges use indices in snapshot.nodes. Blocked
    targets instead retain the original node IDs. Compatible rows preserve the
    runtime neighbor insertion order, including supported zero-weight edges.
    """

    snapshot: SupportTransportSnapshot
    phase: Vector
    ordered_support_neighbors: tuple
    compatible_neighbors: tuple
    excluded_edges: tuple
    components: tuple
    blocked_targets: tuple
    effective_phase_limit: Fraction
    coupling_factor: Fraction
    balance: CompatibleCapacityBalance | None


def observe_coupling_support(G) -> CouplingSupportObservation:
    """Read the actual U3 capacity graph without changing nodes or caches.

    Scope is a nonempty simple undirected loop-free graph with enabled UM capacity
    synchronization and one actual factor in (0,1). The production U3 resolver
    selects each row from current phases and graph limits. Materialized reciprocity
    is checked explicitly; common theoretical gates do not replace that check.

    All supported edges participate in this selection, independently of transport
    weight. An empty selected row is retained as a blocked target and causes
    balance=None; it never becomes an admitted identity update. A present balance
    is only an exact simultaneous capacity model. EPI thresholds, all other live
    gates, phase proposals, functional-link changes and runtime rounding remain
    independent obligations.
    """
    if G.is_directed() or G.is_multigraph():
        raise ValueError(
            "coupling-support observation requires a simple undirected graph"
        )
    nodes = _node_order(tuple(G.nodes))
    if any(node in G.neighbors(node) for node in nodes):
        raise ValueError("coupling-support observation requires loop-free support")
    if not bool(G.graph.get("UM_SYNC_VF", True)):
        raise ValueError(
            "coupling-support capacity analysis requires enabled UM_SYNC_VF"
        )
    from ..operators._phase_gate import resolve_u3_phase_neighbors
    from ..operators.factor_contracts import resolve_runtime_operator_factors

    factors = resolve_runtime_operator_factors(
        G.graph.get("GLYPH_FACTORS"), Glyph.UM, G.graph
    )
    gamma = exact_or_represented_real(factors["UM_vf_sync"], "UM_vf_sync")
    if not 0 < gamma < 1:
        raise ValueError("coupling factor must be strictly between zero and one")
    snapshot = observe_support_transport(G)
    if snapshot.nodes != nodes:
        raise RuntimeError("coupling-support observation changed the node order")
    index = {node: i for i, node in enumerate(nodes)}
    phases = tuple(
        finite_represented_real(
            get_attr(
                G.nodes[node], ALIAS_THETA, None, conv=lambda value: value, strict=True
            ),
            "phase",
        )
        for node in nodes
    )
    raw_rows = tuple(tuple(G.neighbors(node)) for node in nodes)
    selections = tuple(
        resolve_u3_phase_neighbors(
            G.graph,
            phases[i][0],
            row,
            phase_getter=lambda neighbor: phases[index[neighbor]][0],
            operator_code="UM",
            require_compatible=False,
        )
        for i, row in enumerate(raw_rows)
    )
    limits = tuple(
        finite_represented_real(item.effective_limit, "effective phase limit")[1]
        for item in selections
    )
    if any(limit != limits[0] for limit in limits):
        raise ValueError("all targets must use one common effective phase limit")
    rows = _neighbor_rows(
        tuple(tuple(index[node] for node in item.neighbors) for item in selections),
        len(nodes),
        allow_empty=True,
    )
    ordered_support = tuple(tuple(index[node] for node in row) for row in raw_rows)
    excluded = tuple(
        sorted(
            (i, j)
            for i, row in enumerate(ordered_support)
            for j in row
            if i < j and j not in rows[i]
        )
    )
    blocked = tuple(nodes[i] for i, row in enumerate(rows) if not row)
    balance = (
        None
        if blocked
        else derive_compatible_capacity_balance(
            nodes=nodes,
            neighbors=rows,
            capacity=snapshot.capacity,
            coupling_factor=gamma,
        )
    )
    return CouplingSupportObservation(
        snapshot,
        tuple(value[1] for value in phases),
        ordered_support,
        rows,
        excluded,
        _components(rows),
        blocked,
        limits[0],
        gamma,
        balance,
    )


@dataclass(frozen=True)
class AntipodalRegionPhaseBalance:
    """Exact UM/IL phase Jacobians in one declared two-triangle chart.

    This detached reference identifies neither a live graph nor a binary64
    derivative. Its strict certificate concerns an eigenvalue above one in
    the real local phase model; it is not a complete-runtime stability claim.
    """

    coupling_phase_factor: Fraction
    coherence_phase_factor: Fraction
    coupling_matrix: Matrix
    coherence_matrix: Matrix
    product_matrix: Matrix
    trace: Fraction
    determinant: Fraction
    det_identity_minus_product: Fraction
    strict_expansion_certificate: bool


def derive_antipodal_region_phase_balance(
    *,
    coupling_phase_factor,
    coherence_phase_factor,
) -> AntipodalRegionPhaseBalance:
    """Derive the exact local phase response of all-target UM followed by IL.

    The fixed support has triangles (0,1,2) and (3,4,5), and bridge (2,3).
    Around the mathematical phases (0,0,0,pi,pi,pi), the tangent embedding
    is (a,a,b,-b,-a,-a). Within a sufficiently small lifted chart, U3 keeps
    precisely the two triangles compatible. All-target bidirectional UM
    includes the target in each triangle's phasor mean. Every receiving node
    gets three identical exact-real phase proposals; the production merge
    averages their shortest-arc displacements. Missing cross-region links
    remain incompatible, and the already complete triangles gain no links.

    Write t for the UM phase factor and alpha for the IL phase factor.
    UM's local mean is C=Arg(2*exp(i*a)+exp(i*b)), giving Jacobian
    U=((1-t/3,t/3),(2*t/3,1-2*t/3)). IL reads the full graph, including the
    bridge: its bridge mean is Arg(2*exp(i*a)-exp(-i*b)), so its Jacobian
    is M=((1-alpha/2,alpha/2),(2*alpha,1)). Both stages are simultaneous;
    their composition has P=M*U, never U*M.

    Reflection and equal-interior symmetries preserve this two-coordinate
    nonlinear chart locally, where phasor resultants and circular lifts are
    regular. They do not give an invariant neighborhood under repetition.
    The identity det(I-P)=-alpha**2*(1-t)-2*alpha*t/3 is negative for alpha>0,
    hence the monic real characteristic polynomial has a root above one.
    At alpha=0 the constant direction is neutral. No numerical eigenvalue
    calculation or finite-difference limit is needed for these conclusions.

    Both factors lie in [0,1]; t=0 is only a detached algebraic control,
    excluded by canonical UM admission. Rational inputs remain exact and
    other real inputs retain their represented binary64 value. Exact pi in
    the base chart is not identified with represented math.pi. Actual phase
    gates, EPI and capacity evolution, rounding and full-word admission are
    outside this reference. No nonlinear trajectory is executed here.
    """
    from .phase_response import derive_phase_response

    t = exact_or_represented_real(coupling_phase_factor, "coupling_phase_factor")
    alpha = exact_or_represented_real(coherence_phase_factor, "coherence_phase_factor")
    if not 0 <= t <= 1 or not 0 <= alpha <= 1:
        raise ValueError("phase factors must lie in [0, 1]")
    signs = (1, 1, 1, -1, -1, -1)
    gram = tuple(tuple(Fraction(left * right) for right in signs) for left in signs)
    compatible = ((1, 2), (0, 2), (0, 1), (4, 5), (3, 5), (3, 4))
    closed = tuple((i, *row) for i, row in enumerate(compatible))
    full = ((1, 2), (0, 2), (0, 1, 3), (2, 4, 5), (3, 5), (3, 4))
    coupling_full = derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=closed,
        receiver_sources=closed,
        phase_factor=t,
    )
    coherence_full = derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=full,
        receiver_sources=tuple((i,) for i in range(6)),
        phase_factor=alpha,
    )

    def restrict(matrix):
        basis = ((1, 1, 0, 0, -1, -1), (0, 0, 1, -1, 0, 0))
        columns = tuple(tuple(dot(row, vector) for row in matrix) for vector in basis)
        for column in columns:
            a, b = column[0], column[2]
            if column != (a, a, b, -b, -a, -a):
                raise RuntimeError(
                    "shared phase response left the antipodal tangent subspace"
                )
        return tuple(tuple(column[i] for column in columns) for i in (0, 2))

    coupling = restrict(coupling_full.jacobian)
    coherence = restrict(coherence_full.jacobian)
    product = exact_square_matrix_product(coherence, coupling)
    trace = product[0][0] + product[1][1]
    determinant = product[0][0] * product[1][1] - product[0][1] * product[1][0]
    identity_det = (1 - product[0][0]) * (1 - product[1][1]) - product[0][1] * product[
        1
    ][0]
    expected = -(alpha**2) * (1 - t) - 2 * alpha * t / 3
    strict = identity_det < 0
    if (
        identity_det != expected
        or identity_det != 1 - trace + determinant
        or strict != (alpha > 0)
    ):
        raise RuntimeError(
            "exact antipodal phase response lost its characteristic identity"
        )
    return AntipodalRegionPhaseBalance(
        t,
        alpha,
        coupling,
        coherence,
        product,
        trace,
        determinant,
        identity_det,
        strict,
    )


@dataclass(frozen=True)
class AntipodalRegionPhaseResponse:
    """One exact tangent action and its six-coordinate quadratic diagnostic.

    Energy means half the squared norm of (a,a,b,-b,-a,-a), equivalently
    half the diag(4,2) norm of (a,b). It is neither EPI energy nor a declared
    Lyapunov function of the nonlinear engine.
    """

    reference: AntipodalRegionPhaseBalance
    before: Vector
    after_coupling: Vector
    after_coherence: Vector
    embedded_before: Vector
    embedded_after: Vector
    energy_before: Fraction
    energy_after: Fraction
    energy_change: Fraction


def observe_antipodal_region_phase_response(
    reference,
    *,
    interior,
    bridge,
) -> AntipodalRegionPhaseResponse:
    """Apply the rederived exact Jacobians to a declared tangent direction.

    Inputs are tangent coordinates, not finite graph phases. Large values
    are valid directions but imply no admissible finite perturbation. Public
    reference caches are rebuilt from the two factors before any action.
    """
    if not isinstance(reference, AntipodalRegionPhaseBalance):
        raise TypeError("reference must be an AntipodalRegionPhaseBalance")
    reference = derive_antipodal_region_phase_balance(
        coupling_phase_factor=reference.coupling_phase_factor,
        coherence_phase_factor=reference.coherence_phase_factor,
    )
    before = ordered_vector((interior, bridge), "phase tangent")
    coupled = tuple(dot(row, before) for row in reference.coupling_matrix)
    after = tuple(dot(row, coupled) for row in reference.coherence_matrix)

    def embed(values):
        a, b = values
        return a, a, b, -b, -a, -a

    embedded_before, embedded_after = embed(before), embed(after)
    energy_before = dot(embedded_before, embedded_before) / 2
    energy_after = dot(embedded_after, embedded_after) / 2
    if after != tuple(dot(row, before) for row in reference.product_matrix):
        raise RuntimeError("exact antipodal phase action lost composition order")
    return AntipodalRegionPhaseResponse(
        reference,
        before,
        coupled,
        after,
        embedded_before,
        embedded_after,
        energy_before,
        energy_after,
        energy_after - energy_before,
    )
