"""Minimal joint EPI/potential observations of a held fine nodal model.

Geometry uses exact paths on the per-edge lengths materialized by the shared
reader. A separate residual retains floating-path/kernel/readout arithmetic.
The common affine realization owner derives the coordinates and decoder;
no additional state law, pressure fit, solver or graph evolution is introduced.
"""

from dataclasses import dataclass
from fractions import Fraction

import networkx as nx

from .._exact_time import finite_represented_real
from ._cycle_algebra import Matrix, Vector, dot
from ._edge_semantics import structural_path_weight
from .canonical import (
    CoherenceLengthEstimate,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
    observe_phase_curvature,
)
from .epi_memory import (
    AffineNodalRealization,
    ForcedSupportClosure,
    observe_affine_nodal_realization,
    observe_forced_support_closure,
)
from .forced_support import derive_forced_support_balance
from .forcing_realization import NonEpiForcingObservation, capture_non_epi_forcing
from .hybrid_operator_stability import _exact_matrix_product
from .phase_curvature import PhaseCurvatureObservation

__all__ = [
    "ExactPotentialGeometry",
    "ForcedSupportGeometry",
    "observe_forced_support_geometry",
    "PressureObservationWitness",
    "ForcedSupportTetradDependencies",
    "observe_forced_support_tetrad_dependencies",
]


@dataclass(frozen=True)
class ExactPotentialGeometry:
    """Detached inverse-square geometry of materialized structural edge lengths.

    Edge endpoints are indices into ``nodes``. Multigraph edges have already
    been reduced by the shared minimum-length rule, independently of summed
    transport conductance. Rational path addition happens after that reader's
    binary64 edge materialization, not after a floating distance/kernel was
    computed. Zero-distance and unreachable pairs follow the canonical zero
    contribution policy rather than being inverted.
    """

    nodes: tuple
    directed: bool
    edge_lengths: tuple[tuple[int, int, Fraction], ...]
    distances: tuple[tuple[Fraction | None, ...], ...]
    kernel: Matrix
    length_policy: str


@dataclass(frozen=True)
class ForcedSupportGeometry:
    """Closed linear observation, with the full held-source and field boundary.

    Minimality concerns all-state linear realizations retaining the declared
    EPI means and averaged *model* potential. It is not complete tetrad
    reconstruction, a new nodal ontology, or identification of stored pressure
    with the model. Potential corrections use pressure, without capacity
    factors; the field arithmetic residual is kept separate from them.
    """

    fine_capture: NonEpiForcingObservation
    closure: ForcedSupportClosure
    realization: AffineNodalRealization
    geometry: ExactPotentialGeometry
    potential_rows: Matrix
    potential_offset: Vector
    model_pressure: Vector
    model_potential: Vector
    model_output: Vector
    fresh_pressure_potential_defect: Vector
    stored_pressure_potential_defect: Vector
    field_arithmetic_residual: Vector
    observed_potential: Vector
    exact_identity_checks: tuple[str, ...]
    scope: str


@dataclass(frozen=True)
class PressureObservationWitness:
    """A hidden fine-form change invisible to C but visible to model pressure."""

    column_index: int
    hidden_delta: Vector
    pressure_difference: Vector


@dataclass(frozen=True)
class ForcedSupportTetradDependencies:
    """Held phase read-outs and exact pressure-input sufficiency for the tetrad.

    ``pressure_observation_closed`` concerns the entire fine model pressure,
    not minimality of any nonlinear scalar read-out. A nonzero residual does
    not prove failure of a coherence-length factorization. The observed
    length uses stored pressure, runtime floating paths, and the full fit/
    selected-spectrum provenance. It is not a model forecast from C alone.
    """

    geometry: ForcedSupportGeometry
    phase_observation: PhaseCurvatureObservation
    mean_phase_gradient: Vector
    mean_phase_curvature: tuple[Fraction | None, ...]
    observed_coherence_length: CoherenceLengthEstimate
    pressure_rows: Matrix
    pressure_decoder: Matrix
    pressure_hidden_residual: Matrix
    pressure_observation_closed: bool
    decoded_model_pressure: Vector
    unresolved_model_pressure: Vector
    pressure_witness: PressureObservationWitness | None
    exact_identity_checks: tuple[str, ...]
    scope: str


def _pressure_rows(closure):
    """Convert the held EPI-rate generator to its fine pressure observation."""
    return tuple(
        tuple(-value / nu for value in row)
        for row, nu in zip(
            closure.micro_generator, closure.reference.source.capacity, strict=True
        )
    )


def _potential_geometry(graph, nodes):
    """Reuse edge-channel semantics and NetworkX shortest paths exactly."""
    if tuple(graph.nodes) != nodes:
        raise ValueError("geometry node order differs from the nodal capture")
    index = {node: i for i, node in enumerate(nodes)}
    read = structural_path_weight(graph)
    edges = tuple(
        (index[left], index[right], Fraction.from_float(read(left, right, data)))
        for left in nodes
        for right, data in graph.adj[left].items()
    )
    directed = bool(graph.is_directed())
    metric_graph = nx.DiGraph() if directed else nx.Graph()
    metric_graph.add_nodes_from(range(len(nodes)))
    metric_graph.add_weighted_edges_from(edges, weight="length")
    paths = dict(nx.all_pairs_dijkstra_path_length(metric_graph, weight="length"))
    distances = tuple(
        tuple(
            Fraction(paths[i][j]) if j in paths[i] else None for j in range(len(nodes))
        )
        for i in range(len(nodes))
    )
    zero = Fraction(0)
    kernel = tuple(
        tuple(
            zero if i == j or value is None or value == 0 else 1 / value**2
            for j, value in enumerate(row)
        )
        for i, row in enumerate(distances)
    )
    return ExactPotentialGeometry(
        nodes=nodes,
        directed=directed,
        edge_lengths=edges,
        distances=distances,
        kernel=kernel,
        length_policy=(
            "shared structural_path_weight: explicit length, legacy weight, then unit; "
            "parallel minimum; exact rational paths on materialized binary64 edges; "
            "self, zero-distance and unreachable pairs contribute zero"
        ),
    )


def observe_forced_support_geometry(
    graph,
    blocks,
    *,
    max_rank_calls=4096,
) -> ForcedSupportGeometry:
    """Derive one minimal joint EPI/potential realization without evolving G.

    The bounded default NumPy forcing-capture domain, positive connected
    reciprocal transport, positive capacities and EPI coefficient are
    inherited from the existing owners. The fine law holds phase, capacity,
    support, edge lengths and coefficients fixed. Nonclosed EPI partitions
    are allowed: the shared realization retains the required extra rows.

    If A=e*diag(nu)*L and f is the independently captured pressure source,
    pressure has linear part -diag(1/nu)*A, not -A at heterogeneous capacity.
    Seed the shared algorithm with (R; -R*K*diag(1/nu)*A) and known offset
    (0; R*K*f). All arithmetic after edge/phase materialization is rational.
    The separate canonical field call is an observed numerical readout;
    finite conversion/path/kernel/summation defects remain in its residual.
    No emitted record has causal runtime or prospective-evaluation provenance.
    """
    capture = capture_non_epi_forcing(graph)
    reference = derive_forced_support_balance(
        capture.snapshot, epi_weight=capture.epi_weight, forcing=capture.forcing
    )
    closure = observe_forced_support_closure(reference, blocks)
    geometry = _potential_geometry(graph, capture.snapshot.nodes)
    pressure_rows = _pressure_rows(closure)
    rk = _exact_matrix_product(closure.projection, geometry.kernel)
    rows = _exact_matrix_product(rk, pressure_rows)
    offset = tuple(dot(row, capture.forcing) for row in rk)
    m = len(closure.blocks)
    realization = observe_affine_nodal_realization(
        reference,
        (*closure.projection, *rows),
        output_offset=((Fraction(0),) * m + offset),
        max_rank_calls=max_rank_calls,
    )
    model_pressure = tuple(
        capture.epi_weight * gradient + force
        for gradient, force in zip(
            capture.snapshot.epi_gradient, capture.forcing, strict=True
        )
    )
    model_potential = tuple(dot(row, model_pressure) for row in geometry.kernel)
    model_output = tuple(dot(row, model_potential) for row in closure.projection)

    def field_defect(pressure):
        return tuple(dot(row, pressure) for row in rk)

    kernel_defect = field_defect(capture.kernel_pressure_defect)
    stored_defect = field_defect(capture.stored_pressure_residual)
    observed = compute_structural_potential(graph, alpha=2.0, landmark_ratio=None)
    fine_observed = tuple(
        finite_represented_real(observed[node], f"potential[{i}]")[1]
        for i, node in enumerate(capture.snapshot.nodes)
    )
    observed_potential = tuple(dot(row, fine_observed) for row in closure.projection)
    exact_stored_output = field_defect(capture.snapshot.stored_pressure)
    arithmetic_defect = tuple(
        a - b for a, b in zip(observed_potential, exact_stored_output, strict=True)
    )
    reconstructed = tuple(
        sum(values, Fraction(0))
        for values in zip(
            model_output, kernel_defect, stored_defect, arithmetic_defect, strict=True
        )
    )
    checks = {
        "joint output includes original EPI means": realization.output_state[:m]
        == closure.projected_epi,
        "joint output retains affine potential offset": realization.output_state[m:]
        == model_output,
        "joint EPI rates match original held model": realization.output_rate[:m]
        == closure.projected_nodal_rate,
        "potential rows retain inverse-capacity pressure conversion": tuple(
            dot(row, closure.epi) + force
            for row, force in zip(pressure_rows, capture.forcing, strict=True)
        )
        == model_pressure,
        "field defects telescope without capacity factors": reconstructed
        == observed_potential,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(f"joint geometry identity failure: {failures}")
    return ForcedSupportGeometry(
        fine_capture=capture,
        closure=closure,
        realization=realization,
        geometry=geometry,
        potential_rows=rows,
        potential_offset=offset,
        model_pressure=model_pressure,
        model_potential=model_potential,
        model_output=model_output,
        fresh_pressure_potential_defect=kernel_defect,
        stored_pressure_potential_defect=stored_defect,
        field_arithmetic_residual=arithmetic_defect,
        observed_potential=observed_potential,
        exact_identity_checks=tuple(checks),
        scope=(
            "Minimal all-state linear realization of EPI means and averaged inverse-square "
            "model potential with fixed source, capacity, support and materialized edge lengths. "
            "Exact represented-coefficient/path model and separately measured field residuals; "
            "no complete tetrad, evolving fine law, partition selection, runtime or physical "
            "emergence certificate."
        ),
    )


def observe_forced_support_tetrad_dependencies(
    graph,
    blocks,
    *,
    max_rank_calls=4096,
) -> ForcedSupportTetradDependencies:
    """Audit the remaining read-outs of one held EPI/potential realization.

    The existing geometry observer owns the exact nodal model and C/T state.
    Set M=-diag(1/nu)*A, Dp=M*T and E=M-Dp*C. Then p=Dp*s+f+E*x;
    all-state fine model pressure is determined by s iff E=0. A nonzero
    column gives delta=(I-T*C)e_j with C*delta=0 and M*delta!=0. This is
    independent of whether the current x happens to hide the residual.

    In this connected positive-capacity/EPI domain, ker(M)=span(1), while
    C retains the partition means. Therefore E=0 iff C has full fine rank.
    This excludes proper linear compression retaining both all pressure and
    those means. It does not exclude scalar nonlinear read-out compression.

    Phase/support are held input: the shared phase observer supplies constant
    read-outs, not a phase evolution law or a phase inferred from form. An
    undefined curvature remains None in any average containing it. The xi
    observer retains numerical fit/fallback/unavailable status on actual
    stored pressure; its paths are not replaced by rational model distances.
    A detached graph copy confines its graph-local spectral cache writes.
    No solver, event, parameter selection or prospective evidence is created.
    """
    detached = graph.copy()
    geometry = observe_forced_support_geometry(
        detached,
        blocks,
        max_rank_calls=max_rank_calls,
    )
    closure, realization = geometry.closure, geometry.realization
    phase = observe_phase_curvature(detached)
    if phase.nodes != geometry.fine_capture.snapshot.nodes:
        raise RuntimeError("phase node order differs from the nodal capture")
    if phase.primitive_phases != tuple(map(float, geometry.fine_capture.phase)):
        raise RuntimeError("phase read-out differs from the held primitive phases")
    gradients = tuple(
        finite_represented_real(row.gradient, "phase gradient")[1] for row in phase.rows
    )
    curvatures = tuple(
        (
            None
            if row.curvature is None
            else finite_represented_real(row.curvature, "phase curvature")[1]
        )
        for row in phase.rows
    )
    mean_gradient = tuple(dot(row, gradients) for row in closure.projection)
    mean_curvature = tuple(
        (
            None
            if any(
                weight and value is None
                for weight, value in zip(row, curvatures, strict=True)
            )
            else sum(
                (
                    weight * value
                    for weight, value in zip(row, curvatures, strict=True)
                    if weight
                ),
                Fraction(0),
            )
        )
        for row in closure.projection
    )
    length = estimate_coherence_length_with_provenance(detached)
    matrix = _pressure_rows(closure)
    product = _exact_matrix_product
    decoder = product(matrix, realization.right_inverse)
    reconstructed_rows = product(decoder, realization.observation)
    residual = tuple(
        tuple(left - right for left, right in zip(row, reconstructed, strict=True))
        for row, reconstructed in zip(matrix, reconstructed_rows, strict=True)
    )
    closed = not any(value for row in residual for value in row)
    decoded = tuple(
        dot(row, realization.reduced_state) + force
        for row, force in zip(decoder, geometry.fine_capture.forcing, strict=True)
    )
    unresolved = tuple(dot(row, closure.epi) for row in residual)
    checks = {
        "pressure decoder plus hidden residual": tuple(
            left + right for left, right in zip(decoded, unresolved, strict=True)
        )
        == geometry.model_pressure,
        "all-pressure and means require full fine linear state": closed
        == (realization.dimension == realization.full_state_dimension),
    }
    witness = None
    if not closed:
        n = realization.full_state_dimension
        j = next(j for j in range(n) if any(row[j] for row in residual))
        tc = product(realization.right_inverse, realization.observation)
        delta = tuple(Fraction(i == j) - tc[i][j] for i in range(n))
        difference = tuple(dot(row, delta) for row in matrix)
        checks["hidden witness preserves every retained coordinate"] = not any(
            dot(row, delta) for row in realization.observation
        )
        checks["hidden witness changes fine model pressure"] = any(
            difference
        ) and difference == tuple(row[j] for row in residual)
        witness = PressureObservationWitness(j, delta, difference)
    failures = tuple(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(f"tetrad dependency identity failure: {failures}")
    return ForcedSupportTetradDependencies(
        geometry=geometry,
        phase_observation=phase,
        mean_phase_gradient=mean_gradient,
        mean_phase_curvature=mean_curvature,
        observed_coherence_length=length,
        pressure_rows=matrix,
        pressure_decoder=decoder,
        pressure_hidden_residual=residual,
        pressure_observation_closed=closed,
        decoded_model_pressure=decoded,
        unresolved_model_pressure=unresolved,
        pressure_witness=witness,
        exact_identity_checks=tuple(checks),
        scope=(
            "Held fine phase/support read-outs and all-state linear model-pressure "
            "dependency test. Pressure closure suffices for pressure-based model read-outs "
            "with fixed side inputs, but is not necessary for xi alone. Observed xi uses "
            "actual stored pressure and runtime floating path bins, preserving fit/selected "
            "dimensionless spectrum/unavailable provenance. No model-to-stored identification, "
            "complete tetrad minimality, emergent phase, selected geometry or runtime forecast."
        ),
    )
