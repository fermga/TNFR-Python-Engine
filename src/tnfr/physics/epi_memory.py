r"""Observe closure and memory derived by projecting fixed nodal diffusion.

For ``x' = -A x``, let ``R`` be the reversible partition average, ``P``
its lift, and ``Q = I - P R``. Eliminating the unresolved coordinate ``Q x``
gives, in exact arithmetic,

    y' = -R A P y + f(t) + integral_0^t K(t-s) y(s) ds,
    K(t) = R A exp(-Q A Q t) Q A P,
    f(t) = -R A exp(-Q A Q t) Q x(0).

Every coefficient comes from the existing nodal generator and partition.
The implementation samples this identity on detached vectors using the shared
approximate matrix exponential. It observes numerical residuals, not certified
error enclosures or runtime trajectories. It neither advances a graph nor
introduces an operator or a REMESH delay law. The exact proof and the P4/P5
controls are in ``theory/DERIVED_EPI_MEMORY.md``. A separate exact observer
below admits a rebuilt fixed forced-support reference and tests all-state
affine closure, without an exponential or runtime evolution.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

import numpy as np

from .._exact_time import exact_or_represented_real
from ..dynamics._euler_kernel import euler_update
from ..mathematics.krylov import exact_rank
from ._cycle_algebra import Matrix, Vector, dot, ordered_vector
from ._exact_linear_algebra import exact_matrix_inverse
from .forced_support import ForcedSupportBalance
from .forced_support import _pattern as _forced_pattern
from .forced_support import _reference as _validated_forced_reference
from .hybrid_operator_stability import _exact_matrix_product
from .spectral_projectors import matrix_exponential
from .structural_morphism import _build_reversible_partition_geometry

__all__ = [
    "EpiMemorySample",
    "EpiMemoryObservation",
    "observe_epi_memory",
    "ForcedSupportClosure",
    "ForcedSupportClosureWitness",
    "observe_forced_support_closure",
    "ForcedSupportRealization",
    "ForcedSupportRealizationLevel",
    "observe_forced_support_realization",
    "AffineNodalRealization",
    "observe_affine_nodal_realization",
    "ForcedSupportEulerFrame",
    "ForcedSupportEulerPrediction",
    "predict_forced_support_realization_euler",
]


@dataclass(frozen=True)
class ForcedSupportClosureWitness:
    """Two detached scalar states with one projection and different model rates.

    The unit coordinate selects a deterministic algebraic witness. Neither
    state is claimed to satisfy a runtime chart, clipping or grammar domain.
    """

    hidden_column_index: int
    hidden_delta: Vector
    before_epi: Vector
    after_epi: Vector
    common_macro: Vector
    before_projected_rate: Vector
    after_projected_rate: Vector
    projected_rate_difference: Vector


@dataclass(frozen=True)
class ForcedSupportClosure:
    """Exact reversible projection of one rebuilt held affine nodal model.

    H and Hbar are diagonal with the displayed positive metric weights.
    ``hidden_to_macro`` is RAQ, ``macro_to_hidden`` is C=QAP, and
    ``instantaneous_kernel`` is K(0)=RAQAP. The exact identity
    Hbar*K(0)=C.T*H*C establishes its weighted nonnegative Gram form.
    All-state projected closure does not require Qb=0. Every rate uses the
    explicit held forcing, never a pressure inferred from stored DeltaNFR.
    Public fields are detached observations, not execution proof seals.
    """

    reference: ForcedSupportBalance
    nodes: tuple
    blocks: tuple[tuple, ...]
    metric_weights: Vector
    macro_metric_weights: Vector
    lift: Matrix
    projection: Matrix
    hidden_projector: Matrix
    micro_generator: Matrix
    affine_source: Vector
    macro_generator: Matrix
    projected_source: Vector
    hidden_affine_source: Vector
    hidden_to_macro: Matrix
    macro_to_hidden: Matrix
    instantaneous_kernel: Matrix
    weighted_instantaneous_kernel: Matrix
    coupling_gram: Matrix
    all_state_affine_closed: bool
    lifted_affine_subspace_invariant: bool
    epi: Vector
    projected_epi: Vector
    hidden_epi: Vector
    projected_nodal_rate: Vector
    affine_macro_rate: Vector
    hidden_rate_contribution: Vector
    current_mean: Fraction
    current_mean_rate: Fraction
    current_relative_error: Vector
    projected_relative_error: Vector
    current_relative_rate: Vector
    centered_rate_residual: Vector
    witness: ForcedSupportClosureWitness | None
    exact_identity_checks: tuple[str, ...]
    scope: str


def _exact_partition(nodes, blocks):
    """Validate the ordered proper-coarsening domain used by EPI geometry."""
    if isinstance(blocks, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("blocks must be an ordered sequence of ordered node blocks")
    try:
        raw = tuple(blocks)
    except TypeError as exc:
        raise TypeError("blocks must be an ordered sequence") from exc
    if not 2 <= len(raw) < len(nodes):
        raise ValueError("partition must strictly reduce to at least two blocks")
    result = []
    for block in raw:
        if isinstance(block, (str, bytes, bytearray, Mapping, Set)):
            raise TypeError("each block must be an ordered sequence of node IDs")
        try:
            values = tuple(block)
        except TypeError as exc:
            raise TypeError(
                "each block must be an ordered sequence of node IDs"
            ) from exc
        if not values:
            raise ValueError("partition blocks must be nonempty")
        result.append(values)
    flattened = tuple(node for block in result for node in block)
    if (
        len(flattened) != len(nodes)
        or len(set(flattened)) != len(nodes)
        or set(flattened) != set(nodes)
    ):
        raise ValueError("partition must contain each reference node exactly once")
    return tuple(result)


def _held_affine_nodal_model(reference, epi):
    """Rebuild the shared held model x'=-A*x+b from primitive source inputs."""
    ref = _validated_forced_reference(reference)
    n = len(ref.source.nodes)
    values = ref.source.epi if epi is None else ordered_vector(epi, "epi")
    if len(values) != n:
        raise ValueError("epi must match the reference node order")
    matrix = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for i, j, weight in ref.source.conductance:
        value = ref.epi_weight * ref.source.capacity[i] * weight / ref.strengths[i]
        matrix[i][i] += value
        matrix[i][j] -= value
    generator = tuple(tuple(row) for row in matrix)
    source = tuple(
        nu * force for nu, force in zip(ref.source.capacity, ref.forcing, strict=True)
    )
    return ref, generator, source, values


def observe_forced_support_closure(
    reference, blocks, *, epi=None
) -> ForcedSupportClosure:
    r"""Test exact all-state closure of y=Rx for x'=-Ax+b on held coefficients.

    The reference is rebuilt by the forced-support owner, requiring connected
    symmetric positive conductance, positive capacities and EPI coefficient.
    ``blocks`` contains ordered node IDs, is nonempty/disjoint/exhaustive and
    strictly reduces the state to at least two blocks. No tolerance is used.

    A=e*diag(nu)*D^-1*B, b=diag(nu)*F, H=diag(d/nu), Hbar=P.T*H*P,
    R=Hbar^-1*P.T*H, Q=I-PR and Abar=RAP. The projected equation closes
    for every scalar x iff RAQ=0. Reversibility makes this equivalent to
    C=QAP=0 and K(0)=RAQAP=0 via Hbar*K(0)=C.T*H*C. Qb may be nonzero:
    projected autonomy differs from invariance of the affine lifted subspace.

    If closure fails, the first nonzero column j of RAQ supplies h=Qe_j.
    States x and x+h have identical Rx and different R(-Ax+b). ``epi``
    supplies the base x, defaulting to the reference's captured EPI. These are
    detached unrestricted scalar witnesses, not executed or admissible graph
    states. The optional current-model centering is always u=x-mean_H(x)-z
    for this rebuilt reference and satisfies u'=-Au; no old target is used.
    No exponential, trajectory, fitted source or delay law is constructed.
    """
    ref, a, b, values = _held_affine_nodal_model(reference, epi)
    nodes = ref.source.nodes
    partition = _exact_partition(nodes, blocks)
    n, m = len(nodes), len(partition)
    zero = Fraction(0)
    one = Fraction(1)
    position = {node: index for index, node in enumerate(nodes)}
    membership = {
        position[node]: index for index, block in enumerate(partition) for node in block
    }
    h = ref.metric_weights
    hbar = tuple(
        sum((h[position[node]] for node in block), zero) for block in partition
    )
    p = tuple(
        tuple(one if membership[i] == a else zero for a in range(m)) for i in range(n)
    )
    r = tuple(
        tuple(h[i] / hbar[a] if membership[i] == a else zero for i in range(n))
        for a in range(m)
    )
    product = _exact_matrix_product
    pr = product(p, r)
    q = tuple(
        tuple((one if i == j else zero) - pr[i][j] for j in range(n)) for i in range(n)
    )
    ap, ra = product(a, p), product(r, a)
    abar = product(r, ap)
    raq, c = product(ra, q), product(q, ap)
    k0 = product(raq, ap)
    weighted_kernel = tuple(
        tuple(hbar[i] * value for value in row) for i, row in enumerate(k0)
    )
    weighted_c = tuple(tuple(h[i] * value for value in row) for i, row in enumerate(c))
    gram = product(tuple(zip(*c, strict=True)), weighted_c)

    def mv(matrix, vector):
        return tuple(dot(row, vector) for row in matrix)

    def is_zero(matrix):
        return not any(value for row in matrix for value in row)

    identity = tuple(tuple(one if i == j else zero for j in range(m)) for i in range(m))
    h_a = tuple(tuple(h[i] * a[i][j] for j in range(n)) for i in range(n))
    h_q = tuple(tuple(h[i] * q[i][j] for j in range(n)) for i in range(n))
    weighted_raq = tuple(
        tuple(hbar[i] * value for value in row) for i, row in enumerate(raq)
    )
    checks = {
        "RP=I": product(r, p) == identity,
        "Q^2=Q": product(q, q) == q,
        "RQ=0": is_zero(product(r, q)),
        "QP=0": is_zero(product(q, p)),
        "A1=0": mv(a, (one,) * n) == (zero,) * n,
        "HA=A^T H": h_a == tuple(zip(*h_a, strict=True)),
        "HQ=Q^T H": h_q == tuple(zip(*h_q, strict=True)),
        "Hbar RAQ=C^T H": weighted_raq == tuple(zip(*weighted_c, strict=True)),
        "K0=RA C": k0 == product(ra, c),
        "Hbar K0=C^T H C": weighted_kernel == gram,
        "RAQ=0 iff QAP=0 iff K0=0": is_zero(raq) == is_zero(c) == is_zero(k0),
    }
    rb, qb = mv(r, b), mv(q, b)
    projected, hidden = mv(r, values), mv(q, values)
    rate = tuple(force - drift for force, drift in zip(b, mv(a, values), strict=True))
    projected_rate = mv(r, rate)
    macro_rate = tuple(
        force - drift for force, drift in zip(rb, mv(abar, projected), strict=True)
    )
    hidden_rate = tuple(-value for value in mv(raq, values))
    checks["projected affine rate identity"] = projected_rate == tuple(
        left + right for left, right in zip(macro_rate, hidden_rate, strict=True)
    )
    pattern = _forced_pattern(ref, values)
    relative_rate = tuple(-value for value in mv(a, pattern.relative_error))
    centered_residual = tuple(
        observed - ref.mean_drift - expected
        for observed, expected in zip(rate, relative_rate, strict=True)
    )
    checks["current-model centered rate identity"] = not any(centered_residual)
    checks["current-model weighted mean drift"] = (
        dot(h, rate) == sum(h, zero) * ref.mean_drift
    )
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"exact forced closure identities failed: {failed}")
    closed = is_zero(raq)
    witness = None
    if not closed:
        column = next(j for j in range(n) if any(row[j] for row in raq))
        delta = tuple(row[column] for row in q)
        after = tuple(
            value + change for value, change in zip(values, delta, strict=True)
        )
        after_rate = mv(
            r,
            tuple(force - drift for force, drift in zip(b, mv(a, after), strict=True)),
        )
        difference = tuple(
            right - left for left, right in zip(projected_rate, after_rate, strict=True)
        )
        if (
            mv(r, after) != projected
            or mv(q, delta) != delta
            or not any(difference)
            or difference != tuple(-row[column] for row in raq)
        ):
            raise RuntimeError("exact same-projection different-rate witness failed")
        witness = ForcedSupportClosureWitness(
            column,
            delta,
            values,
            after,
            projected,
            projected_rate,
            after_rate,
            difference,
        )
    return ForcedSupportClosure(
        reference=ref,
        nodes=nodes,
        blocks=partition,
        metric_weights=h,
        macro_metric_weights=hbar,
        lift=p,
        projection=r,
        hidden_projector=q,
        micro_generator=a,
        affine_source=b,
        macro_generator=abar,
        projected_source=rb,
        hidden_affine_source=qb,
        hidden_to_macro=raq,
        macro_to_hidden=c,
        instantaneous_kernel=k0,
        weighted_instantaneous_kernel=weighted_kernel,
        coupling_gram=gram,
        all_state_affine_closed=closed,
        lifted_affine_subspace_invariant=closed and not any(qb),
        epi=values,
        projected_epi=projected,
        hidden_epi=hidden,
        projected_nodal_rate=projected_rate,
        affine_macro_rate=macro_rate,
        hidden_rate_contribution=hidden_rate,
        current_mean=pattern.mean,
        current_mean_rate=ref.mean_drift,
        current_relative_error=pattern.relative_error,
        projected_relative_error=mv(r, pattern.relative_error),
        current_relative_rate=relative_rate,
        centered_rate_residual=centered_residual,
        witness=witness,
        exact_identity_checks=tuple(checks),
        scope=(
            "Exact all-state projection test for one rebuilt held affine model on unrestricted "
            "real scalar EPI. H, its current-model relative profile and forcing are fixed. "
            "A detached algebraic witness need not satisfy a graph chart, clipping or grammar. "
            "No trajectory, runtime closure, empirical macro-entity or delay law is certified."
        ),
    )


@dataclass(frozen=True)
class ForcedSupportRealizationLevel:
    """One completely examined output-Krylov level O*A**power."""

    power: int
    rank_before: int
    rank_after: int
    candidate_rows: int
    selected_row_indices: tuple[int, ...]


@dataclass(frozen=True)
class ForcedSupportRealization:
    """Minimal all-state linear observation with known affine dynamics.

    s=C*x obeys s'=-G*s+C*b and reproduces y=R*x=D*s for the rebuilt
    held model. Rows of C may be signed/nonlocal coordinates, not nodes.
    Minimality is among linear observations retaining R for every scalar x;
    nonlinear or trajectory-restricted reductions are outside the statement.
    Resource diagnostics count this realization's rank/product calls and
    materialized coefficient bit lengths, not internal elimination costs or
    the separately rebuilt closure/reference work.
    """

    closure: ForcedSupportClosure
    dimension: int
    full_state_dimension: int
    extra_coordinates: int
    rank_progression: tuple[int, ...]
    level_records: tuple[ForcedSupportRealizationLevel, ...]
    selected_row_labels: tuple[tuple[int, int], ...]
    pivot_columns: tuple[int, ...]
    observation: Matrix
    right_inverse: Matrix
    reduced_generator: Matrix
    output_map: Matrix
    reduced_source: Vector
    epi: Vector
    reduced_state: Vector
    reduced_rate: Vector
    projected_state: Vector
    projected_rate: Vector
    reconstructed_projected_state: Vector
    reconstructed_projected_rate: Vector
    exact_identity_checks: tuple[str, ...]
    rank_calls: int
    max_rank_calls: int
    matrix_product_calls: int
    completed_levels: int
    stabilization_power: int
    max_coefficient_bits: int
    scope: str


@dataclass(frozen=True)
class AffineNodalRealization:
    """Minimal linear state reproducing supplied affine held-model outputs.

    For x'=-A*x+b and y=O*x+o0, the state s=C*x satisfies
    s'=-G*s+C*b and y=D*s+o0. O and o0 are declared observations, not
    derived physical laws. Constant offsets do not require a state coordinate.
    Minimality concerns all scalar x and autonomous linear state observations;
    trajectory-restricted, nonlinear and changing-model reductions differ.
    A rank-zero observation has empty state and constant output o0.
    """

    reference: ForcedSupportBalance
    output_rows: Matrix
    output_offset: Vector
    output_count: int
    output_rank: int
    dimension: int
    full_state_dimension: int
    extra_coordinates: int
    rank_progression: tuple[int, ...]
    level_records: tuple[ForcedSupportRealizationLevel, ...]
    selected_row_labels: tuple[tuple[int, int], ...]
    pivot_columns: tuple[int, ...]
    observation: Matrix
    right_inverse: Matrix
    reduced_generator: Matrix
    output_map: Matrix
    reduced_source: Vector
    epi: Vector
    reduced_state: Vector
    reduced_rate: Vector
    output_state: Vector
    output_rate: Vector
    reconstructed_output_state: Vector
    reconstructed_output_rate: Vector
    exact_identity_checks: tuple[str, ...]
    rank_calls: int
    max_rank_calls: int
    matrix_product_calls: int
    completed_levels: int
    stabilization_power: int
    max_coefficient_bits: int
    scope: str


def _check_realization_rank_budget(max_rank_calls):
    if type(max_rank_calls) is not int:
        raise TypeError("max_rank_calls must be a positive non-boolean integer")
    if max_rank_calls < 1:
        raise ValueError("max_rank_calls must be positive")


def observe_affine_nodal_realization(
    reference,
    output_rows,
    *,
    output_offset=None,
    epi=None,
    max_rank_calls=4096,
) -> AffineNodalRealization:
    r"""Realize y=O*x+o0 for one rebuilt held nodal model x'=-A*x+b.

    The forced-support owner validates connected symmetric transport and
    positive capacities/EPI coefficient, rebuilding cached model fields.
    ``output_rows`` is a nonempty ordered collection of finite exact or
    represented real rows, each matching the reference node order. Redundant,
    signed and all-zero rows are permitted. ``output_offset`` has one known
    constant per row and defaults to zero. No output is inferred from a rate.

    Scan every row of O, OA, OA^2, ... and retain a row exactly when rational
    rank increases. The first complete level after power zero with no new
    direction certifies invariance. For initial rank r>0 it occurs by power
    n-r+1; rank one can require the full n+1 levels through power n. The
    all-zero case is explicitly checked through power one, without inversion.
    Dependent or zero rows never terminate their level early.

    Independent retained rows C give T from a deterministic invertible column
    minor, G=CAT and D=OT. Full identities CT=I, CA=GC and O=DC verify the
    resulting s=C*x, s'=-G*s+C*b and y=D*s+o0. Every autonomous all-state
    linear observation retaining O contains all OA^k, proving minimality in
    that class. Known b and o0 alter neither this rank nor the needed state
    dimension: neither is appended as a homogeneous coordinate. This is not
    minimality over nonlinear observations or restricted trajectories.

    ``max_rank_calls`` is a positive non-boolean operational rank-call limit,
    not a physical parameter or a bound on wall time, memory, intermediate
    elimination work or coefficient bit growth. Exhaustion returns no partial
    result or approximate fallback. No graph, trajectory, exponential, phase
    law, measurement provenance or autonomous partition selection is supplied.
    """
    _check_realization_rank_budget(max_rank_calls)
    ref, a, b, x = _held_affine_nodal_model(reference, epi)
    if isinstance(output_rows, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("output_rows must be an ordered collection of rows")
    try:
        rows = tuple(output_rows)
    except TypeError as exc:
        raise TypeError("output_rows must be an ordered collection of rows") from exc
    if not rows:
        raise ValueError("output_rows must be nonempty")
    rows = tuple(ordered_vector(row, f"output_rows[{i}]") for i, row in enumerate(rows))
    if any(len(row) != len(a) for row in rows):
        raise ValueError("each output row must match the reference node order")
    offset = (
        (Fraction(0),) * len(rows)
        if output_offset is None
        else ordered_vector(output_offset, "output_offset")
    )
    if len(offset) != len(rows):
        raise ValueError("output_offset must contain one constant per output row")
    return _affine_nodal_realization_core(
        ref,
        a,
        b,
        x,
        rows,
        offset,
        max_rank_calls=max_rank_calls,
    )


def _affine_nodal_realization_core(ref, a, b, x, r, offset, *, max_rank_calls):
    """One invariant-row algorithm for validated generic and partition outputs."""
    n, m = len(a), len(r)
    rank_calls = product_calls = max_bits = 0

    def retain(matrix):
        nonlocal max_bits
        for row in matrix:
            for value in row:
                max_bits = max(
                    max_bits,
                    abs(value.numerator).bit_length(),
                    value.denominator.bit_length(),
                )
        return matrix

    def rank(matrix):
        nonlocal rank_calls
        if rank_calls >= max_rank_calls:
            raise ValueError("exact rank-call budget exhausted; realization incomplete")
        rank_calls += 1
        return exact_rank(matrix)

    def product(left, right):
        nonlocal product_calls
        product_calls += 1
        return retain(_exact_matrix_product(left, right))

    def mv(matrix, vector):
        result = tuple(dot(row, vector) for row in matrix)
        retain((result,))
        return result

    retain(a)
    retain(r)
    retain((b, x, offset))
    basis, labels, levels = [], [], []
    level = r
    for power in range(n + 1):
        previous_rank = len(basis)
        selected = []
        for row_index, row in enumerate(level):
            observed_rank = rank((*basis, row))
            if observed_rank not in (len(basis), len(basis) + 1):
                raise RuntimeError(
                    "exact independent-row selection lost rank consistency"
                )
            if observed_rank > len(basis):
                basis.append(row)
                labels.append((power, row_index))
                selected.append(row_index)
        levels.append(
            ForcedSupportRealizationLevel(
                power,
                previous_rank,
                len(basis),
                m,
                tuple(selected),
            )
        )
        if power > 0 and len(basis) == previous_rank:
            break
        if power < n:
            level = product(level, a)
    else:
        raise RuntimeError(
            "exact row-space closure did not stabilize within its dimension bound"
        )
    c = tuple(basis)
    dimension = len(c)
    output_rank = levels[0].rank_after
    columns, chosen = [], []
    zero, one = Fraction(0), Fraction(1)
    if dimension:
        for j in range(n):
            column = tuple(row[j] for row in c)
            observed_rank = rank((*columns, column))
            if observed_rank not in (len(columns), len(columns) + 1):
                raise RuntimeError("exact pivot-column selection lost rank consistency")
            if observed_rank > len(columns):
                columns.append(column)
                chosen.append(j)
            if len(columns) == dimension:
                break
        if len(columns) != dimension:
            raise RuntimeError(
                "independent observation rows have no invertible column minor"
            )
        minor = tuple(tuple(row[j] for j in chosen) for row in c)
        inverse = retain(exact_matrix_inverse(minor))
        selected_positions = {j: i for i, j in enumerate(chosen)}
        t = retain(
            tuple(
                (
                    inverse[selected_positions[i]]
                    if i in selected_positions
                    else (zero,) * dimension
                )
                for i in range(n)
            )
        )
        ca = product(c, a)
        g, d = product(ca, t), product(r, t)
    else:
        # Empty tuples alone lose their column count. Handle the known
        # 0-by-n, n-by-0, 0-by-0 and m-by-0 shapes explicitly rather than
        # changing the nonempty shared product/inverse contracts.
        t = ((),) * n
        ca, g, d = (), (), ((),) * m
    identity = tuple(
        tuple(one if i == j else zero for j in range(dimension))
        for i in range(dimension)
    )
    source, state = mv(c, b), mv(c, x)
    reduced_rate = tuple(
        force - drift for force, drift in zip(source, mv(g, state), strict=True)
    )
    fine_rate = tuple(force - drift for force, drift in zip(b, mv(a, x), strict=True))
    output_state = tuple(
        value + constant for value, constant in zip(mv(d, state), offset, strict=True)
    )
    projected_state = tuple(
        value + constant for value, constant in zip(mv(r, x), offset, strict=True)
    )
    output_rate, projected_rate = mv(d, reduced_rate), mv(r, fine_rate)
    retain((reduced_rate, fine_rate, output_state, projected_state))
    checks = {
        "rank(C)=dimension": rank(c) == dimension,
        "C T=I": (product(c, t) if dimension else ()) == identity,
        "C A=G C": ca == (product(g, c) if dimension else ()),
        "O=D C": r == (product(d, c) if dimension else ((zero,) * n,) * m),
        "complete-level stabilization": levels[-1].rank_before == levels[-1].rank_after,
        "initial output rank retained": output_rank <= dimension,
        "reduced affine nodal rate": reduced_rate == mv(c, fine_rate),
        "output state reconstruction": output_state == projected_state,
        "output rate reconstruction": output_rate == projected_rate,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"exact sufficient-observation identities failed: {failed}")
    return AffineNodalRealization(
        reference=ref,
        output_rows=r,
        output_offset=offset,
        output_count=m,
        output_rank=output_rank,
        dimension=dimension,
        full_state_dimension=n,
        extra_coordinates=dimension - output_rank,
        rank_progression=tuple(level.rank_after for level in levels),
        level_records=tuple(levels),
        selected_row_labels=tuple(labels),
        pivot_columns=tuple(chosen),
        observation=c,
        right_inverse=t,
        reduced_generator=g,
        output_map=d,
        reduced_source=source,
        epi=x,
        reduced_state=state,
        reduced_rate=reduced_rate,
        output_state=projected_state,
        output_rate=projected_rate,
        reconstructed_output_state=output_state,
        reconstructed_output_rate=output_rate,
        exact_identity_checks=tuple(checks),
        rank_calls=rank_calls,
        max_rank_calls=max_rank_calls,
        matrix_product_calls=product_calls,
        completed_levels=len(levels),
        stabilization_power=levels[-1].power,
        max_coefficient_bits=max_bits,
        scope=(
            "Minimal all-state linear observation retaining the declared outputs of one fixed "
            "exact affine scalar nodal model. Reduced coordinates may be signed/nonlocal and "
            "are not automatically canonical nodes. Constant output offsets are held data, not "
            "new dynamic coordinates or measurement-provenance certificates. Full dimension "
            "excludes only proper linear "
            "compression in this class; nonlinear/reachable-state reductions remain separate. "
            "No shared observation across changed models, trajectory, exponential, runtime "
            "closure, persistence or empirical entity is certified."
        ),
    )


def observe_forced_support_realization(
    reference,
    blocks,
    *,
    epi=None,
    max_rank_calls=4096,
) -> ForcedSupportRealization:
    r"""Derive the smallest invariant row space containing the family outputs.

    Rebuild exact partition geometry and delegate R, RA, RA^2, ... to the
    same invariant-row owner as ``observe_affine_nodal_realization``. The
    proper partition gives at least two independent rows, so its complete
    stabilization occurs by power n-1. A dependent row never terminates a
    level. The existing partition observation and resource fields are retained.

    The shared owner proves CT=I, CA=GC and R=DC for independent state C,
    with s=C*x, s'=-G*s+C*b and y=D*s. Minimality is among autonomous linear
    observations retaining R for every scalar x; no constant state, source
    fit, implicit centering, trajectory or exponential is introduced.
    ``max_rank_calls`` is a positive non-boolean operational limit on exact
    rank invocations, not on wall time, memory or elimination bit growth.
    Exhaustion raises without a partial result or approximate-rank fallback.
    """
    _check_realization_rank_budget(max_rank_calls)
    closure = observe_forced_support_closure(reference, blocks, epi=epi)
    result = _affine_nodal_realization_core(
        closure.reference,
        closure.micro_generator,
        closure.affine_source,
        closure.epi,
        closure.projection,
        (Fraction(0),) * len(closure.projection),
        max_rank_calls=max_rank_calls,
    )
    if (
        result.output_rank != len(closure.projection)
        or result.output_state != closure.projected_epi
        or result.output_rate != closure.projected_nodal_rate
    ):
        raise RuntimeError("partition realization differs from its rebuilt closure")
    check_names = {
        "O=D C": "R=D C",
        "output state reconstruction": "projected state reconstruction",
        "output rate reconstruction": "projected rate reconstruction",
    }
    return ForcedSupportRealization(
        closure=closure,
        dimension=result.dimension,
        full_state_dimension=result.full_state_dimension,
        extra_coordinates=result.extra_coordinates,
        rank_progression=result.rank_progression,
        level_records=result.level_records,
        selected_row_labels=result.selected_row_labels,
        pivot_columns=result.pivot_columns,
        observation=result.observation,
        right_inverse=result.right_inverse,
        reduced_generator=result.reduced_generator,
        output_map=result.output_map,
        reduced_source=result.reduced_source,
        epi=result.epi,
        reduced_state=result.reduced_state,
        reduced_rate=result.reduced_rate,
        projected_state=result.output_state,
        projected_rate=result.output_rate,
        reconstructed_projected_state=result.reconstructed_output_state,
        reconstructed_projected_rate=result.reconstructed_output_rate,
        exact_identity_checks=tuple(
            check_names.get(name, name) for name in result.exact_identity_checks
        ),
        rank_calls=result.rank_calls,
        max_rank_calls=result.max_rank_calls,
        matrix_product_calls=result.matrix_product_calls,
        completed_levels=result.completed_levels,
        stabilization_power=result.stabilization_power,
        max_coefficient_bits=result.max_coefficient_bits,
        scope=(
            "Minimal all-state linear observation retaining the declared outputs of one fixed "
            "exact affine scalar nodal model. Reduced coordinates may be signed/nonlocal and "
            "are not automatically canonical nodes. Full dimension excludes only proper linear "
            "compression in this class; nonlinear/reachable-state reductions remain separate. "
            "No shared observation across changed models, trajectory, exponential, runtime "
            "closure, persistence or empirical entity is certified."
        ),
    )


@dataclass(frozen=True)
class ForcedSupportEulerFrame:
    """One exact model frame decoded from the sufficient linear state."""

    ordinal: int
    time: Fraction
    reduced_state: Vector
    reduced_rate: Vector
    epi: Vector
    fine_rate: Vector
    projected_epi: Vector
    encoding_residual: Vector
    rate_residual: Vector
    output_residual: Vector
    fine_euler_residual: Vector


@dataclass(frozen=True)
class ForcedSupportEulerPrediction:
    """Detached exact Euler prediction, not a graph execution certificate.

    The full-rank observation is rebuilt from primitive held-model inputs.
    ``frames[0]`` is the supplied initial state; each subsequent frame follows
    the corresponding step. Fine matrices propagate additive model/runtime
    defects only after a separate caller authenticates those defects. The
    convex-step flags concern the ideal fixed-model disagreement map, not
    clipping, pressure realization, grammar or continuous-time accuracy.
    """

    realization: ForcedSupportRealization
    steps: Vector
    frames: tuple[ForcedSupportEulerFrame, ...]
    fine_euler_matrices: tuple[Matrix, ...]
    elapsed_time: Fraction
    convex_step_admissible: tuple[bool, ...]
    max_steps: int
    scope: str


def predict_forced_support_realization_euler(
    reference,
    blocks,
    steps,
    *,
    epi=None,
    max_rank_calls=4096,
    max_steps=256,
) -> ForcedSupportEulerPrediction:
    r"""Forecast full microscopic EPI through its rebuilt sufficient state.

    For each exact nonnegative duration h, use the shared nodal Euler kernel
    with rational operands to compute s_next=s+h*(-G*s+C*b), then decode
    x_next=T*s_next. Check the full fine recurrence x_next=(I-h*A)*x+h*b,
    its rate, encoding and output identities. No observed future response is
    accepted as input, and no graph, history, pressure, phase or capacity is
    written. This is an exact discretization of the declared held model,
    not its continuous solution or the rounded/clipped engine trajectory.

    Full rank is required because the result promises every microscopic
    coordinate; a lower-rank sufficient observation does not determine x.
    The ordered step sequence must be nonempty and at most ``max_steps``
    long. Zero steps are identities. Limits are positive non-boolean integers;
    exhaustion returns no partial prediction. They bound step materialization
    and rank calls, not rational bit growth, memory or wall time. A returned
    object does not prove that a caller retained it before runtime execution.
    """
    if type(max_steps) is not int:
        raise TypeError("max_steps must be a positive non-boolean integer")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if isinstance(steps, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("steps must be an ordered sequence")
    try:
        iterator = iter(steps)
    except TypeError as exc:
        raise TypeError("steps must be an ordered sequence") from exc
    durations = []
    for value in iterator:
        if len(durations) >= max_steps:
            raise ValueError("step budget exhausted; prediction incomplete")
        duration = exact_or_represented_real(value, "step duration")
        if duration < 0:
            raise ValueError("step durations must be nonnegative")
        durations.append(duration)
    if not durations:
        raise ValueError("steps must be nonempty")
    durations = tuple(durations)
    realization = observe_forced_support_realization(
        reference,
        blocks,
        epi=epi,
        max_rank_calls=max_rank_calls,
    )
    n = realization.full_state_dimension
    if realization.dimension != n:
        raise ValueError("full microscopic prediction requires a full-rank realization")
    closure = realization.closure
    a, b = closure.micro_generator, closure.affine_source
    c, t = realization.observation, realization.right_inverse
    g, d = realization.reduced_generator, realization.output_map
    zero, one = Fraction(0), Fraction(1)

    def mv(matrix, vector):
        return tuple(dot(row, vector) for row in matrix)

    def difference(left, right):
        return tuple(x - y for x, y in zip(left, right, strict=True))

    def frame(ordinal, time, state, expected_epi):
        decoded = mv(t, state)
        fine_rate = difference(b, mv(a, decoded))
        reduced_rate = difference(realization.reduced_source, mv(g, state))
        projected = mv(closure.projection, decoded)
        residuals = (
            difference(mv(c, decoded), state),
            difference(mv(c, fine_rate), reduced_rate),
            difference(projected, mv(d, state)),
            difference(decoded, expected_epi),
        )
        if any(any(residual) for residual in residuals):
            raise RuntimeError("exact sufficient-state Euler identities failed")
        return ForcedSupportEulerFrame(
            ordinal,
            time,
            state,
            reduced_rate,
            decoded,
            fine_rate,
            projected,
            *residuals,
        )

    frames = [frame(0, zero, realization.reduced_state, realization.epi)]
    matrices = []
    for ordinal, duration in enumerate(durations, start=1):
        previous = frames[-1]
        next_state = tuple(
            euler_update(value, duration, rate)
            for value, rate in zip(
                previous.reduced_state, previous.reduced_rate, strict=True
            )
        )
        matrix = tuple(
            tuple(
                euler_update(one if i == j else zero, duration, -a[i][j])
                for j in range(n)
            )
            for i in range(n)
        )
        expected_epi = tuple(
            euler_update(value, duration, rate)
            for value, rate in zip(previous.epi, previous.fine_rate, strict=True)
        )
        affine_epi = tuple(
            euler_update(value, duration, force)
            for value, force in zip(mv(matrix, previous.epi), b, strict=True)
        )
        if expected_epi != affine_epi:
            raise RuntimeError(
                "exact affine Euler matrix lost its nodal update identity"
            )
        matrices.append(matrix)
        frames.append(
            frame(ordinal, previous.time + duration, next_state, expected_epi)
        )
    return ForcedSupportEulerPrediction(
        realization=realization,
        steps=durations,
        frames=tuple(frames),
        fine_euler_matrices=tuple(matrices),
        elapsed_time=frames[-1].time,
        convex_step_admissible=tuple(
            value <= closure.reference.max_convex_step for value in durations
        ),
        max_steps=max_steps,
        scope=(
            "Exact rational Euler forecast of one rebuilt held affine scalar nodal model, "
            "advanced in its full-rank sufficient coordinates and decoded without loss. "
            "No runtime graph writes, clipping, pressure refresh, continuous-solution accuracy, "
            "causal pre-execution seal, damaged-pattern recovery or autonomous maintenance "
            "is certified. Runtime defects and input chronology require separate evidence."
        ),
    )


@dataclass(frozen=True)
class EpiMemorySample:
    """One finite binary64 observation; arrays are detached and read-only.

    ``source_free_epi`` retains the derived kernel with zero initial hidden
    state. ``markov_epi`` omits both the kernel and the initial-state source.
    ``identity_residual`` measures the full reconstructed rate against the
    projected fine rate; it is not an upper bound on exponential error.
    """

    time: float
    kernel: np.ndarray
    initial_source: np.ndarray
    convolution: np.ndarray
    projected_epi: np.ndarray
    markov_epi: np.ndarray
    source_free_epi: np.ndarray
    projected_rate: np.ndarray
    reconstructed_rate: np.ndarray
    identity_residual: float
    relative_identity_residual: float
    fine_semigroup_residual: float
    markov_semigroup_residual: float


@dataclass(frozen=True)
class EpiMemoryObservation:
    """Fixed-geometry projected diffusion diagnostics, without live provenance.

    ``instantaneous_generator`` is the represented product ``R A P``;
    ``quotient_generator`` is independently materialized by the shared
    reversible geometry builder. Their discrepancy and projector defects are
    exposed. ``closure_within_tolerance`` is a numerical diagnostic inherited
    from that builder, never a reason to erase small memory terms.
    """

    nodes: tuple
    blocks: tuple[tuple, ...]
    projection: np.ndarray
    lift: np.ndarray
    micro_generator: np.ndarray
    instantaneous_generator: np.ndarray
    quotient_generator: np.ndarray
    hidden_projector: np.ndarray
    hidden_generator: np.ndarray
    macro_metric_weights: np.ndarray
    initial_epi: np.ndarray
    initial_macro: np.ndarray
    initial_hidden: np.ndarray
    right_inverse_residual: float
    projector_residual: float
    quotient_generator_residual: float
    generator_residual: float
    closure_within_tolerance: bool
    tolerance: float
    numerical_tolerance: float
    samples: tuple[EpiMemorySample, ...]
    scope: str


def _real_vector(values, name: str) -> np.ndarray:
    """Reject coercion of strings, booleans and complex data into a real chart."""
    try:
        raw = np.asarray(values, dtype=object)
        if raw.ndim != 1 or not raw.size:
            raise ValueError(f"{name} must be a nonempty real vector")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in raw
        ):
            raise ValueError(f"{name} must contain real numbers only")
        result = np.array(raw, dtype=float)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real vector") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _readonly(values: np.ndarray) -> np.ndarray:
    """Use immutable backing bytes, so writeability cannot be re-enabled."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("EPI memory observation exceeds finite numeric range")
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _norm(values: np.ndarray) -> float:
    result = float(np.linalg.norm(values, ord=np.inf))
    if not np.isfinite(result):
        raise ValueError("EPI memory residual exceeds finite numeric range")
    return result


def _exponential(generator: np.ndarray, time: float) -> np.ndarray:
    scaled = time * generator
    if not np.all(np.isfinite(scaled)):
        raise ValueError("EPI memory exponential exceeds finite numeric range")
    _norm(scaled)
    result = matrix_exponential(scaled)
    if not np.all(np.isfinite(result)):
        raise ValueError("EPI memory exponential exceeds finite numeric range")
    return result


def _stationary_weights(metric: np.ndarray) -> np.ndarray:
    scaled = metric / np.max(metric)
    return scaled / np.sum(scaled)


def _check_stochastic(
    flow: np.ndarray, stationary: np.ndarray, tolerance: float
) -> float:
    """Check necessary exact diffusion identities, not exponential accuracy."""
    residual = max(
        _norm(np.sum(flow, axis=1) - 1.0),
        max(0.0, -float(np.min(flow))),
        _norm(stationary @ flow - stationary),
    )
    if residual > tolerance:
        raise ValueError(
            "EPI memory exponential failed stochastic semigroup checks "
            f"(residual={residual}, numerical_tolerance={tolerance})"
        )
    return residual


def observe_epi_memory(
    graph,
    partition,
    initial_epi,
    *,
    times,
    tolerance: float = 1e-10,
    numerical_tolerance: float = 1e-10,
) -> EpiMemoryObservation:
    """Sample projected diffusion, its derived memory and omission controls.

    ``initial_epi`` is an explicit finite real scalar vector in ``tuple(graph)``
    order; stored graph EPI is not read. ``times`` is a nonempty vector of finite
    nonnegative observation times, preserved in supplied order. They are sample
    coordinates, not fitted physical parameters or integration timesteps.

    Geometry follows ``certify_epi_coarse_graining``: fixed effective symmetric
    nonnegative conductance, positive finite capacities and row strengths,
    and a complete partition reducing to a connected quotient of at least two
    blocks. This also permits directed input whose effective conductance is
    symmetric. Phase, topology changes, events and REMESH are outside scope.

    The NumPy-only exponential is the existing scaling/squaring Taylor routine.
    Reported rate and geometry residuals expose finite arithmetic discrepancies;
    a small residual alone does not establish an accuracy bound. The diagnostic
    uses dense matrices of order at most twice the number of nodes. Independently
    of the geometry ``tolerance``, ``numerical_tolerance`` bounds observed
    row-sum, positivity and stationary-weight defects of the fine and Markov
    propagators. The generator's relative constant/weighted-total defects are
    checked before sampling, including at zero time. Samples
    failing these necessary stochasticity checks are rejected; passing them
    still does not certify exponential accuracy. Both tolerances are numerical
    policies, not physical parameters.
    """
    initial = _real_vector(initial_epi, "initial_epi")
    sample_times = _real_vector(times, "times")
    numeric_tol = float(_real_vector([numerical_tolerance], "numerical_tolerance")[0])
    if numeric_tol <= 0.0:
        raise ValueError("numerical_tolerance must be positive")
    if np.any(sample_times < 0.0):
        raise ValueError("times must be nonnegative")
    geometry = _build_reversible_partition_geometry(
        graph, partition, tolerance=tolerance, label="EPI memory"
    )
    n = len(geometry.nodes)
    if initial.shape != (n,):
        raise ValueError("initial_epi must have one value per graph node")

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            r, p = geometry.projection, geometry.lift
            a = geometry.micro_generator
            q = np.eye(n) - p @ r
            ra = r @ a
            instantaneous = ra @ p
            hidden = q @ a @ q
            hidden_drive = q @ a @ p
            macro_weights = _stationary_weights(geometry.macro_metric_weights)
            fine_weights = _stationary_weights(r.T @ macro_weights)
            generator_scale = _norm(a)
            generator_residual = max(_norm(np.sum(a, axis=1)), _norm(fine_weights @ a))
            if generator_scale > 0.0:
                generator_residual /= generator_scale
            if generator_residual > numeric_tol:
                raise ValueError(
                    "EPI memory generator failed constant/weighted-total checks"
                )
            macro_initial = r @ initial
            hidden_initial = q @ initial
            lifted_initial = p @ macro_initial

            # The lower block integrates the memory forcing along the fine
            # reference, independently of the projected rate reconstruction.
            augmented = np.block([[-a, np.zeros((n, n))], [hidden_drive @ r, -hidden]])
            samples = []
            for raw_time in sample_times:
                time = float(raw_time)
                flow = _exponential(augmented, time)
                hidden_flow = _exponential(-hidden, time)
                fine_flow = flow[:n, :n]
                markov_flow = _exponential(-instantaneous, time)
                fine_residual = _check_stochastic(fine_flow, fine_weights, numeric_tol)
                markov_residual = _check_stochastic(
                    markov_flow, macro_weights, numeric_tol
                )
                fine = fine_flow @ initial
                projected = r @ fine
                source = -ra @ hidden_flow @ hidden_initial
                convolution = ra @ (flow[n:, :n] @ initial)
                direct_rate = -ra @ fine
                reconstructed = -instantaneous @ projected + source + convolution
                residual = _norm(direct_rate - reconstructed)
                scale = max(1.0, _norm(direct_rate), _norm(reconstructed))
                samples.append(
                    EpiMemorySample(
                        time=time,
                        kernel=_readonly(ra @ hidden_flow @ hidden_drive),
                        initial_source=_readonly(source),
                        convolution=_readonly(convolution),
                        projected_epi=_readonly(projected),
                        markov_epi=_readonly(markov_flow @ macro_initial),
                        source_free_epi=_readonly(r @ fine_flow @ lifted_initial),
                        projected_rate=_readonly(direct_rate),
                        reconstructed_rate=_readonly(reconstructed),
                        identity_residual=residual,
                        relative_identity_residual=residual / scale,
                        fine_semigroup_residual=fine_residual,
                        markov_semigroup_residual=markov_residual,
                    )
                )

            return EpiMemoryObservation(
                nodes=geometry.nodes,
                blocks=geometry.blocks,
                projection=_readonly(r),
                lift=_readonly(p),
                micro_generator=_readonly(a),
                instantaneous_generator=_readonly(instantaneous),
                quotient_generator=_readonly(geometry.macro_generator),
                hidden_projector=_readonly(q),
                hidden_generator=_readonly(hidden),
                macro_metric_weights=_readonly(geometry.macro_metric_weights),
                initial_epi=_readonly(initial),
                initial_macro=_readonly(macro_initial),
                initial_hidden=_readonly(hidden_initial),
                right_inverse_residual=_norm(r @ p - np.eye(len(geometry.blocks))),
                projector_residual=_norm(q @ q - q),
                quotient_generator_residual=_norm(
                    instantaneous - geometry.macro_generator
                ),
                generator_residual=generator_residual,
                closure_within_tolerance=geometry.nodal_closure_within_tolerance,
                tolerance=float(tolerance),
                numerical_tolerance=numeric_tol,
                samples=tuple(samples),
                scope=(
                    "Finite binary64 offline samples of fixed reversible "
                    "pure-EPI diffusion; exact projected-memory identities "
                    "are separate from approximate exponential evaluation. "
                    "No graph evolution, runtime provenance, REMESH equivalence, "
                    "mesh convergence or empirical correspondence is certified."
                ),
            )
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("EPI memory observation exceeds finite numeric range") from exc
