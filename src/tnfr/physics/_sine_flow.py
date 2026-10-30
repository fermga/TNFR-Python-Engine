"""Complete interval/jet sine fields with explicitly held law coefficients.

This shared field consumes every nodal form and primitive phase. It supplies
neither preparation, acquisition, an observation map nor a solver policy.
"""

from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import MAX_COMPARISON_DIMENSION
from ..mathematics._interval_taylor import Jet
from ..mathematics._interval_taylor import sin as jet_sin
from ..mathematics._rational_interval import I, sin
from ._sine_admission import _sine_model_coefficients
from .relational_observations import _ordered
from .relational_sine_comparison import _sine_form_gradient, _sine_rate_evaluator


def _held_sine_neighbor_field(
    model, degrees, capacity, neighbor_rows, *, reuse_pair_sines=True
):
    """Prepare complete held weighted rows in the model's original clock.

    Each row contains ``(neighbor_index, bare_neighbor_count,
    conductance_sum)`` triples. Counts are positive ordinary integers, sum
    to the supplied bare degree, and need not agree in opposite directions
    of a reduced family. Internal neighbors contribute to degree but have
    exactly zero gradient and phase current. Zero conductance still retains
    the bare phase support. Capacities and conductance sums are immutable
    exact rationals or rational intervals and must be nonnegative.

    This boundary admits the numerical rows; the caller owns support
    provenance, any reduction/invariance proof and the held-law premise.
    No reciprocity, source membership or clock change is inferred here.
    Evaluation consumes all form coordinates followed by all phase lifts,
    as exact/interval scalars or uniformly ordered Taylor jets.

    With ``reuse_pair_sines=True``, each unordered pair is evaluated once
    in increasing-index orientation, then negated for the reverse row.
    ``False`` preserves the independently directed sine arithmetic needed
    to reproduce historical certificates: outward trigonometric bounds
    need not be bitwise odd even though mathematical sine is odd.
    """
    if type(reuse_pair_sines) is not bool:
        raise TypeError("reuse_pair_sines must be Boolean")
    degrees = _ordered(degrees, "degrees")
    size = len(degrees)
    if not size or any(type(degree) is not int or degree <= 0 for degree in degrees):
        raise ValueError("degrees must be nonempty positive ordinary integers")

    def coefficient(value, label):
        interval = I.coerce(value)
        if interval.lo < 0:
            raise ValueError(f"{label} must be nonnegative")
        return value if isinstance(value, I) else Q(value)

    capacity = _ordered(capacity, "capacity", limit=size + 1)
    if len(capacity) != size:
        raise ValueError("capacity must retain every nodal coefficient")
    capacity = tuple(coefficient(value, "capacity") for value in capacity)
    neighbor_rows = _ordered(neighbor_rows, "neighbor_rows", limit=size + 1)
    if len(neighbor_rows) != size:
        raise ValueError("neighbor_rows must retain every nodal row")
    rows = []
    for i, supplied in enumerate(neighbor_rows):
        row, seen, total = [], set(), 0
        for entry in _ordered(supplied, "neighbor row", limit=size + 1):
            entry = _ordered(entry, "neighbor entry", limit=4)
            if len(entry) != 3:
                raise ValueError(
                    "neighbor entries require index, count and conductance sum"
                )
            j, count, weight = entry
            if type(j) is not int or not 0 <= j < size or j in seen:
                raise ValueError(
                    "neighbor indices must be unique ordinary in-range integers"
                )
            if type(count) is not int or count <= 0:
                raise ValueError(
                    "bare neighbor counts must be positive ordinary integers"
                )
            weight = coefficient(weight, "conductance sum")
            seen.add(j)
            total += count
            if j != i:
                row.append((j, count, weight))
        if total != degrees[i]:
            raise ValueError("degrees must equal the complete bare neighbor counts")
        rows.append(tuple(row))
    rows = tuple(rows)
    evaluate = _sine_rate_evaluator(model, degrees, capacity)

    def flow(values):
        values = _ordered(values, "state", limit=2 * size + 1)
        if len(values) != 2 * size:
            raise ValueError(
                "state must retain every reduced/full form and phase coordinate"
            )
        is_jet = isinstance(values[0], Jet)
        if is_jet:
            order = values[0].order
            if any(
                not isinstance(value, Jet) or value.order != order for value in values
            ):
                raise ValueError("state Taylor jets must all have the same order")
            zero, sine = Jet.constant(0, order), jet_sin
        else:
            if any(isinstance(value, Jet) for value in values):
                raise ValueError("state cannot mix scalar coordinates and Taylor jets")
            values = tuple(I.coerce(value) for value in values)
            zero, sine = I(0), sin
        forms, phases = values[:size], values[size:]
        gradient = tuple(
            sum(((forms[i] - forms[j]) * weight for j, _, weight in row), zero)
            for i, row in enumerate(rows)
        )
        pair_sines = {}

        def current(i, j):
            if not reuse_pair_sines:
                return sine(phases[j] - phases[i])
            pair = (i, j) if i < j else (j, i)
            if pair not in pair_sines:
                left, right = pair
                pair_sines[pair] = sine(phases[right] - phases[left])
            return pair_sines[pair] if i < j else -pair_sines[pair]

        currents = tuple(
            sum((current(i, j) * count for j, count, _ in row), zero)
            for i, row in enumerate(rows)
        )
        rates = evaluate(gradient, currents)
        return rates["form_rates"] + rates["phase_rates"]

    return flow


def _full_sine_field(model, geometry, degrees):
    """Prepare complete held rows and transform both to structural tau=e*t."""
    loss, _, _ = _sine_model_coefficients(model, positive_loss=True)
    size = len(geometry.nodes)
    if size == 0:
        raise ValueError("complete sine field requires nonempty nodal support")
    degrees = _ordered(degrees, "degrees", limit=size + 1)
    if len(degrees) != size or any(
        type(degree) is not int or degree <= 0 for degree in degrees
    ):
        raise ValueError("degrees must be complete positive ordinary integers")
    edges = tuple(geometry.edges)
    neighbors = tuple(
        tuple(j if i == node else i for i, j in edges if node in (i, j))
        for node in range(size)
    )
    if degrees != tuple(map(len, neighbors)):
        raise ValueError("degrees must match every indexed support neighbor row")
    rates_for = _sine_rate_evaluator(model, degrees, (Q(1),) * size)

    def flow(state):
        if len(state) != 2 * size:
            raise ValueError("complete sine field requires both full nodal rows")
        epi, phase = state[:size], state[size:]
        is_jet = isinstance(state[0], Jet)
        zero = Jet.constant(0, state[0].order) if is_jet else I(0)
        sine = jet_sin if is_jet else sin
        currents = [zero for _ in range(size)]
        for i, j in edges:
            current = sine(phase[j] - phase[i])
            currents[i] += current
            currents[j] -= current
        rates = rates_for(_sine_form_gradient(epi, neighbors), tuple(currents))
        return tuple(
            value / loss for value in rates["form_rates"] + rates["phase_rates"]
        )

    def domain(_):
        # The complete sine law is globally smooth. This is not an acute
        # sector, identity, work or native resultant-domain certificate.
        return (Q(1),)

    return flow, domain


def _prepared_quarter_sine_rows(*, gradient, current, capacity, gamma):
    """Assemble admitted complete weighted D/4 rows in their supplied clock.

    Gradient and current have the same shape as positive held capacities.
    The current includes its conductance weights. This arithmetic works with
    rational intervals or common-order Taylor jets; it neither selects the
    bare-neighbor law nor supplies a reduction or a trajectory.
    """
    return tuple(
        (-q + gamma * s) * (nu / 4) for q, s, nu in zip(gradient, current, capacity)
    ) + tuple(q * (gamma * nu / 4) for q, nu in zip(gradient, capacity))


def _held_quarter_sine_field(
    *,
    capacity,
    gamma,
    weighted_edges,
    retained_indices=None,
    hidden_form=(),
    hidden_phase=(),
):
    """Compile weighted quarter-law rows with optional fixed hidden references.

    ``capacity`` retains the full positive nodal vector. Edges are simple
    indexed ``(i,j,weight)`` triples. A nonempty ordered retained index set
    declares the evolving coordinates; its complement follows full node
    order and supplies fixed hidden form/phase references. Omitting the set
    prepares all rows. Evaluation consumes exactly the retained forms then
    retained lifted phases, returning only those two evolving rows. Hidden
    neighbors are constant boundary values; no hidden dynamic row is built.

    All coefficient and shape admission precedes interval or sine work.
    States accept exact or represented finite scalars, rational intervals,
    or uniformly ordered jets. Constants are lifted to the same jet order.
    The caller establishes the source, held law, interpretation of phases
    and any approximation error; this function executes no solver or event.
    """
    capacity = _ordered(capacity, "capacity", limit=MAX_COMPARISON_DIMENSION + 1)
    size = len(capacity)
    if not 1 <= size <= MAX_COMPARISON_DIMENSION:
        raise ValueError("quarter field requires between 1 and 24 nodal capacities")
    capacity = tuple(
        exact_or_represented_real(value, f"capacity[{i}]")
        for i, value in enumerate(capacity)
    )
    gamma = exact_or_represented_real(gamma, "gamma")
    if gamma < 0 or any(value <= 0 for value in capacity):
        raise ValueError(
            "quarter field requires positive capacities and nonnegative gamma"
        )
    max_edges = size * (size - 1) // 2
    raw_edges = _ordered(weighted_edges, "weighted_edges", limit=max_edges + 1)
    if len(raw_edges) > max_edges:
        raise ValueError("quarter field exceeds its simple-edge budget")
    edges, seen = [], set()
    for raw in raw_edges:
        edge = _ordered(raw, "weighted edge", limit=4)
        if len(edge) != 3:
            raise ValueError("weighted edges require two indices and one conductance")
        i, j, weight = edge
        if any(type(index) is not int or not 0 <= index < size for index in (i, j)):
            raise ValueError("weighted edge indices must be ordinary in-range integers")
        pair = (min(i, j), max(i, j))
        if i == j or pair in seen:
            raise ValueError("quarter field support must be simple and loop-free")
        seen.add(pair)
        weight = exact_or_represented_real(weight, "conductance")
        if weight < 0:
            raise ValueError("conductances must be nonnegative")
        edges.append((i, j, weight))
    retained = (
        tuple(range(size))
        if retained_indices is None
        else _ordered(retained_indices, "retained_indices", limit=size + 1)
    )
    if (
        not retained
        or len(retained) > size
        or any(type(i) is not int or not 0 <= i < size for i in retained)
        or len(set(retained)) != len(retained)
    ):
        raise ValueError("retained indices must be a nonempty ordered distinct subset")
    hidden = tuple(i for i in range(size) if i not in retained)
    references = []
    for raw, label in ((hidden_form, "hidden_form"), (hidden_phase, "hidden_phase")):
        raw = _ordered(raw, label, limit=len(hidden) + 1)
        if len(raw) != len(hidden):
            raise ValueError(f"{label} must match the hidden complement order")
        references.append(
            tuple(exact_or_represented_real(value, label) for value in raw)
        )
    retained_positions = {node: local for local, node in enumerate(retained)}
    hidden_positions = {node: local for local, node in enumerate(hidden)}
    active_edges = tuple(
        (i, j, weight)
        for i, j, weight in edges
        if weight and (i in retained_positions or j in retained_positions)
    )
    retained_capacity = tuple(capacity[i] for i in retained)
    count = len(retained)

    def flow(state):
        values = _ordered(state, "retained state", limit=2 * count + 1)
        if len(values) != 2 * count:
            raise ValueError(
                "state must contain exactly the retained form and phase rows"
            )
        jets = tuple(isinstance(value, Jet) for value in values)
        if any(jets):
            if not all(jets) or len({value.order for value in values}) != 1:
                raise ValueError("retained jets must have one common order")
            order = values[0].order

            def constant(value):
                return Jet.constant(value, order)

            zero, sine = constant(0), jet_sin
        else:
            # Admit the whole raw row before allocating intervals or taking sine.
            normalized = tuple(
                (
                    value
                    if isinstance(value, I)
                    else exact_or_represented_real(value, "state")
                )
                for value in values
            )
            values = tuple(I.coerce(value) for value in normalized)
            constant = I.coerce
            zero, sine = I(0), sin
        hidden_forms = tuple(constant(value) for value in references[0])
        hidden_phases = tuple(constant(value) for value in references[1])

        def coordinate(node, phase):
            if node in retained_positions:
                return values[retained_positions[node] + (count if phase else 0)]
            return (hidden_phases if phase else hidden_forms)[hidden_positions[node]]

        gradient, currents = [zero] * count, [zero] * count
        for i, j, weight in active_edges:
            difference = (coordinate(i, False) - coordinate(j, False)) * weight
            current = sine(coordinate(j, True) - coordinate(i, True)) * weight
            if i in retained_positions:
                slot = retained_positions[i]
                gradient[slot] += difference
                currents[slot] += current
            if j in retained_positions:
                slot = retained_positions[j]
                gradient[slot] -= difference
                currents[slot] -= current
        return _prepared_quarter_sine_rows(
            gradient=tuple(gradient),
            current=tuple(currents),
            capacity=retained_capacity,
            gamma=gamma,
        )

    return flow
