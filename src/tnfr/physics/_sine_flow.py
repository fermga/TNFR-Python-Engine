"""Complete interval/jet sine fields with explicitly held law coefficients.

This shared field consumes every nodal form and primitive phase. It supplies
neither preparation, acquisition, an observation map nor a solver policy.
"""

from fractions import Fraction as Q

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
