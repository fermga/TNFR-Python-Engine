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
