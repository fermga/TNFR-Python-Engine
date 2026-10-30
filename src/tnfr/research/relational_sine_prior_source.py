"""Separate known-source role for the fixed prior-to-future software protocol.

Only preparation reads the hidden coordinates to emit earlier derivative
observations. The predictor module never imports this source role. This is
information separation in a reproducible software test, not physical evidence
or a security boundary against a reader inspecting the source archive.
"""

from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ..sdk.relational_reports import _project
from .relational_acquisition import _exact, _require

_NODES = ("left", "right", "hidden")
_NEIGHBORS = ((2,), (2,), (0, 1))


def prepare_sine_prior_source():
    """Freeze one exact preparation and emit static, padded prior evidence.

    The full-field derivatives below use independent edge sums and their
    chain rule, not the inverse or its acceleration cancellation formula.
    No future sample or trajectory is computed here.
    """
    form = (Q(0), Q(0), Q(1))
    phase = (Q(0), Q(1, 2), Q(0))
    capacity = (Q(1), Q(2), Q(1))
    e, a, b = Q(1, 2), 1 / (2 * pi_interval()), 1 / (2 * pi_interval())
    gradient = tuple(
        sum((form[i] - form[j] for j in row), Q(0)) for i, row in enumerate(_NEIGHBORS)
    )
    current = tuple(
        sum((sin(I(phase[j] - phase[i])) for j in row), I(0))
        for i, row in enumerate(_NEIGHBORS)
    )
    velocity = tuple(
        (-e * gradient[i] + a * current[i]) * capacity[i] / len(row)
        for i, row in enumerate(_NEIGHBORS)
    )
    omega = tuple(
        b * gradient[i] * capacity[i] / len(row) for i, row in enumerate(_NEIGHBORS)
    )
    gradient_rate = tuple(
        sum((velocity[i] - velocity[j] for j in row), I(0))
        for i, row in enumerate(_NEIGHBORS)
    )
    current_rate = tuple(
        sum(
            (cos(I(phase[j] - phase[i])) * (omega[j] - omega[i]) for j in row),
            I(0),
        )
        for i, row in enumerate(_NEIGHBORS)
    )
    acceleration = tuple(
        (-e * gradient_rate[i] + a * current_rate[i]) * capacity[i] / len(row)
        for i, row in enumerate(_NEIGHBORS)
    )
    alpha = tuple(
        b * gradient_rate[i] * capacity[i] / len(row)
        for i, row in enumerate(_NEIGHBORS)
    )
    padding = Q(1, 1 << 30)

    def evidence(values):
        return {
            node: _project((value.lo - padding, value.hi + padding))
            for node, value in zip(_NODES[:2], values[:2])
        }

    prior = {
        "schema": "tnfr.sine-prior-evidence.v1",
        "source_id": "synthetic-prior-derivative-enclosures-v1",
        "clock_id": "declared-structural-time",
        "observation_time": 0,
        "evidence_window": [0, 0],
        "forecast_start": _project(Q(1, 32)),
        "form_rate_bounds": evidence(velocity),
        "phase_rate_bounds": evidence(omega),
        "form_acceleration_bounds": evidence(acceleration),
        "phase_acceleration_bounds": evidence(alpha),
        "observation_error": {
            "absolute_padding": _project(padding),
            "differentiation_truncation": 0,
            "kind": "synthetic_instantaneous_derivatives_not_finite_difference_samples",
            "arithmetic": "outward_dyadic128_independent_forward_edge_chain_rule",
        },
    }
    source = {
        "schema": "tnfr.sine-reserved-source.v1",
        "nodes": list(_NODES),
        "neighbors": [list(row) for row in _NEIGHBORS],
        "initial": _project(form + phase + (capacity[-1],)),
        "visible_capacity": _project(capacity[:2]),
        "scope": "known_exact_software_preparation_not_a_physical_measurement",
    }
    return prior, source


def evaluate_sine_prior_source(source):
    """Evaluate the sealed preparation once, after issuing the forecast."""
    from ..dynamics.relational import RelationalExchangeModel
    from ..physics.relational_sine_forecast import bound_sine_flow

    expected = prepare_sine_prior_source()[1]
    _require(source == expected, "unsupported or altered sine source preparation")
    report = bound_sine_flow(
        tuple(I(_exact(value)) for value in source["initial"]),
        neighbors=_NEIGHBORS,
        visible_capacity=(Q(1), Q(2)),
        model=RelationalExchangeModel(1, phase_domain="regular"),
        observation_time=Q(0),
        end_time=Q(1, 16),
        time_step=Q(1, 128),
        order=6,
    )
    return report
