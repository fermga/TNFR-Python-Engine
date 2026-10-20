"""One fixed local-VAL response with associated full-law temporal samples.

The declared preparation is admitted separately from its ideal contact
ancestry. Numerical evolution and the independent interval certificate have
different owners. This adapter never proposes or executes THOL.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as Q
from math import fsum, pi, sin
from unittest.mock import patch

import networkx as nx

from .._exact_time import finite_represented_real
from ..alias import get_attr, set_attr, set_dnfr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA
from ..dynamics.integrators import rk4_coupled_step
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import I, pi_interval
from ..mathematics._rational_interval import sin as interval_sin
from ..operators import network_stage
from ..operators._thol_config import resolve_thol_bifurcation_threshold
from ..operators.definitions import Expansion, SelfOrganization, Silence
from ..operators.grammar_debt import node_debt, reset_grammar_state_from_history
from ..operators.grammar_dynamics import validate_candidate
from ..operators.grammar_execution import ValidatedSequence
from ..operators.nodal_equation import observe_structural_acceleration
from ..operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from ..operators.word_execution import execute_network_operator_stage
from ..physics.relational_sine_support import bound_relational_sine_support_exchange
from ..sdk._state import copy_graph_state
from .local_capacity_history_certificate import admit_source, certify_history
from .retained_graph_state import record_graph_state

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
TIMES = (Q(7), Q(7) + Q(1, 2**20), Q(7) + Q(1, 2**19))
PARENT = 2
CONTEXT = {"initial_epi_nonzero": True, "compatibility_profile": "core"}
PRESSURE_ERROR_BUDGET = Q(1, 2**46)


class _EventClock:
    @classmethod
    def now(cls, tz=None):
        return datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def capture(graph):
    """Admit actual primitives and the live unforced law at this endpoint."""
    return bound_relational_sine_support_exchange(graph, reference_model=MODEL)


def numerical_field(source):
    """Construct independent binary64 full rows on the declared held support.

    The interval field used for certification is assembled by the existing
    weighted-symmetry owner. This separate edge calculation retains unwrapped
    phase lifts and refreshes both rows at every supplied state.
    """
    source = capture_source(source)
    if (
        source.reference_model.effective_weights != (0.0, 1.0)
        or source.reference_model.storage_scale != 1
    ):
        raise ValueError("the numerical adapter requires e=0, w=beta=1")
    size = len(source.nodes)
    positions = {node: i for i, node in enumerate(source.nodes)}
    edges = tuple(
        (
            positions[left],
            positions[right],
            finite_represented_real(weight, "conductance")[0],
        )
        for (left, right), weight in zip(source.edges, source.conductances, strict=True)
    )
    coefficients = []
    for nu, degree in zip(source.capacity, source.degrees, strict=True):
        capacity = finite_represented_real(nu, "capacity")[0]
        coefficient = finite_represented_real(capacity / (pi * degree), "mobility")[0]
        if capacity != 0 and coefficient == 0:
            raise ValueError("nonzero mobility is lost in binary64")
        coefficients.append(coefficient)

    def field(time, state):
        if len(state) != 2 * size:
            raise ValueError("complete numerical state requires both nodal rows")
        forms, phases = state[:size], state[size:]
        form_terms, phase_terms = [[] for _ in range(size)], [[] for _ in range(size)]
        for left, right, weight in edges:
            current = sin(phases[right] - phases[left])
            gradient = weight * (forms[left] - forms[right])
            form_terms[left].append(current)
            form_terms[right].append(-current)
            phase_terms[left].append(gradient)
            phase_terms[right].append(-gradient)
        return tuple(
            coefficients[i] * fsum(row) for i, row in enumerate(form_terms)
        ) + tuple(coefficients[i] * fsum(row) for i, row in enumerate(phase_terms))

    return field


def capture_source(source):
    """Re-admit primitives without trusting report-derived fields."""
    from ..physics.relational_sine_support import _rebuild_sine_support_comparison

    return _rebuild_sine_support_comparison(source)


def _fresh_pressure_bounds(source):
    positions = {node: i for i, node in enumerate(source.nodes)}
    currents = [I(0) for _ in source.nodes]
    for left, right in source.edges:
        i, j = positions[left], positions[right]
        current = interval_sin(I(source.phase[j] - source.phase[i]))
        currents[i] += current
        currents[j] -= current
    return tuple(
        value / (pi_interval() * degree)
        for value, degree in zip(currents, source.degrees, strict=True)
    )


def refresh_pressure(graph):
    """Evaluate independent sine pressure, without dividing an observed rate."""
    source = capture(graph)
    # Refine the snapshot's cheaper bounds for this separate precision budget.
    # The same independently specified map is retained, without rate inversion.
    enclosures = _fresh_pressure_bounds(source)
    for index, node in enumerate(source.nodes):
        phase = float(source.phase[index])
        pressure = fsum(
            sin(float(get_attr(graph.nodes[other], ALIAS_THETA)) - phase)
            for other in graph.neighbors(node)
        ) / (pi * source.degrees[index])
        enclosure = enclosures[index]
        error = max(abs(Q(pressure) - enclosure.lo), abs(Q(pressure) - enclosure.hi))
        if error > PRESSURE_ERROR_BUDGET:
            raise ValueError(
                f"represented pressure at node {node} exceeds the frozen evaluation budget: "
                f"error={float(error)}, numerical={pressure}, interval={enclosure}"
            )
        set_dnfr(graph, node, pressure)
    return source


def _admit_execution_settings(graph):
    if graph.graph.get("GRAMMAR_REJECTION_MODE") != "raise":
        raise ValueError("the frozen protocol requires grammar rejection by raising")
    if graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS") is not True:
        raise ValueError("the frozen protocol requires strict operator preconditions")
    if graph.graph.get("EDGE_AWARE_ENABLED") is not False:
        raise ValueError("the frozen protocol requires capacity-only VAL")
    if any(str(key).startswith(("VAL_", "THOL_")) for key in graph.graph):
        raise ValueError(
            "VAL and THOL policies must retain their unoverridden defaults"
        )
    factors = graph.graph.get("GLYPH_FACTORS")
    if type(factors) is not dict or set(factors) != {"VAL_scale"}:
        raise ValueError("the frozen protocol declares only the VAL scale")
    _, factor = finite_represented_real(factors["VAL_scale"], "VAL_scale")
    if factor != 128:
        raise ValueError("the frozen VAL factor is 128")


def declared_preparation():
    """Declare one rounded unsplit contact; do not replay a source producer."""
    graph = nx.Graph(
        RANDOM_SEED=17,
        _t=7.0,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EDGE_AWARE_ENABLED=False,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        GLYPH_FACTORS={"VAL_scale": 128.0},
    )
    mean, boost = 5 / 16, 1 / 1024
    alpha = 2 * pi / 5
    for node in range(20):
        phase = (node % 10 // 2) * alpha
        if node in (2, 3):
            phase = (1 - boost) * alpha
        elif node in (8, 9):
            phase = (4 + boost) * alpha
        graph.add_node(
            node,
            EPI=mean + (boost if node < 2 else 0),
            theta=phase,
            nu_f=0.5 if node < 2 else 1.0,
            Si=0.5,
            EPI_kind="declared_rounded_post_contact_preparation",
            glyph_history=(
                ["AL", "UM", "SHA"]
                if node == 0
                else ["AL", "AL", "AL", "UM", "SHA"] if node == 1 else []
            ),
        )
        reset_grammar_state_from_history(graph.nodes[node])
    for offset in (0, 10):
        for pair in range(5):
            for left in (0, 1):
                for right in (0, 1):
                    graph.add_edge(
                        offset + 2 * pair + left,
                        offset + 2 * ((pair + 1) % 5) + right,
                        weight=1.0,
                        length=1.0,
                    )
    graph.add_edge(0, 1, weight=1.0, length=1.0)
    weight = 1 - boost / (4 * (2 * mean + boost + 1e-9))
    for left in (0, 1):
        for right in (10, 11):
            graph.add_edge(left, right, weight=weight, length=1.0)
    refresh_pressure(graph)
    return graph


def _state(graph):
    return tuple(
        float(get_attr(graph.nodes[node], alias))
        for alias in (ALIAS_EPI, ALIAS_THETA)
        for node in graph
    )


def encode_result(value):
    """Project interval/dataclass evidence without hiding unavailable values."""
    if isinstance(value, I):
        return {"lo": value.lo, "hi": value.hi}
    if hasattr(value, "__dataclass_fields__"):
        return encode_result(asdict(value))
    if isinstance(value, dict):
        return {key: encode_result(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(encode_result(item) for item in value)
    return value


def evaluate_local_history(graph):
    """Execute one VAL, two full-law RK4 steps and detached admission checks."""
    _admit_execution_settings(graph)
    _, clock = finite_represented_real(graph.graph.get("_t"), "structural clock")
    if tuple(graph) != tuple(range(20)) or clock != 7:
        raise ValueError("fixed source order and structural start time required")
    if graph.nodes[PARENT].get("glyph_history"):
        raise ValueError("parent must not borrow another target's glyph history")
    initial = capture(graph)
    if initial.capacity != (Q(1, 2), Q(1, 2)) + (Q(1),) * 18:
        raise ValueError("pre-VAL capacities differ from the fixed preparation")
    preparation = record_graph_state(graph)
    refresh_pressure(graph)
    operators = (Expansion(), SelfOrganization(), Silence())
    sequence = ValidatedSequence(operators, context=CONTEXT)
    with patch.object(network_stage, "datetime", _EventClock):
        stage = execute_network_operator_stage(
            graph, operators[0], (PARENT,), sequence_context=sequence.step(0)
        )
    source = capture(graph)
    admission = admit_source(source)
    if (source.epi, source.phase, source.edges, source.conductances) != (
        initial.epi,
        initial.phase,
        initial.edges,
        initial.conductances,
    ):
        raise ValueError("capacity-only event changed excluded coordinates")
    after_val = record_graph_state(graph)
    field = numerical_field(source)
    states = [_state(graph)]
    numerical_steps = []
    snapshots = [source]
    pressures = [
        tuple(float(get_attr(graph.nodes[node], ALIAS_DNFR)) for node in graph)
    ]
    history = [(float(TIMES[0]), states[0][PARENT])]
    graph.nodes[PARENT]["epi_time_history"] = tuple(history)
    for before, after in zip(TIMES, TIMES[1:]):
        stages = []

        def recorded_field(time, state):
            rates = field(time, state)
            stages.append(
                {
                    "time": Q(time),
                    "state": tuple(map(Q, state)),
                    "rates": tuple(map(Q, rates)),
                }
            )
            return rates

        state = rk4_coupled_step(
            states[-1], recorded_field, t=float(before), dt=float(after - before)
        )
        residuals = tuple(
            Q(value)
            - Q(old)
            - (after - before)
            * sum(
                weight * row["rates"][i]
                for weight, row in zip((1, 2, 2, 1), stages, strict=True)
            )
            / 6
            for i, (old, value) in enumerate(zip(states[-1], state, strict=True))
        )
        numerical_steps.append(
            {
                "start": before,
                "end": after,
                "stages": tuple(stages),
                "represented_quadrature_residuals": residuals,
            }
        )
        for i, node in enumerate(graph):
            set_attr(graph.nodes[node], ALIAS_EPI, state[i])
            # A real phase lift is retained. No represented 2*pi wrapping.
            set_attr(graph.nodes[node], ALIAS_THETA, state[i + 20])
        graph.graph["_t"] = float(after)
        snapshots.append(refresh_pressure(graph))
        pressures.append(
            tuple(float(get_attr(graph.nodes[node], ALIAS_DNFR)) for node in graph)
        )
        states.append(state)
        history.append((float(after), state[PARENT]))
        graph.nodes[PARENT]["epi_time_history"] = tuple(history)
    certificate = certify_history(source, TIMES, tuple(states))
    before_checks = copy_graph_state(graph)
    observation = observe_structural_acceleration(graph, PARENT)
    validate_self_organization_strict(graph, PARENT, emit_warnings=False)
    grammar = validate_candidate(graph, PARENT, "THOL")
    threshold = resolve_thol_bifurcation_threshold(graph.graph)
    after_checks = copy_graph_state(graph)
    checks = {
        "local_stage_only": stage.glyph == "VAL" and stage.nodes_processed == 1,
        "same_law_parent_endpoint_accuracy": certificate.endpoint_budget_certified,
        "certified_represented_trigger": certificate.trigger_lower_bound > Q(threshold),
        "recorded_trigger": observation.available and observation.value > threshold,
        "recorded_endpoint_matches": observation.current_endpoint_matches_state,
        "explicit_clock_association": Q(graph.graph["_t"])
        == TIMES[-1]
        == Q(history[-1][0]),
        "fresh_parent_pressure_positive": all(
            value.lo > 0 for value in certificate.true_parent_pressure_bounds
        ),
        "strict_endpoint_preconditions": True,
        "actual_target_grammar": grammar.allowed,
        "prefix_debt_retained": node_debt(graph.nodes[PARENT]) == 1,
        "actual_prefix_retained": tuple(graph.nodes[PARENT]["glyph_history"])
        == ("VAL",),
        "no_birth_or_support_change": len(graph) == 20
        and capture(graph).edges == source.edges,
        "read_only_endpoint_checks": (
            after_checks.graph == before_checks.graph
            and dict(after_checks.nodes) == dict(before_checks.nodes)
            and dict(after_checks.edges) == dict(before_checks.edges)
        ),
    }
    endpoint = record_graph_state(graph)
    rho = tuple(
        Q(d) / nu for d, nu in zip(source.degrees, source.capacity, strict=True)
    )
    residual_evidence = []
    for time, state, snapshot, pressure in zip(
        TIMES, states, snapshots, pressures, strict=True
    ):
        exact_rates = snapshot.form_rates + snapshot.phase_rates
        calculated = field(float(time), state)
        residual_evidence.append(
            {
                "time": time,
                "stored_pressures": tuple(map(Q, pressure)),
                "pressure_defects": tuple(
                    I(Q(value)) - bound
                    for value, bound in zip(
                        pressure, _fresh_pressure_bounds(snapshot), strict=True
                    )
                ),
                "rate_defects": tuple(
                    I(Q(value)) - bound
                    for value, bound in zip(calculated, exact_rates, strict=True)
                ),
                "storage_change": snapshot.storage - source.storage,
                "form_mean_change": sum(
                    w * (x - x0)
                    for w, x, x0 in zip(rho, snapshot.epi, source.epi, strict=True)
                )
                / sum(rho),
                "phase_lift_mean_change": sum(
                    w * (x - x0)
                    for w, x, x0 in zip(rho, snapshot.phase, source.phase, strict=True)
                )
                / sum(rho),
            }
        )
    return {
        "schema": "tnfr.local-capacity-history-response.v1",
        "scope": "One declared represented preparation, actual local VAL and numerical temporal record; no THOL event",
        "preparation": preparation,
        "after_val": after_val,
        "endpoint": endpoint,
        "source_admission": encode_result(admission),
        "times": TIMES,
        "states": tuple(tuple(Q(value) for value in row) for row in states),
        "parent_history": tuple((Q(t), Q(x)) for t, x in history),
        "stage": encode_result(stage),
        "numerical_steps": tuple(numerical_steps),
        "residual_evidence": encode_result(residual_evidence),
        "certificate": encode_result(certificate),
        "observation": encode_result(observation),
        "threshold": Q(threshold),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
