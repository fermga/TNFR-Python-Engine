"""One prospectively frozen, same-time IL/OZ pressure-handoff study.

This owner reuses the positive registered contact, canonical stage executor and
weighted-law reader. Importing it evaluates nothing. The cached evaluator runs
one word, without a trajectory or a pressure-retention interval. Exact tagged
exports retain represented values, execution context and all event boundaries.

The module is marked slow so the routine engine gate does not regenerate the
frozen word. Current verification uses the retained-evidence test owner; replay
requires separate admission. The frozen wrapper calls these checks once against
the same captured result, without starting another evaluator instance.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from fractions import Fraction as Q
from functools import lru_cache
from threading import RLock

import mpmath
import networkx as nx
import numpy as np
import pytest

from tests.operators.test_collective_organization_contact import (
    _MODEL,
    _NOW,
    _registered_preparation,
    _SuppliedClock,
)
from tnfr._exact_time import finite_represented_real
from tnfr.alias import get_attr, set_dnfr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.constants.canonical import DELTA_PHI_MAX
from tnfr.dynamics import relational
from tnfr.mathematics._rational_interval import I, arg, cos, pi_interval, sin
from tnfr.node import NodeNX
from tnfr.operators import network_stage
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators.definitions import Coherence, Dissonance
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.grammar_types import (
    SequenceValidationResult,
    StructuralGrammarError,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.research.artifact_io import encode_exact_tree
from tnfr.sdk._state import copy_graph_state
from tnfr.sdk.simple import TNFR
from tnfr.types import Glyph
from tnfr.utils import angle_diff
from tnfr.utils.cache import GRAPH_RUNTIME_CACHE_KEYS

pytestmark = pytest.mark.slow

_TARGETS = (0, 1)
_BUDGET = Q(1, 2**46)
_CUTOFF = Q(1e-9)
_PRESSURE_CACHE_KEYS = frozenset({"_dnfrmax", "_dnfrmax_node"})
_NETWORKX_STORAGE = frozenset({"graph", "_node", "_adj", "__networkx_cache__"})
_NETWORKX_VIEWS = frozenset({"nodes", "edges", "degree", "adj"})


def _typename(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _copy_runtime_graph(graph):
    """Reuse SDK data copying, retaining this study's instance state and order.

    SDK copies omit declared rebuildable caches and admit external resources.
    This fixed simple-graph study additionally preserves its last-operator
    attribute and exact adjacency order. Unexpected instance state rejects
    instead of being silently discarded or assigned an unproved copy contract.
    """
    if type(graph) is not nx.Graph:
        raise TypeError("pressure-handoff source requires an ordinary simple Graph")
    extras = set(vars(graph)) - _NETWORKX_STORAGE - _NETWORKX_VIEWS
    if extras - {"_last_operator_applied"}:
        raise TypeError("pressure-handoff source has unsupported graph-instance state")
    layout = network_stage._networkx_runtime_layout(graph)
    copied = copy_graph_state(graph)
    network_stage._restore_detached_neighbor_order(copied, layout)
    if "_last_operator_applied" in extras:
        copied._last_operator_applied = deepcopy(graph._last_operator_applied)
    return copied


def _encode(value):
    """Strict lossless type projection; unsupported state is an error."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite evidence scalar")
        return {"type": "binary64", "hex": value.hex()}
    if type(value) is complex:
        # Complex components occur in admitted uniform-real BEPI containers
        # and auxiliary metadata. Projection does not admit nonreal form;
        # the shared signed-form owner independently checks the physical law.
        return {
            "type": "builtins.complex",
            "real": _encode(value.real),
            "imaginary": _encode(value.imag),
        }
    if type(value) is Q:
        return encode_exact_tree(value)
    if isinstance(value, np.generic):
        return {
            "type": _typename(value),
            "dtype": str(value.dtype),
            "value": _encode(value.item()),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": _typename(value),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "items": _encode(value.tolist()),
        }
    if isinstance(value, Enum):
        return {"type": _typename(value), "value": _encode(value.value)}
    if isinstance(value, datetime):
        return {"type": _typename(value), "iso": value.isoformat(), "fold": value.fold}
    if isinstance(value, Mapping):
        return {
            "type": _typename(value),
            "items": [[_encode(key), _encode(item)] for key, item in value.items()],
        }
    if type(value) in (tuple, list, set, frozenset, deque):
        result = {"type": _typename(value), "items": [_encode(item) for item in value]}
        if isinstance(value, deque):
            result["maxlen"] = value.maxlen
        return result
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": _typename(value),
            "fields": {
                field.name: _encode(getattr(value, field.name))
                for field in fields(value)
            },
        }
    raise TypeError(f"unsupported evidence type: {_typename(value)}")


def _state(graph, *, pressure_free=False):
    """Retain complete persistent state and declare rebuildable omissions."""
    excluded = GRAPH_RUNTIME_CACHE_KEYS | (
        _PRESSURE_CACHE_KEYS if pressure_free else frozenset()
    )
    return {
        "graph_type": _typename(graph),
        "node_order": tuple(graph),
        "neighbor_order": tuple((node, tuple(graph.neighbors(node))) for node in graph),
        "graph": deepcopy(
            {key: value for key, value in graph.graph.items() if key not in excluded}
        ),
        "nodes": tuple(
            (
                node,
                deepcopy(
                    {
                        key: value
                        for key, value in data.items()
                        if not (pressure_free and key in ALIAS_DNFR)
                    }
                ),
            )
            for node, data in graph.nodes(data=True)
        ),
        "edges": tuple(
            (left, right, deepcopy(data))
            for left, right, data in graph.edges(data=True)
        ),
        "object_state": deepcopy(
            {
                key: value
                for key, value in vars(graph).items()
                if key not in _NETWORKX_STORAGE | _NETWORKX_VIEWS
            }
        ),
        "omitted_rebuildable_caches": tuple(
            {
                "key": key,
                "type": _typename(value),
                "owner": "tnfr.utils.cache.GRAPH_RUNTIME_CACHE_KEYS",
                "scope": "rebuildable payload only; configuration and event state retained",
            }
            for key, value in graph.graph.items()
            if key in GRAPH_RUNTIME_CACHE_KEYS
        ),
        "omitted_networkx_views": tuple(
            {
                "key": key,
                "type": _typename(value),
                "scope": "view of retained graph storage",
            }
            for key, value in vars(graph).items()
            if key in _NETWORKX_VIEWS
        ),
        "omitted_networkx_cache": tuple(
            {
                "key": key,
                "type": _typename(value),
                "scope": "NetworkX rebuildable cache",
            }
            for key, value in vars(graph).items()
            if key == "__networkx_cache__"
        ),
    }


def _reference_guard(graph):
    """Keep observable container identities, including excluded cache payloads."""
    seen = set()

    def visit(value):
        if value is graph:
            return ("live_graph", id(value))
        if value is graph.graph:
            return ("live_graph_metadata", id(value))
        if value is None or type(value) in (bool, int, str, float, Q):
            return _encode(value)
        identity = id(value)
        if identity in seen:
            return ("reference", identity)
        seen.add(identity)
        if isinstance(value, Mapping):
            return (
                _typename(value),
                identity,
                tuple((visit(key), visit(item)) for key, item in value.items()),
            )
        if type(value) in (tuple, list, set, frozenset, deque):
            return (_typename(value), identity, tuple(map(visit, value)))
        if hasattr(value, "__dict__"):
            return (_typename(value), identity, visit(vars(value)))
        # These opaque leaves occur only inside explicitly excluded caches.
        return (_typename(value), identity)

    caches = tuple(
        (key, visit(value))
        for key, value in graph.graph.items()
        if key in GRAPH_RUNTIME_CACHE_KEYS
    )
    containers = (
        id(graph.graph),
        id(graph._node),
        id(graph._adj),
        tuple((node, id(data)) for node, data in graph.nodes(data=True)),
        tuple((left, right, id(data)) for left, right, data in graph.edges(data=True)),
    )
    return caches, containers, visit(vars(graph).get("__networkx_cache__"))


def _pressure(graph):
    return tuple(
        finite_represented_real(
            get_attr(
                graph.nodes[node], ALIAS_DNFR, strict=True, conv=lambda value: value
            ),
            f"stored pressure {node}",
        )[1]
        for node in graph
    )


def _fields(graph):
    return bound_relational_sine_support_exchange(graph, reference_model=_MODEL)


def _capture(graph):
    report = _fields(graph)
    return {
        "state": _state(graph),
        "fields": report,
        "stored_pressure": _pressure(graph),
        "acceleration": tuple(
            (node, observe_structural_acceleration(graph, node)) for node in graph
        ),
    }


def _refresh(graph, label, records):
    """Publish all admitted Phi midpoints, preserving the exact callback boundary."""
    before = _capture(graph)
    guard = _encode(_state(graph, pressure_free=True))
    references = _reference_guard(graph)
    values = tuple(
        finite_represented_real(interval.midpoint, f"{label} Phi midpoint {node}")
        for node, interval in zip(before["fields"].nodes, before["fields"].pressure)
    )
    residuals = tuple(
        I(represented) - interval
        for (_, represented), interval in zip(values, before["fields"].pressure)
    )
    for node, (materialized, _) in zip(before["fields"].nodes, values):
        set_dnfr(graph, node, materialized)
    preserved = (
        _encode(_state(graph, pressure_free=True)) == guard
        and _reference_guard(graph) == references
    )
    pressures = _pressure(graph)
    maximum = max(map(abs, pressures))
    witness = graph.graph["_dnfrmax_node"]
    cache_valid = (
        Q(graph.graph["_dnfrmax"]) == maximum
        and abs(pressures[tuple(graph).index(witness)]) == maximum
    )
    after = _capture(graph)
    records.append(
        {
            "label": label,
            "before": before,
            "after": after,
            "materialized": tuple(represented for _, represented in values),
            "residuals": residuals,
            "preservation_passed": preserved,
            "cache_valid": cache_valid,
            "allowed_writes": (
                "first authoritative pressure alias per node",
                "_dnfrmax",
                "_dnfrmax_node",
            ),
            "cache_witness": witness,
        }
    )
    if not preserved or not cache_valid:
        raise AssertionError(
            "selected-law callback violated pressure-only preservation/cache contract"
        )


def _wrapped(interval):
    return arg(cos(interval), sin(interval))


def _geometry(report):
    alpha = 2 * pi_interval() / 5
    errors = tuple(
        _wrapped(I(phase) - (node % 10 // 2) * alpha)
        for node, phase in zip(report.nodes, report.phase)
    )
    form_mean = sum(report.epi) / len(report.nodes)
    error_mean = sum(errors, I(0)) / len(report.nodes)
    norm = sum(((value - form_mean) ** 2 for value in report.epi), Q(0)) + sum(
        ((error - error_mean) ** 2 for error in errors), I(0)
    )
    excess = report.storage - 40 * (1 - cos(alpha))
    return {
        "continuous_phase_errors": errors,
        "joint_norm_squared": norm,
        "storage_excess": excess,
    }


def _il_measurements(before, after):
    old, new = before["fields"], after["fields"]
    coefficient = Q(DEFAULT_PHASE_LOCKING_COEFFICIENT)
    displacement = tuple(
        _wrapped(I(new.phase[node] - old.phase[node])) for node in _TARGETS
    )
    ideal = tuple(coefficient * arg(*old.relative_resultant[node]) for node in _TARGETS)
    delta = I(old.phase[1] - old.phase[0])
    left, right = displacement
    correction = -4 * cos(delta + (right - left) / 2) * sin(left / 2) * sin(right / 2)
    isolated = I(0)
    for target, change in zip(_TARGETS, displacement):
        for i, j in old.edges:
            if target in (i, j):
                other = j if i == target else i
                gap = I(old.phase[other] - old.phase[target])
                isolated += cos(gap) - cos(gap - change)
    work = new.storage - old.storage
    ideal_changes = dict(zip(_TARGETS, ideal))
    ideal_work = sum(
        (
            cos(I(old.phase[j] - old.phase[i]))
            - cos(
                I(old.phase[j] - old.phase[i])
                + ideal_changes.get(j, I(0))
                - ideal_changes.get(i, I(0))
            )
            for i, j in old.edges
        ),
        I(0),
    )
    mean_jump = 7 * (left + right) / 52
    actual_mean_jump = (
        sum(
            (Q(degree) / capacity) * _wrapped(I(after_phase - before_phase))
            for degree, capacity, before_phase, after_phase in zip(
                old.degrees, old.capacity, old.phase, new.phase
            )
        )
        / 104
    )
    mp = mpmath.mp.clone()
    mp.dps = 100

    def mpq(value):
        return mp.mpf(value.numerator) / value.denominator

    oracle = []
    for target, change in zip(_TARGETS, displacement):
        neighbors = tuple(
            j if i == target else i for i, j in old.edges if target in (i, j)
        )
        resultant = sum(
            mp.exp(mp.j * mpq(old.phase[node] - old.phase[target]))
            for node in neighbors
        )
        exact_change = mpq(coefficient) * mp.arg(resultant)
        oracle.append(
            {
                "node": target,
                "dps": 100,
                "ideal_displacement_decimal": mp.nstr(exact_change, 105),
                "represented_defect_decimal": mp.nstr(
                    mpq(change.midpoint) - exact_change, 105
                ),
            }
        )
    return {
        "displacements": displacement,
        "ideal_displacements": ideal,
        "phase_defects": tuple(
            actual - exact for actual, exact in zip(displacement, ideal)
        ),
        "work": work,
        "ideal_work": ideal_work,
        "represented_work_defect": work - ideal_work,
        "single_target_work_sum": isolated,
        "shared_edge_correction": correction,
        "work_identity_residual": work - isolated - correction,
        "mean_phase_jump": mean_jump,
        "actual_mean_phase_jump": actual_mean_jump,
        "global_rho": sum(
            Q(degree) / capacity for degree, capacity in zip(old.degrees, old.capacity)
        ),
        "geometry": _geometry(new),
        "oracle100": tuple(oracle),
    }


def _oz_measurements(before, raw, settings):
    """Independent local and ordered-product readout, including suppressed edges."""
    report = before["fields"]
    pressures = before["stored_pressure"]
    factor = settings["OZ_dnfr_factor"]
    threshold = settings["OZ_PHASE_THRESHOLD"]
    minimum = settings["OZ_MIN_PROPAGATION"]
    phase = report.phase
    local, branches, candidates = {}, [], []
    incoming = {node: [] for node in report.nodes}
    neighbors = dict(before["state"]["neighbor_order"])
    weights = {
        frozenset(edge): weight
        for edge, weight in zip(report.edges, report.conductances)
    }
    for target in _TARGETS:
        p = pressures[target]
        exact_local = factor * p if abs(p) > _CUTOFF else Q(0.1)
        represented_local = Q(float(exact_local))
        local[target] = represented_local
        bounds = abs(report.pressure[target])
        branch = (
            "amplification"
            if bounds.lo > _CUTOFF
            else "near_zero_seed" if bounds.hi <= _CUTOFF else "unresolved"
        )
        branches.append(
            {
                "node": target,
                "represented": (
                    "amplification" if abs(p) > _CUTOFF else "near_zero_seed"
                ),
                "ideal": branch,
                "local_exact": exact_local,
                "local_represented": represented_local,
                "local_residual": represented_local - exact_local,
            }
        )
        magnitude = abs(float(represented_local) - float(p))
        for other in neighbors[target]:
            weight = weights[frozenset((target, other))]
            gap = abs(angle_diff(float(phase[target]), float(phase[other])))
            ideal_gap = abs(_wrapped(I(phase[target] - phase[other])))
            admitted = gap <= float(threshold)
            phase_weight = 1.0 - gap / float(threshold) if admitted else None
            products, residuals, vanished = [], [], []
            value = magnitude
            if admitted:
                for multiplier in (float(weight), phase_weight, 1.0):
                    exact = Q(value) * Q(multiplier)
                    value *= multiplier
                    finite_represented_real(value, "OZ candidate product")
                    products.append(Q(value))
                    residuals.append(Q(value) - exact)
                    vanished.append(exact != 0 and value == 0)
            accepted = admitted and value != 0.0 and value >= float(minimum)
            if accepted:
                incoming[other].append((target, value))
            candidates.append(
                {
                    "source": target,
                    "neighbor": other,
                    "magnitude": Q(magnitude),
                    "magnitude_residual": Q(magnitude) - abs(represented_local - p),
                    "weight": weight,
                    "wrapped_gap": Q(gap),
                    "gap_residual": I(Q(gap)) - ideal_gap,
                    "phase_admitted": admitted,
                    "phase_weight": None if phase_weight is None else Q(phase_weight),
                    "phase_weight_residual": (
                        None
                        if phase_weight is None
                        else Q(phase_weight) - (1 - Q(gap) / threshold)
                    ),
                    "products": tuple(products),
                    "product_residuals": tuple(residuals),
                    "nonzero_lost": tuple(vanished),
                    "accepted": accepted,
                }
            )
    merged = tuple(
        Q(
            math.fsum(
                [
                    float(local.get(node, pressures[node])),
                    *(value for _, value in sorted(incoming[node])),
                ]
            )
        )
        for node in report.nodes
    )
    return {
        "branches": tuple(branches),
        "candidates": tuple(candidates),
        "merged_pressure": merged,
        "merge_residual": tuple(
            actual - expected
            for actual, expected in zip(raw["stored_pressure"], merged)
        ),
        "continuous_coordinates_unchanged": all(
            getattr(report, field) == getattr(raw["fields"], field)
            for field in ("nodes", "edges", "conductances", "epi", "phase", "capacity")
        ),
    }


@lru_cache(maxsize=1)
def evaluate_handoff():
    """Evaluate the frozen word once; callers must freeze before invoking."""
    _, _, contact, _ = _registered_preparation((0, 1))
    source = contact[-1]
    # Observe copies so NetworkX's lazy view installation cannot mutate the
    # retained preparation merely by reading its fields or export inventory.
    source_snapshot = _encode(_state(_copy_runtime_graph(source)))
    source_cache_inventory = tuple(
        {
            "key": key,
            "type": _typename(value),
            "scope": "rebuildable payload omitted by shared SDK copy; configuration retained",
        }
        for key, value in source.graph.items()
        if key in GRAPH_RUNTIME_CACHE_KEYS
    )
    source_cache_references = tuple(
        (key, id(value))
        for key, value in source.graph.items()
        if key in GRAPH_RUNTIME_CACHE_KEYS
    )
    graph = _copy_runtime_graph(source)
    graph.graph.update(OZ_NOISE_MODE=False, OZ_ENABLE_PROPAGATION=True)
    operators = (Coherence(), Dissonance())
    tokens = tuple(
        TNFR.operators(operator.glyph.value)["token"] for operator in operators
    )
    context = {"initial_epi_nonzero": True, "compatibility_profile": "core"}
    settings = {
        "phase_coefficient": Q(DEFAULT_PHASE_LOCKING_COEFFICIENT),
        "OZ_PHASE_THRESHOLD": Q(graph.graph.get("OZ_PHASE_THRESHOLD", DELTA_PHI_MAX)),
        "OZ_MIN_PROPAGATION": Q(graph.graph.get("OZ_MIN_PROPAGATION", 0.05)),
        "OZ_NOISE_MODE": False,
        "OZ_ENABLE_PROPAGATION": True,
        "propagation_mode": "phase_weighted",
        "absolute_budget": _BUDGET,
        "oracle_dps": 100,
        "clock": _NOW,
        "targets": _TARGETS,
        "tokens": tokens,
        "grammar_context": context,
    }
    records, stages, captures = (
        [],
        [],
        {
            "inherited": _capture(_copy_runtime_graph(source)),
            "configured": _capture(graph),
        },
    )
    result = {
        "schema": "tnfr.collective-pressure-handoff.v1",
        "status": "pending",
        "settings": settings,
        "admission": None,
        "captures": captures,
        "refreshes": records,
        "stages": stages,
        "source_preserved": None,
        "source_cache_inventory": source_cache_inventory,
        "copy_owner": "tnfr.sdk._state.copy_graph_state plus declared instance-state/neighbor-order adapter",
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("pressure-handoff protocol permits no trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        active_stage = "grammar_preflight"
        try:
            if not all(
                captures["configured"]["fields"].epi[node] != 0 for node in _TARGETS
            ):
                raise ValueError(
                    "initialized-EPI grammar context is not supplied by target state"
                )
            result["admission"] = validate_sequence(list(tokens), context=context)
            if not result["admission"].passed:
                raise ValueError(
                    "declared core word failed configured grammar admission"
                )
            sequence = ValidatedSequence(operators, context=context)
            active_stage = "configuration_preflight"
            il = resolve_runtime_operator_factors(
                graph.graph.get("GLYPH_FACTORS"), Glyph.IL, graph.graph
            )
            oz = resolve_runtime_operator_factors(
                graph.graph.get("GLYPH_FACTORS"), Glyph.OZ, graph.graph
            )
            settings.update(
                IL_dnfr_factor=Q(il["IL_dnfr_factor"]),
                OZ_dnfr_factor=Q(oz["OZ_dnfr_factor"]),
            )
            active_stage = "initial_refresh"
            _refresh(graph, "initial", records)
            captures["initial"] = records[-1]["after"]
            for index, operator in enumerate(operators):
                label = operator.glyph.value
                active_stage = f"{label}_stage"

                def refresh(subject, label=label):
                    _refresh(subject, label, records)

                stage = execute_network_operator_stage(
                    graph,
                    operator,
                    _TARGETS,
                    sequence_context=sequence.step(index),
                    compute_delta_nfr=refresh,
                    include_epi_jump_certificate=True,
                )
                stages.append(stage)
                captures[f"{label}_raw"] = records[-1]["before"]
                captures[f"{label}_refreshed"] = records[-1]["after"]
                captures[f"{label}_completed"] = _capture(graph)
            result["status"] = "completed"
        except Exception as exc:
            if active_stage == "grammar_preflight" or isinstance(
                exc, StructuralGrammarError
            ):
                category = "grammar_rejected"
            elif isinstance(exc, OperatorPreconditionError):
                category = "live_state_rejected"
            elif active_stage == "configuration_preflight":
                category = "configuration_rejected"
            elif isinstance(exc, (ArithmeticError, AssertionError, ValueError)):
                category = "arithmetic_or_evidence_failure"
            else:
                category = "unexpected_failure"
            result["status"] = category
            result["failure"] = {
                "category": category,
                "stage": active_stage,
                "type": _typename(exc),
                "message": str(exc),
                "args": exc.args,
                "attributes": vars(exc),
                "completed_stages": len(stages),
                "state": _capture(graph),
            }
    result["source_preserved"] = (
        _encode(_state(_copy_runtime_graph(source))) == source_snapshot
        and tuple(
            (key, id(value))
            for key, value in source.graph.items()
            if key in GRAPH_RUNTIME_CACHE_KEYS
        )
        == source_cache_references
    )
    if result["status"] == "completed":
        try:
            result["IL"] = _il_measurements(captures["initial"], captures["IL_raw"])
            result["OZ"] = _oz_measurements(
                captures["IL_completed"], captures["OZ_raw"], settings
            )
            result["final_geometry"] = _geometry(captures["OZ_completed"]["fields"])
        except Exception as exc:
            result["analysis_failure"] = {
                "type": _typename(exc),
                "message": str(exc),
                "args": exc.args,
                "attributes": vars(exc),
            }
    return result


def evidence_export(result=None):
    """Return strict tagged evidence; this is a projection, not authentication."""

    def record(value):
        if isinstance(value, SequenceValidationResult):
            return {
                "type": _typename(value),
                **{
                    name: record(getattr(value, name))
                    for name in (
                        "subject",
                        "passed",
                        "summary",
                        "artifacts",
                        "tokens",
                        "canonical_tokens",
                        "message",
                        "metadata",
                    )
                },
                "error": (
                    None
                    if value.error is None
                    else {
                        "type": _typename(value.error),
                        "attributes": record(vars(value.error)),
                    }
                ),
            }
        if isinstance(value, Mapping):
            if any(type(key) is not str for key in value):
                return _encode(value)
            return {
                key: _encode(item) if key == "state" else record(item)
                for key, item in value.items()
            }
        if type(value) in (tuple, list):
            return [record(item) for item in value]
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "type": _typename(value),
                **{
                    field.name: record(getattr(value, field.name))
                    for field in fields(value)
                },
            }
        return _encode(value)

    return encode_exact_tree(record(evaluate_handoff() if result is None else result))


@pytest.fixture(scope="module")
def handoff():
    return evaluate_handoff()


def test_actual_word_preserves_source_and_records_all_stage_boundaries(handoff):
    assert handoff["status"] == "completed", handoff.get("failure")
    assert handoff["source_preserved"]
    assert "analysis_failure" not in handoff, handoff.get("analysis_failure")
    assert handoff["admission"].passed
    assert [record["label"] for record in handoff["refreshes"]] == [
        "initial",
        "IL",
        "OZ",
    ]
    assert len(handoff["captures"]) == 9
    for stage in handoff["stages"]:
        assert stage.schedule == "two_phase_jacobi" and stage.nodes_processed == 2
        assert stage.epi_jump_certificate is None
        assert stage.epi_jump_certificate_abstention_reason is not None
    evidence_export(handoff)


def test_selected_pressure_writer_has_bounded_residuals_and_no_other_effect(handoff):
    for record in handoff["refreshes"]:
        assert record["preservation_passed"]
        assert record["cache_valid"]
        assert all(residual.abs_max <= _BUDGET for residual in record["residuals"])
        before, after = record["before"], record["after"]
        assert before["fields"] == after["fields"]
        assert after["stored_pressure"] == record["materialized"]
        assert all(
            not observation.available and observation.value is None
            for _, observation in after["acceleration"]
        )


def test_full_il_phase_action_and_adjacent_edge_work_are_retained(handoff):
    measured = handoff["IL"]
    assert all(defect.abs_max <= _BUDGET for defect in measured["phase_defects"])
    assert measured["work_identity_residual"].abs_max <= _BUDGET
    assert measured["represented_work_defect"].abs_max <= 14 * _BUDGET
    assert (
        measured["actual_mean_phase_jump"] - measured["mean_phase_jump"]
    ).abs_max <= _BUDGET
    assert measured["work"].hi < 0
    assert measured["global_rho"] == 104
    assert measured["geometry"]["joint_norm_squared"].hi < Q(1, 16) ** 2
    assert measured["geometry"]["storage_excess"].hi < Q(1, 6400)
    old = handoff["captures"]["initial"]
    raw = handoff["captures"]["IL_raw"]
    for field in ("nodes", "edges", "conductances", "epi", "capacity"):
        assert getattr(old["fields"], field) == getattr(raw["fields"], field)
    factor = handoff["settings"]["IL_dnfr_factor"]
    for node in old["fields"].nodes:
        expected = old["stored_pressure"][node] * (factor if node in _TARGETS else 1)
        assert abs(raw["stored_pressure"][node] - expected) <= _BUDGET
        if node not in _TARGETS:
            assert raw["fields"].phase[node] == old["fields"].phase[node]
    assert raw["fields"].phase_rates == old["fields"].phase_rates
    assert any(
        not (after - before).contains(0)
        for before, after in zip(old["fields"].form_rates, raw["fields"].form_rates)
    )


def test_oz_raw_local_and_propagated_values_match_declared_arithmetic(handoff):
    measured = handoff["OZ"]
    assert measured["continuous_coordinates_unchanged"]
    assert all(abs(residual) <= _BUDGET for residual in measured["merge_residual"])
    for branch in measured["branches"]:
        assert abs(branch["local_residual"]) <= _BUDGET
        assert branch["ideal"] != "unresolved"
        assert branch["ideal"] == branch["represented"]
    for candidate in measured["candidates"]:
        assert not any(candidate["nonzero_lost"])
        assert candidate["gap_residual"].abs_max <= _BUDGET
        assert abs(candidate["magnitude_residual"]) <= _BUDGET
        if candidate["phase_weight_residual"] is not None:
            assert abs(candidate["phase_weight_residual"]) <= _BUDGET
        assert all(
            abs(residual) <= _BUDGET for residual in candidate["product_residuals"]
        )


def test_immediate_oz_refresh_restores_same_complete_continuous_state(handoff):
    captures = handoff["captures"]
    il, raw, final = (
        captures[key] for key in ("IL_completed", "OZ_raw", "OZ_completed")
    )
    assert il["fields"] == raw["fields"] == final["fields"]
    assert il["stored_pressure"] == final["stored_pressure"]
    assert raw["stored_pressure"] != il["stored_pressure"]
    assert _encode(il["state"]) != _encode(final["state"])
    assert handoff["final_geometry"]["joint_norm_squared"].hi < Q(1, 256)
    assert handoff["final_geometry"]["storage_excess"].hi < Q(1, 6400)
    for node in _TARGETS:
        old = dict(il["state"]["nodes"])[node]["glyph_history"]
        new = dict(final["state"]["nodes"])[node]["glyph_history"]
        assert old != new


def test_evidence_projection_preserves_signed_zero_and_rejects_unknown_state():
    assert _encode(0.0) != _encode(-0.0)
    assert _encode(1.0) != _encode(1)
    with pytest.raises(TypeError, match="unsupported evidence type"):
        _encode(object())


def test_complex_auxiliary_evidence_keeps_components_without_scalar_admission():
    encoded = _encode(complex(-0.0, 0.25))
    assert encoded == {
        "type": "builtins.complex",
        "real": _encode(-0.0),
        "imaginary": _encode(0.25),
    }
    assert encoded != _encode(complex(0.0, 0.25))
    for value in (complex(float("inf"), 0), complex(0, float("nan"))):
        with pytest.raises(ValueError, match="nonfinite evidence scalar"):
            _encode(value)
    with pytest.raises((TypeError, ValueError)):
        finite_represented_real(complex(-0.0, 0.25), "physical scalar")


def test_runtime_graph_copy_excludes_locks_and_preserves_event_state_and_order():
    graph = nx.Graph(RANDOM_SEED=17, _t=7, policy={"values": [1, 2]})
    graph.add_nodes_from((0, 1, 2))
    graph.add_edge(2, 1, weight=0.75, length=3, evidence=["first"])
    graph.add_edge(0, 1, weight=1.0, length=2, evidence=["second"])
    for node in graph:
        graph.nodes[node].update(
            EPI=0.5, theta=(node + 1) / 10, nu_f=1.0, delta_nfr=0.2
        )
    graph.nodes[0].update(
        EPI=0.5,
        theta=0.1,
        nu_f=1.0,
        delta_nfr=0.2,
        glyph_history=deque(("AL", "SHA"), maxlen=8),
        lineage={"events": ["origin"]},
    )
    graph._last_operator_applied = "silence"
    adapter = NodeNX.from_graph(graph, 0)
    graph.graph["_tnfr_cache_manager"] = {"lock": RLock(), "values": ["cached"]}
    original_cache = graph.graph["_node_cache"]
    original_neighbors = tuple(graph.neighbors(1))
    with pytest.raises(TypeError, match="RLock"):
        deepcopy(graph)

    copied = _copy_runtime_graph(graph)

    assert GRAPH_RUNTIME_CACHE_KEYS.isdisjoint(copied.graph)
    assert copied._last_operator_applied == "silence"
    assert tuple(copied) == tuple(graph)
    assert tuple(copied.neighbors(1)) == original_neighbors == (2, 0)
    assert _encode(copied.graph) == _encode(
        {
            key: value
            for key, value in graph.graph.items()
            if key not in GRAPH_RUNTIME_CACHE_KEYS
        }
    )
    assert _encode(tuple(copied.nodes(data=True))) == _encode(
        tuple(graph.nodes(data=True))
    )
    assert _encode(tuple(copied.edges(data=True))) == _encode(
        tuple(graph.edges(data=True))
    )
    assert graph.graph["_node_cache"] is original_cache
    assert original_cache[0] is adapter
    copied_adapter = NodeNX.from_graph(copied, 0)
    assert copied_adapter is not adapter and copied_adapter.G is copied
    records = []
    _refresh(copied, "synthetic_copy_check", records)
    assert records[0]["preservation_passed"] and records[0]["cache_valid"]
    evidence_export({"refresh": records[0]})
    copied.nodes[0]["glyph_history"].append("IL")
    copied.nodes[0]["lineage"]["events"].append("copy")
    copied.graph["policy"]["values"].append(3)
    copied.edges[2, 1]["evidence"].append("copy")
    assert list(graph.nodes[0]["glyph_history"]) == ["AL", "SHA"]
    assert graph.nodes[0]["lineage"] == {"events": ["origin"]}
    assert graph.graph["policy"] == {"values": [1, 2]}
    assert graph.edges[2, 1]["evidence"] == ["first"]


def test_runtime_graph_copy_rejects_unowned_instance_state():
    graph = nx.Graph()
    graph.extra_context = {"events": []}
    with pytest.raises(TypeError, match="unsupported graph-instance state"):
        _copy_runtime_graph(graph)
