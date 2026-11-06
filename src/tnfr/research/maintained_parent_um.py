"""Independent ideal UM selection on the retained maintained-parent successor.

This detached calculation calls neither an operator nor a proposal. Runtime
coefficients are exact represented inputs to a mathematical reset, whose
transcendental values are enclosed independently. The separate numerical
allowance is a prospective acceptance budget, not a proved libm error bound.
Artifact association and the preceding birth certificate belong to the source
consumer; equal primitive coordinates do not authenticate their acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from math import pi, tau

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..alias import get_attr
from ..constants import DEFAULTS
from ..constants.aliases import ALIAS_DNFR, ALIAS_SI
from ..constants.canonical import UM_COMPAT_THRESHOLD
from ..mathematics._rational_interval import I, arg, cos, pi_interval, sin
from ..operators._phase_gate import resolve_u3_phase_limits
from ..operators.factor_contracts import (
    _resolve_coupling_flags,
    resolve_runtime_operator_factors,
)
from ..physics.relational_sine_support import bound_relational_sine_frozen_isolates
from ..types import Glyph
from .local_capacity_history import MODEL, TIMES

PARENT = 2
CHILD = "2_sub_0"
NUMERICAL_ALLOWANCE = Q(1, 2**40)
DECISION_GUARD = 4 * NUMERICAL_ALLOWANCE
_EPSILON = Q(1e-9)


@dataclass(frozen=True)
class UmCandidatePrediction:
    """One complete nonneighbor inventory entry, including rejected entries."""

    candidate_index: int
    pre_distance: I
    pre_admitted: bool
    prefilter_margin: I
    post_distance: I | None
    postfilter_margin: I | None
    score: I | None
    score_margin: I | None
    selected: bool


@dataclass(frozen=True)
class UmResetPrediction:
    """Ideal reset enclosures, with a separate conditional rounding budget.

    Edges use indices into ``nodes``. ``phase_after`` retains the original
    real lift at unchanged nodes and the interior zero-to-tau chart at changed
    nodes. This is an explicitly selected lift, not an intrinsic circular mean.
    No numerical allowance is folded into an ideal field below.
    """

    nodes: tuple
    parent: int
    epi: tuple[Q, ...]
    phase_before: tuple[Q, ...]
    phase_after: tuple[I, ...]
    capacity_before: tuple[Q, ...]
    capacity_after: tuple[I, ...]
    raw_pressure_before: tuple[Q, ...]
    raw_pressure_after: tuple[I, ...]
    sense_indices: tuple[Q, ...]
    old_edges: tuple[tuple[int, int], ...]
    old_conductances: tuple[Q, ...]
    new_edges: tuple[tuple[int, int], ...]
    new_conductances: tuple[I, ...]
    compatible_indices: tuple[int, ...]
    phase_changed_indices: tuple[int, ...]
    capacity_changed_indices: tuple[int, ...]
    consensus: I
    consensus_real: I
    consensus_imaginary: I
    phase_factor: Q
    capacity_factor: Q
    pressure_factor: Q
    phase_limit: Q
    score_threshold: Q
    existing_pre_margins: tuple[I, ...]
    existing_post_margins: tuple[I, ...]
    candidates: tuple[UmCandidatePrediction, ...]
    parent_alignment: I
    numerical_allowance: Q = NUMERICAL_ALLOWANCE


def _represented(value, label):
    exact = exact_or_represented_real(value, label)
    represented = finite_represented_real(value, label)[1]
    if exact != represented:
        raise ValueError(f"{label} must already be exactly represented in binary64")
    return represented


def _raw(graph, node, aliases, default=None):
    return get_attr(graph.nodes[node], aliases, default, strict=True, conv=lambda x: x)


def principal_difference(left, right):
    """Enclose the signed shortest arc only on one certified regular branch.

    This exact mathematical operation uses true pi. The selected source has
    strict branch margins. Inputs meeting or crossing a branch cut reject;
    no midpoint chooses a branch when the whole interval is unresolved.
    """
    difference = I.coerce(left) - I.coerce(right)
    p = pi_interval()
    for winding in range(-2, 3):
        candidate = difference - 2 * winding * p
        if candidate.lo > -p.lo and candidate.hi < p.lo:
            return candidate
    raise ValueError("phase interval lacks a certified regular shortest-arc branch")


def _signed_gate_margin(distance, limit):
    """Positive means admitted; negative means excluded, with no fuzzy tie."""
    return I(limit) - distance


def _decision(margin, label):
    """Require four prospective allowances of clearance, independently of sign.

    This leaves room for two emitted phase errors and one later distance
    evaluation error. A score also has its own prospective arithmetic budget;
    its phase sensitivity is at most 1/(2*pi) per incident endpoint. These
    implications are conditional until an actual response checks its defects.
    """
    if margin.lo > DECISION_GUARD:
        return True
    if margin.hi < -DECISION_GUARD:
        return False
    raise ValueError(f"{label} lacks the declared numerical decision margin")


def _admit_configuration(graph):
    data = graph.graph
    if any(str(key).startswith("UM_") for key in data):
        raise ValueError("selected prediction requires absent UM overrides")
    if "_node_sample" in data or "DELTA_PHI_MAX" in data:
        raise ValueError("selected prediction requires the full default candidate gate")
    if data.get("GLYPH_FACTORS") != {"VAL_scale": 128.0}:
        raise ValueError("selected prediction retains only the preceding VAL factor")
    if (
        type(data.get("RANDOM_SEED")) is not int
        or data["RANDOM_SEED"] != 17
        or data.get("GRAMMAR_REJECTION_MODE") != "raise"
        or data.get("VALIDATE_OPERATOR_PRECONDITIONS") is not True
        or data.get("EDGE_AWARE_ENABLED") is not False
    ):
        raise ValueError(
            "selected prediction requires unchanged strict execution policy"
        )
    if _represented(data.get("_t"), "event clock") != TIMES[-1]:
        raise ValueError("selected prediction requires the retained endpoint clock")
    flags = _resolve_coupling_flags(data)
    if not all(
        (
            flags.bidirectional,
            flags.sync_vf,
            flags.stabilize_dnfr,
            flags.functional_links,
        )
    ):
        raise ValueError("selected prediction requires all four coupling branches")
    factors = resolve_runtime_operator_factors(
        data.get("GLYPH_FACTORS"), Glyph.UM, data
    )
    resolved = tuple(
        _represented(factors[key], key)
        for key in ("UM_theta_push", "UM_vf_sync", "UM_dnfr_reduction")
    )
    if any(not 0 < factor < 1 for factor in resolved):
        raise ValueError("selected default factors must be interior")
    _, phase_limit = resolve_u3_phase_limits(data, operator_code="UM")
    threshold = _represented(
        DEFAULTS.get("UM_COMPAT_THRESHOLD", UM_COMPAT_THRESHOLD), "score threshold"
    )
    if threshold != Q(UM_COMPAT_THRESHOLD):
        raise ValueError("selected score threshold requires the canonical default")
    return (*resolved, _represented(phase_limit, "phase limit"), threshold)


def _score(parent_form, candidate_form, parent_si, candidate_si, distance):
    """Exact simplification of the normalized form-affinity quotient.

    All admitted forms are positive, so cancelling the same positive scale
    leaves abs(x-y)/(x+y+epsilon). Intermediate runtime rounding is excluded
    from this mathematical identity and has a separately declared budget.
    """
    form_similarity = 1 - abs(parent_form - candidate_form) / (
        parent_form + candidate_form + _EPSILON
    )
    sense_similarity = 1 - abs(parent_si - candidate_si)
    phase_strength = 1 - distance / Q(pi)
    return phase_strength / 2 + Q(1, 4) * (form_similarity + sense_similarity)


def predict_parent_um(graph):
    """Re-admit primitive state and derive the one-target canonical ideal map.

    The public source consumer additionally checks retained bytes, history,
    birth and actual grammar. This function admits the mathematical map's
    primitive domain and defaults without trusting any stored report fields.
    Every nonneighbor appears in the returned inventory. Unresolved threshold
    decisions reject instead of being rounded into a favorable classification.
    """
    phase_factor, capacity_factor, pressure_factor, limit, threshold = (
        _admit_configuration(graph)
    )
    source = bound_relational_sine_frozen_isolates(graph, reference_model=MODEL)
    for label, values in (
        ("form", source.epi),
        ("phase", source.phase),
        ("capacity", source.capacity),
        ("conductance", source.conductances),
    ):
        for value in values:
            _represented(value, label)
    nodes = source.nodes
    if nodes != tuple(range(20)) + (CHILD,):
        raise ValueError(
            "selected prediction requires the retained twenty-one-node order"
        )
    positions = {node: index for index, node in enumerate(nodes)}
    if tuple(graph.neighbors(PARENT)) != (0, 1, 4, 5) or graph.degree(CHILD) != 0:
        raise ValueError(
            "selected prediction requires original parent support and isolate"
        )
    if source.capacity != (Q(1, 2), Q(1, 2), Q(128)) + (Q(1),) * 17 + (Q(121.6),):
        raise ValueError("selected prediction requires unchanged retained capacities")
    if not all(0 < value < 1 for value in source.epi):
        raise ValueError(
            "selected form-affinity chart requires positive forms below one"
        )
    if any(not -Q(pi) < value < Q(tau) for value in source.phase):
        raise ValueError("selected prediction requires the retained real phase lifts")
    sense = tuple(
        _represented(_raw(graph, node, ALIAS_SI, 0.0), "sense index") for node in nodes
    )
    if sense != (Q(1, 2),) * 20 + (Q(0),):
        raise ValueError(
            "selected sense indices require old one-half and child default zero"
        )
    pressure = tuple(
        _represented(_raw(graph, node, ALIAS_DNFR), "stored pressure") for node in nodes
    )
    phase = tuple(I(value) for value in source.phase)
    neighbor_indices = tuple(positions[node] for node in graph.neighbors(PARENT))
    pre_margins = tuple(
        _signed_gate_margin(
            abs(principal_difference(phase[PARENT], phase[index])), limit
        )
        for index in neighbor_indices
    )
    compatible = tuple(
        index
        for index, margin in zip(neighbor_indices, pre_margins)
        if _decision(margin, "existing-neighbor prefilter")
    )
    if not compatible:
        raise ValueError("selected UM needs an existing compatible neighbor")
    inputs = (PARENT, *compatible)
    real = sum((cos(phase[index]) for index in inputs), I(0))
    imaginary = sum((sin(phase[index]) for index in inputs), I(0))
    consensus = arg(real, imaginary)
    after = list(phase)
    changed = tuple(sorted(inputs))
    for index in changed:
        after[index] = phase[index] + phase_factor * principal_difference(
            consensus, phase[index]
        )
        if (
            not NUMERICAL_ALLOWANCE < after[index].lo
            or not after[index].hi < Q(tau) - NUMERICAL_ALLOWANCE
        ):
            raise ValueError("selected reset lacks an interior zero-to-tau phase chart")
    post_margins = tuple(
        _signed_gate_margin(
            abs(principal_difference(after[PARENT], after[index])), limit
        )
        for index in compatible
    )
    if not all(
        _decision(margin, "existing-neighbor post-merge gate")
        for margin in post_margins
    ):
        raise ValueError("ideal reset violates an existing-neighbor post-merge gate")
    capacity_after = [I(value) for value in source.capacity]
    mean_capacity = sum((source.capacity[index] for index in compatible), Q(0)) / len(
        compatible
    )
    capacity_after[PARENT] = I(
        source.capacity[PARENT]
        + capacity_factor * (mean_capacity - source.capacity[PARENT])
    )
    alignment = sum(
        (
            1 - abs(principal_difference(after[PARENT], after[index])) / Q(pi)
            for index in compatible
        ),
        I(0),
    ) / len(compatible)
    pressure_after = [I(value) for value in pressure]
    pressure_after[PARENT] = I(pressure[PARENT]) * (1 - pressure_factor * alignment)
    candidates = []
    new_edges, new_weights = [], []
    for index, node in enumerate(nodes):
        if node == PARENT or graph.has_edge(PARENT, node):
            continue
        pre_distance = abs(principal_difference(phase[PARENT], phase[index]))
        pre_margin = _signed_gate_margin(pre_distance, limit)
        pre_admitted = _decision(pre_margin, "candidate prefilter")
        post_distance = post_margin = score = score_margin = None
        selected = False
        if pre_admitted:
            post_distance = abs(principal_difference(after[PARENT], after[index]))
            post_margin = _signed_gate_margin(post_distance, limit)
            if _decision(post_margin, "candidate post-merge phase gate"):
                score = _score(
                    source.epi[PARENT],
                    source.epi[index],
                    sense[PARENT],
                    sense[index],
                    post_distance,
                )
                score_margin = score - threshold
                selected = _decision(score_margin, "candidate score gate")
                if selected:
                    if score.lo < 0 or score.hi > 1:
                        raise ValueError(
                            "selected ideal conductance leaves the unit interval"
                        )
                    new_edges.append((PARENT, index))
                    new_weights.append(score)
        candidates.append(
            UmCandidatePrediction(
                index,
                pre_distance,
                pre_admitted,
                pre_margin,
                post_distance,
                post_margin,
                score,
                score_margin,
                selected,
            )
        )
    return UmResetPrediction(
        nodes=nodes,
        parent=PARENT,
        epi=source.epi,
        phase_before=source.phase,
        phase_after=tuple(after),
        capacity_before=source.capacity,
        capacity_after=tuple(capacity_after),
        raw_pressure_before=pressure,
        raw_pressure_after=tuple(pressure_after),
        sense_indices=sense,
        old_edges=tuple(
            (positions[left], positions[right]) for left, right in source.edges
        ),
        old_conductances=source.conductances,
        new_edges=tuple(new_edges),
        new_conductances=tuple(new_weights),
        compatible_indices=compatible,
        phase_changed_indices=changed,
        capacity_changed_indices=(PARENT,),
        consensus=consensus,
        consensus_real=real,
        consensus_imaginary=imaginary,
        phase_factor=phase_factor,
        capacity_factor=capacity_factor,
        pressure_factor=pressure_factor,
        phase_limit=limit,
        score_threshold=threshold,
        existing_pre_margins=pre_margins,
        existing_post_margins=post_margins,
        candidates=tuple(candidates),
        parent_alignment=alignment,
    )
