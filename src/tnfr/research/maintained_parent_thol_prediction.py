"""Detached prospective reset bounds for the retained maintained-parent gate.

This adapter does not call THOL, its proposal or metabolic implementation.
It bounds the declared default algebra from admitted primitive coordinates.
The numerical allowance is a frozen acceptance budget for the implementation,
not an assumed universal bound on libm or a trajectory certificate. Artifact
association and the parent's independent temporal certificate belong to the
consumer; equal scalar values do not authenticate that history.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from math import pi, tau

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR
from ..constants.canonical import COUPLING_GENTLE, COUPLING_MODERATE
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import I, cos, sin, sqrt
from ..operators._thol_constants import THOL_CHILD_VF_DAMPING, THOL_SUB_EPI_SCALING
from ..operators.nodal_equation import observe_structural_acceleration
from ..physics.relational_sine_support import bound_relational_sine_support_exchange
from .local_capacity_history_certificate import TIMES, admit_source

NUMERICAL_ALLOWANCE = Q(1, 2**44)
PARENT = 2
CHILD = "2_sub_0"
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")


@dataclass(frozen=True)
class MetabolicPrediction:
    """Exact means and independent ideal transcendental signal enclosures."""

    neighbor_count: int
    mean_neighbor_epi: Q
    epi_gradient: Q
    mean_cos: I
    mean_sin: I
    phase_variance: I
    coupling_strength_mean: I
    raw_child_form: I
    child_form: I


@dataclass(frozen=True)
class TholResetPrediction:
    """Conditional default event prediction, before any runtime proposal."""

    parent: int
    child: str
    source_nodes: tuple
    successor_nodes: tuple
    source_edges: tuple
    source_conductances: tuple[Q, ...]
    successor_edges: tuple
    expected_child_count: int
    child_degree: int
    parent_form: Q
    parent_phase: Q
    parent_capacity: Q
    parent_pressure: Q
    observed_acceleration: Q
    pressure_gain: Q
    exact_pressure_increment: Q
    raw_pressure_ideal: Q
    raw_pressure_bounds: I
    pressure_square_storage_jump: Q
    neighbors: tuple
    metabolic: MetabolicPrediction
    child_form_bounds: I
    child_capacity: Q
    child_phase: Q
    child_pressure: Q
    child_hierarchy_level: int
    child_hierarchy_path: tuple
    child_record_operator_step: int
    expected_parent_glyph_history: tuple
    expected_parent_u2_debt: int
    event_storage_jump: Q
    old_component_mean_jumps: tuple[Q, Q]
    global_positive_mean_metric_available: bool
    numerical_allowance: Q = NUMERICAL_ALLOWANCE


def _represented(value, label):
    exact = exact_or_represented_real(value, label)
    represented = finite_represented_real(value, label)[1]
    if exact != represented:
        raise ValueError(f"{label} must already be exactly represented in binary64")
    return represented


def _unit_clip(interval):
    """Image of an interval under the supplied nondecreasing unit clip."""
    return I(max(Q(0), min(Q(1), interval.lo)), max(Q(0), min(Q(1), interval.hi)))


def _with_allowance(interval):
    return I(interval.lo - NUMERICAL_ALLOWANCE, interval.hi + NUMERICAL_ALLOWANCE)


def _metabolic_prediction(parent_form, parent_phase, forms, phases):
    """Bound the default amplitude without reading cached metabolic signals.

    The selected domain has positive forms below one and parent-to-neighbor
    phase separations strictly below represented pi. Thus affinity uses the
    unwrapped separation with the runtime's represented pi coefficient. The
    transcendental variance uses mathematical sine, cosine and square root.
    """
    parent_form = _represented(parent_form, "parent form")
    parent_phase = _represented(parent_phase, "parent phase")
    forms = tuple(_represented(x, "neighbor form") for x in forms)
    phases = tuple(_represented(x, "neighbor phase") for x in phases)
    if not forms or len(forms) != len(phases):
        raise ValueError("metabolic inputs require equal nonempty neighbor rows")
    if not all(Q(0) <= value <= Q(1) for value in (parent_form, *forms)):
        raise ValueError("selected metabolic forms must lie in [0, 1]")
    differences = tuple(abs(value - parent_phase) for value in phases)
    if any(value >= Q(pi) for value in differences):
        raise ValueError("selected affinities require strict unwrapped pi separation")
    count = len(forms)
    mean = sum(forms, Q(0)) / count
    gradient = mean - parent_form
    mean_cos = sum((cos(I(value)) for value in phases), I(0)) / count
    mean_sin = sum((sin(I(value)) for value in phases), I(0)) / count
    variance = _unit_clip(1 - sqrt(mean_cos**2 + mean_sin**2))
    affinity = I(1 - sum(differences, Q(0)) / (count * Q(pi)))
    raw = (
        I(parent_form * Q(THOL_SUB_EPI_SCALING))
        + gradient * Q(COUPLING_MODERATE)
        + variance * Q(COUPLING_GENTLE)
    )
    return MetabolicPrediction(
        count,
        mean,
        gradient,
        mean_cos,
        mean_sin,
        variance,
        affinity,
        raw,
        _unit_clip(raw),
    )


def _admit_default_context(graph):
    data = graph.graph
    if any(str(key).startswith("THOL_") for key in data):
        raise ValueError("selected prediction requires absent THOL overrides")
    if "BIFURCATION_THRESHOLD_TAU" in data:
        raise ValueError("selected prediction requires the unchanged default threshold")
    if data.get("GLYPH_FACTORS") != {"VAL_scale": 128.0}:
        raise ValueError("selected prediction retains only the preceding VAL factor")
    if (
        data.get("GRAMMAR_REJECTION_MODE") != "raise"
        or data.get("VALIDATE_OPERATOR_PRECONDITIONS") is not True
        or data.get("EDGE_AWARE_ENABLED") is not False
    ):
        raise ValueError(
            "selected prediction requires unchanged strict execution policy"
        )
    if (
        _represented(data.get("EPI_MIN"), "EPI_MIN") != -1
        or _represented(data.get("EPI_MAX"), "EPI_MAX") != 1
    ):
        raise ValueError("selected child rail requires EPI bounds [-1, 1]")
    if _represented(data.get("_t"), "event clock") != TIMES[-1]:
        raise ValueError("selected prediction requires the retained endpoint clock")
    if data.get("hierarchy", {}) != {}:
        raise ValueError("selected prediction requires no preexisting hierarchy")
    for node, values in graph.nodes(data=True):
        for key in ("sub_nodes", "sub_epis", "_hierarchy_path"):
            if values.get(key, []) != []:
                raise ValueError(f"selected source must have no {key}")
        if "parent_node" in values:
            raise ValueError("selected source must have no parent references")
        for key in ("hierarchy_level", "_bifurcation_level"):
            value = values.get(key, 0)
            if type(value) is not int or value != 0:
                raise ValueError(f"selected source requires zero {key}")
    target = graph.nodes[PARENT]
    if (
        tuple(target.get("glyph_history", ())) != ("VAL",)
        or type(target.get("_grammar_u2_debt")) is not int
        or target["_grammar_u2_debt"] != 1
        or type(target.get("_operator_step")) is not int
        or target["_operator_step"] != 1
        or target.get("_grammar_prior_coherence") is not False
    ):
        raise ValueError("selected prediction requires the actual retained VAL prefix")


def predict_thol_reset(graph):
    """Re-admit the fixed parent and predict one default, isolated child.

    No graph is mutated and no operator/proposal/metabolism is executed. Zero
    edge-storage and old-component mean jumps follow from unchanged existing
    coordinates, capacities and edges, plus zero incident child edges; they
    are not inferred by subtracting overlapping storage intervals. The global
    positive inverse-capacity mean metric is unavailable at an isolate.
    """
    source = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    admit_source(source)
    _admit_default_context(graph)
    target = graph.nodes[PARENT]
    pressure = _represented(
        get_attr(target, ALIAS_DNFR, strict=True, conv=lambda value: value),
        "parent stored pressure",
    )
    if not Q(0) < pressure < Q(1):
        raise ValueError(
            "selected stored pressure must lie strictly between zero and one"
        )
    observation = observe_structural_acceleration(graph, PARENT)
    if (
        not observation.available
        or observation.source != "epi_time_history"
        or observation.history_length != 3
        or observation.current_endpoint_matches_state is not True
    ):
        raise ValueError("selected prediction requires the live three-sample history")
    samples = observation.samples
    if tuple(_represented(row[0], "sample time") for row in samples) != TIMES:
        raise ValueError("selected history must retain all three original sample times")
    acceleration = _represented(observation.value, "observed acceleration")
    if not Q(0.1) < acceleration < Q(1):
        raise ValueError(
            "selected parent requires positive acceleration above default tau"
        )
    index = source.nodes.index(PARENT)
    parent_form, parent_phase, parent_capacity = (
        source.epi[index],
        source.phase[index],
        source.capacity[index],
    )
    if not Q(0) <= parent_phase < Q(tau):
        raise ValueError("selected child phase requires the unchanged positive chart")
    neighbors = tuple(graph.neighbors(PARENT))
    positions = {node: i for i, node in enumerate(source.nodes)}
    metabolic = _metabolic_prediction(
        parent_form,
        parent_phase,
        tuple(source.epi[positions[node]] for node in neighbors),
        tuple(source.phase[positions[node]] for node in neighbors),
    )
    gain = Q(COUPLING_GENTLE)
    increment = gain * acceleration
    raw_pressure = pressure + increment
    child_capacity = parent_capacity * Q(THOL_CHILD_VF_DAMPING)
    if _represented(child_capacity, "child capacity") != child_capacity:
        raise ValueError("child capacity product is not exactly representable")
    return TholResetPrediction(
        parent=PARENT,
        child=CHILD,
        source_nodes=source.nodes,
        successor_nodes=(*source.nodes, CHILD),
        source_edges=source.edges,
        source_conductances=source.conductances,
        successor_edges=source.edges,
        expected_child_count=1,
        child_degree=0,
        parent_form=parent_form,
        parent_phase=parent_phase,
        parent_capacity=parent_capacity,
        parent_pressure=pressure,
        observed_acceleration=acceleration,
        pressure_gain=gain,
        exact_pressure_increment=increment,
        raw_pressure_ideal=raw_pressure,
        raw_pressure_bounds=_with_allowance(I(raw_pressure)),
        pressure_square_storage_jump=(raw_pressure**2 - pressure**2) / 2,
        neighbors=neighbors,
        metabolic=metabolic,
        child_form_bounds=_with_allowance(metabolic.child_form),
        child_capacity=child_capacity,
        child_phase=parent_phase,
        child_pressure=Q(0),
        child_hierarchy_level=1,
        child_hierarchy_path=(PARENT,),
        child_record_operator_step=2,
        expected_parent_glyph_history=("VAL", "THOL"),
        expected_parent_u2_debt=0,
        event_storage_jump=Q(0),
        old_component_mean_jumps=(Q(0), Q(0)),
        global_positive_mean_metric_available=False,
    )
