"""Conditional child OZ/refresh and observation timing on the retained cell.

This read-only admission derives a future execution contract. It executes no
event, refresh or trajectory, materializes no state from an interval, and
installs no history. Future numerical acquisition must independently achieve
the stated error and live-clock associations.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..dynamics.propagation import resolve_dissonance_propagation_policy
from ..mathematics._rational_interval import I
from ..operators._oz_config import OZStrictPolicy, resolve_oz_strict_policy
from ..operators.dissonance import Dissonance, _resolve_dissonance_propagation
from ..operators.factor_contracts import (
    _resolve_dissonance_noise_mode,
    resolve_runtime_operator_factors,
)
from ..operators.grammar_debt import node_debt
from ..operators.grammar_dynamics import validate_candidate
from ..operators.grammar_execution import ValidatedSequence
from ..operators.self_organization import SelfOrganization
from ..operators.silence import Silence
from ..physics._sample_reader_budget import (
    Binary64AccelerationReaderBudget,
    bound_binary64_acceleration_reader,
)
from ..physics.relational_observations import (
    RelationalSampleJetBudget,
    bound_relational_sample_jet_budget,
)
from ..types import Glyph
from .maintained_child_history import CHILD, CHILD_INDEX, PARENT
from .maintained_child_joint_source import CELL_END, CELL_START, load_child_joint_window
from .maintained_parent_thol import _represented, _signature

SAMPLE_STEP = Q(1, 1024)
SAMPLE_ERROR_ALLOWANCE = Q(1, 2**24)
PRESSURE_ERROR_ALLOWANCE = Q(1, 2**24)
_UNIT_ROUNDOFF = Q(1, 2**53)


@dataclass(frozen=True)
class ChildOZTimingAdmission:
    """Conditional analytical margins, not executed context or measured history.

    The materialization allowances are obligations on a later acquisition.
    The zero storage jump and continuation equivalence require the canonical
    pressure-only OZ map and the declared full constitutive refresh, without
    an intervening controller that consumes changed grammar/telemetry state.
    """

    child: str
    parent: int
    source_time: Q
    elapsed_sample_times: tuple[Q, Q, Q]
    strict_policy: OZStrictPolicy
    represented_form_bounds: I
    incoming_pressure_bounds: I
    gain: Q
    raw_pressure_bounds: I
    local_increment_bounds: I
    propagation_increment_upper_bound: Q
    propagation_cutoff: Q
    propagation_mode: str
    propagation_enabled: bool
    propagation_cutoff_excludes_parent: bool
    selected_word: tuple[str, ...]
    source_oz_grammar_allowed: bool
    source_thol_error_rules: tuple[str, ...]
    source_form_admits_strict_oz: bool
    future_oz_scalar_admission: bool
    pressure_error_allowance: Q
    sample_budget: RelationalSampleJetBudget
    reader_budget: Binary64AccelerationReaderBudget
    threshold: Q
    minimum_principal_margin: Q
    conditional_storage_jump: Q
    source_receipts: tuple[dict, ...]
    executed_event: bool = False
    acquired_history: bool = False
    acquired_u4b_context: bool = False
    full_thol_admission_established: bool = False


def _grammar_errors(result):
    return tuple(item.rule for item in result.violations if item.severity == "error")


def admit_child_oz_timing(evidence_dir, *, repository_root):
    """Rebuild the fixed actual cell and admit one prospective composition.

    The input is a source path, never a cached favorable report. No prefix
    selection or numerical integration is repeated. OZ policies and grammar
    come from the execution owners; the strict validator itself is not called
    because its successful probe writes telemetry. The degree policy is only
    advisory; this particular degree-one child also satisfies that advisory.
    """
    source = load_child_joint_window(evidence_dir, repository_root=repository_root)
    return _admit_child_oz_timing_source(source)


def _admit_child_oz_timing_source(source):
    """Internal composition on the reader's fresh source in one admission.

    This entry point does not authenticate a cached report. Public consumers
    reload the complete reader before calling it; acquisition can reuse that
    same fresh admission without loading the source chain a second time.
    """
    retained, bounds = source.retention, source.bounds
    graph, prediction = retained.admitted.graph, retained.prediction
    signature = _signature(graph)
    data = graph.nodes[CHILD]
    if (
        graph.graph.get("GRAMMAR_REJECTION_MODE") != "raise"
        or graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS") is not True
        or tuple(data.get("glyph_history", ()))
        or node_debt(data) != 0
        or _resolve_dissonance_noise_mode(graph.graph)
        or not _resolve_dissonance_propagation(graph, {})
    ):
        raise ValueError(
            "child timing requires the actual strict, empty-context source"
        )
    policy = resolve_oz_strict_policy(graph.graph)
    propagation = resolve_dissonance_propagation_policy(graph.graph)
    gain = _represented(
        resolve_runtime_operator_factors(
            graph.graph.get("GLYPH_FACTORS"), Glyph.OZ, graph.graph
        )["OZ_dnfr_factor"],
        "OZ gain",
    )
    form = bounds.form + I(-SAMPLE_ERROR_ALLOWANCE, SAMPLE_ERROR_ALLOWANCE)
    pressure = bounds.pressure + I(-PRESSURE_ERROR_ALLOWANCE, PRESSURE_ERROR_ALLOWANCE)
    # A narrow normal, positive branch makes the binary64 multiplication
    # estimate explicit, without extending it to underflow or near-zero seeding.
    if not (Q(1, 100) < pressure.lo <= pressure.hi < Q(1, 10) and 1 < gain < 2):
        raise ValueError("OZ timing is outside the admitted positive normal branch")
    ideal_raw = pressure * gain
    roundoff = _UNIT_ROUNDOFF * ideal_raw.abs_max
    raw = ideal_raw + I(-roundoff, roundoff)
    if raw.lo <= pressure.hi or raw.hi > 2 * pressure.lo:
        raise ValueError("local OZ increment lacks a positive Sterbenz domain")
    # q' - q is exact by Sterbenz, retaining the same incoming q in the gain.
    increment = pressure * (gain - 1) + I(-roundoff, roundoff)
    if increment.lo <= Q(1, 1000):
        raise ValueError("local OZ increment is not separated from nonnormal values")
    edge_index = prediction.edges.index((PARENT, CHILD_INDEX))
    weight = prediction.conductances[edge_index]
    if not Q(1, 2) < weight < 1:
        raise ValueError("actual child conductance is outside the admitted range")
    # Runtime phase_weight lies in [0,1]. Underflow of an arbitrarily small
    # product only needs an absolute subnormal rounding allowance. The final
    # default frequency factor is one. This bound includes both multiplications.
    tiny = Q(1, 2**1075)
    upper = (increment.hi * weight * (1 + _UNIT_ROUNDOFF) + tiny) * (
        1 + _UNIT_ROUNDOFF
    ) + tiny
    cutoff = _represented(propagation.min_propagation, "propagation cutoff")
    scalar_admission = (
        form.lo >= _represented(policy.min_epi, "OZ minimum form")
        and pressure.abs_max <= _represented(policy.max_dnfr, "OZ maximum pressure")
        and prediction.capacities[CHILD_INDEX]
        >= _represented(policy.min_vf, "OZ minimum capacity")
        and len(graph[CHILD]) >= policy.min_degree
    )
    word = ValidatedSequence(
        (Dissonance(), SelfOrganization(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    oz_grammar = validate_candidate(graph, CHILD, "OZ", sequence_context=word.step(0))
    thol_grammar = validate_candidate(
        graph, CHILD, "THOL", sequence_context=word.step(1)
    )
    source_time = _represented(graph.graph["_t"], "source structural time")
    elapsed_times = (CELL_START, CELL_START + SAMPLE_STEP, CELL_END)
    if elapsed_times[2] - elapsed_times[1] != SAMPLE_STEP:
        raise ValueError("child sample clock differs from the fixed cell")
    # This global jerk bound is retained by the general sample-jet owner.
    # Its point-derivative truncation term is not consumed by the whole-window
    # acceleration theorem, which needs only the observation-value error.
    motion = prediction.bounds
    jerk = motion.mobility[CHILD_INDEX] * (
        motion.edge_speed_bounds[edge_index] ** 2
        + motion.edge_acceleration_magnitudes[edge_index]
    )
    sample_budget = bound_relational_sample_jet_budget(
        sample_step=SAMPLE_STEP,
        sample_error_bounds=(SAMPLE_ERROR_ALLOWANCE,) * 3,
        third_derivative_bound=jerk.hi,
        observation_time=source_time + CELL_START,
    )
    reader = bound_binary64_acceleration_reader(
        form_bounds=bounds.form,
        rate_bounds=bounds.form_rate,
        acceleration_bounds=bounds.form_acceleration,
        sample_step=SAMPLE_STEP,
        observation_time=source_time + CELL_START,
        sample_error_bounds=(SAMPLE_ERROR_ALLOWANCE,) * 3,
    )
    if (
        not scalar_admission
        or not oz_grammar.allowed
        or "U4b" not in _grammar_errors(thol_grammar)
        or upper >= cutoff
        or not reader.exact
        or reader.represented_acceleration_bounds.lo <= source.threshold
        or reader.acceleration_sample_error_bound
        != sample_budget.acceleration_sample_error_bound
    ):
        raise ValueError("the fixed child OZ/timing composition was not admitted")
    if _signature(graph) != signature:
        raise ValueError("read-only child OZ/timing admission changed persistent state")
    return ChildOZTimingAdmission(
        child=CHILD,
        parent=PARENT,
        source_time=source_time,
        elapsed_sample_times=elapsed_times,
        strict_policy=policy,
        represented_form_bounds=form,
        incoming_pressure_bounds=pressure,
        gain=gain,
        raw_pressure_bounds=raw,
        local_increment_bounds=increment,
        propagation_increment_upper_bound=upper,
        propagation_cutoff=cutoff,
        propagation_mode="phase_weighted",
        propagation_enabled=True,
        propagation_cutoff_excludes_parent=True,
        selected_word=word.names,
        source_oz_grammar_allowed=oz_grammar.allowed,
        source_thol_error_rules=_grammar_errors(thol_grammar),
        source_form_admits_strict_oz=(
            prediction.epi[CHILD_INDEX]
            >= _represented(policy.min_epi, "OZ minimum form")
        ),
        future_oz_scalar_admission=scalar_admission,
        pressure_error_allowance=PRESSURE_ERROR_ALLOWANCE,
        sample_budget=sample_budget,
        reader_budget=reader,
        threshold=source.threshold,
        minimum_principal_margin=source.cell.minimum_principal_margin,
        conditional_storage_jump=Q(0),
        source_receipts=source.receipts,
    )
