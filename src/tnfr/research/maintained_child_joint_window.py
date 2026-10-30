"""Retrospective joint child admission on already retained SHA time cells.

This observer selects the first whole certified cell meeting the unchanged
form, positive-pressure and absolute-acceleration criteria. It evaluates no
trajectory, acquires no samples and supplies no executed grammar context.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import I
from ..operators._thol_config import (
    resolve_thol_bifurcation_threshold,
    resolve_thol_min_epi,
)
from ..operators.grammar_dynamics import validate_candidate
from .maintained_child_history import CHILD, CHILD_INDEX, PARENT
from .maintained_parent_thol import _signature
from .maintained_sha_retention_evidence import load_sha_retention
from .maintained_temporal_window import bound_node_history_box


@dataclass(frozen=True)
class JointConditions:
    """Separate interval decisions; unresolved bounds are not exclusions."""

    form: str
    pressure: str
    acceleration: str
    identity: str
    joint: bool


@dataclass(frozen=True)
class ChildJointCell:
    """Rebuilt ranges on one original whole-time enclosure."""

    index: int
    elapsed_start: Q
    elapsed_end: Q
    form: I
    pressure: I
    acceleration: I
    minimum_principal_margin: Q
    conditions: JointConditions


@dataclass(frozen=True)
class ChildJointWindow:
    """A new observation of retained evidence, not a prospective prediction."""

    status: str
    child: str
    parent: int
    source_time: Q
    minimum_form: Q
    threshold: Q
    retained_cells: int
    inspected_cells: tuple[ChildJointCell, ...]
    selected_cell: ChildJointCell | None
    source_grammar_error_rules: tuple[str, ...]
    source_receipts: tuple
    retrospective_observation: bool = True
    acquired_history: bool = False
    acquired_u4b_context: bool = False
    executed_event: bool = False
    full_thol_admission_established: bool = False


def classify_joint_conditions(
    *, form, pressure, acceleration, minimum_form, threshold, principal_margins
):
    """Classify admitted ranges without replacing joint whole-time evidence.

    The caller must associate every range with the same time cell and complete
    law. Form equality passes; pressure and absolute acceleration are strict.
    A nonpositive sufficient principal margin abstains, never proves an exit.
    """
    form, pressure, acceleration = map(I.coerce, (form, pressure, acceleration))
    minimum_form = exact_or_represented_real(minimum_form, "minimum form")
    threshold = exact_or_represented_real(threshold, "acceleration threshold")
    if minimum_form < 0 or threshold < 0:
        raise ValueError("joint admission policies must be nonnegative")
    margins = tuple(
        exact_or_represented_real(value, "principal margin")
        for value in principal_margins
    )
    if not margins:
        raise ValueError("joint admission requires original principal margins")
    form_status = (
        "admitted"
        if form.lo >= minimum_form
        else "excluded" if form.hi < minimum_form else "unresolved"
    )
    pressure_status = (
        "admitted"
        if pressure.lo > 0
        else "excluded" if pressure.hi <= 0 else "unresolved"
    )
    acceleration_status = (
        "admitted"
        if acceleration.lo > threshold or acceleration.hi < -threshold
        else "excluded" if acceleration.abs_max <= threshold else "unresolved"
    )
    identity_status = "admitted" if min(margins) > 0 else "unresolved"
    values = form_status, pressure_status, acceleration_status, identity_status
    return JointConditions(*values, joint=all(value == "admitted" for value in values))


def inspect_child_joint_window(evidence_dir, *, repository_root):
    """Re-admit the complete retained certificate and inspect one fixed prefix.

    Both signs of acceleration are allowed by the canonical absolute threshold.
    A first certified cell need not contain the first actual admissible time:
    preceding rectangular enclosures can be inconclusive. The selection stops
    without refining cells, altering thresholds or extending the fixed horizon.
    """
    retained = load_sha_retention(evidence_dir, repository_root=repository_root)
    admitted, prediction = retained.admitted, retained.prediction
    graph = admitted.graph
    signature = _signature(graph)
    if (
        prediction.nodes[CHILD_INDEX] != CHILD
        or graph.nodes[CHILD]["parent_node"] != PARENT
        or tuple(graph[CHILD]) != (PARENT,)
        or retained.summary["status"] != "completed"
        or len(retained.steps) != 256
    ):
        raise ValueError(
            "joint child observation requires the complete actual certificate"
        )
    minimum_form = exact_or_represented_real(
        resolve_thol_min_epi(graph.graph), "THOL minimum EPI"
    )
    threshold = exact_or_represented_real(
        resolve_thol_bifurcation_threshold(graph.graph), "THOL threshold"
    )
    source_time = exact_or_represented_real(graph.graph["_t"], "source structural time")
    grammar = validate_candidate(graph, CHILD, "THOL")
    errors = tuple(
        value.rule for value in grammar.violations if value.severity == "error"
    )
    inspected, selected = [], None
    for step in retained.steps:
        bounds = bound_node_history_box(
            epi=step.tube[:21],
            phases=step.tube[21:],
            capacities=prediction.capacities,
            edges=prediction.edges,
            conductances=prediction.conductances,
            shifts=prediction.shifts,
            target_index=CHILD_INDEX,
        )
        # Both margins refer to the original branches under the freshly
        # re-admitted source/law. Neither is a caller-supplied verdict.
        margins = tuple(
            max(direct, energy.lo)
            for direct, energy in zip(
                bounds.principal_margins,
                prediction.bounds.horizon_remaining_margins,
                strict=True,
            )
        )
        conditions = classify_joint_conditions(
            form=bounds.form,
            pressure=bounds.pressure,
            acceleration=bounds.form_acceleration,
            minimum_form=minimum_form,
            threshold=threshold,
            principal_margins=margins,
        )
        cell = ChildJointCell(
            step.index,
            step.time,
            step.time + step.duration,
            bounds.form,
            bounds.pressure,
            bounds.form_acceleration,
            min(margins),
            conditions,
        )
        inspected.append(cell)
        if conditions.joint:
            selected = cell
            break
    if _signature(graph) != signature:
        raise ValueError("joint observation changed persistent source state")
    return ChildJointWindow(
        status="joint_window_admitted" if selected else "no_joint_cell_certified",
        child=CHILD,
        parent=PARENT,
        source_time=source_time,
        minimum_form=minimum_form,
        threshold=threshold,
        retained_cells=len(retained.steps),
        inspected_cells=tuple(inspected),
        selected_cell=selected,
        source_grammar_error_rules=errors,
        source_receipts=retained.receipts,
    )
