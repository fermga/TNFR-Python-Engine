"""Typed interior-law and endpoint observations of an actually completed AL episode."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from fractions import Fraction
from typing import Any

from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from .emission_episode import EmissionEpisodeResult, execute_emission_episode
from .operator_growth import observe_operator_epi_growth

__all__ = ("EmissionEpisodeObservation", "observe_emission_episode")

_AUTHORITY = object()
SOLVER_DIAGNOSTIC = "solver_diagnostic"
EMISSION_INTERIOR = "emission_interior"


@dataclass(frozen=True, slots=True)
class EmissionEpisodeObservation:
    """A passive typed report; interior zero is derived from the chosen held law.

    The episode's two endpoints do not supply sampled curvature or either
    outside boundary rate. Older graph history has its own observation owner.
    Solver diagnostics are captured at completion, before subsequent operators.
    """

    nodes: tuple[Any, ...]
    targets: tuple[Any, ...]
    exact_start_time: Fraction
    exact_end_time: Fraction
    exact_duration: Fraction
    exact_observed_rates: tuple[Fraction, ...]
    exact_held_rates: tuple[Fraction, ...]
    exact_interior_accelerations: tuple[Fraction, ...]
    solver_rates: tuple[float, ...]
    solver_accelerations: tuple[float, ...]
    source: str = field(default="emission_episode_interior", init=False)
    acceleration_basis: str = field(default="held_emission_law", init=False)
    episode_sampled_acceleration_available: bool = field(default=False, init=False)
    boundary_transition_available: bool = field(default=False, init=False)


def observe_emission_episode(
    result: EmissionEpisodeResult,
) -> EmissionEpisodeObservation:
    """Read intact episode evidence without supplying any live action authority."""
    if (
        type(result) is not EmissionEpisodeResult
        or not result._proof_fields_are_intact()
    ):
        raise ValueError(
            "Emission observation requires an intact actual episode result"
        )
    flow = observe_operator_epi_growth(result.flow_result)
    count = len(result.nodes)
    if (
        not flow.available
        or result.flow_result.all_positive_flow_intervals_binary64_held_pressure_identified
        is not True
        or flow.exact_epi_before != result.exact_epi_before
        or flow.exact_epi_after != result.exact_epi_after
        or any(
            len(value) != count
            for value in (
                result.exact_rates,
                result.solver_rates,
                result.solver_accelerations,
            )
        )
    ):
        raise ValueError("Emission observation source coverage is unavailable")
    return EmissionEpisodeObservation(
        result.nodes,
        result.targets,
        flow.exact_start_time,
        flow.exact_end_time,
        flow.exact_duration,
        flow.exact_net_growth_rate,
        result.exact_rates,
        (Fraction(0),) * count,
        result.solver_rates,
        result.solver_accelerations,
    )


def _observation_values(observation):
    return tuple(
        getattr(observation, item.name) for item in fields(EmissionEpisodeObservation)
    )


def _selector_acceleration_source(value):
    if type(value) is not str or value not in (SOLVER_DIAGNOSTIC, EMISSION_INTERIOR):
        raise ValueError(
            "selector_acceleration_source must be 'solver_diagnostic' or 'emission_interior'"
        )
    return value


def _require_episode_parametric_selector(graph, selector=None):
    from ..dynamics import selectors

    original, bindings = next(
        (kind, bindings)
        for kind, bindings in selectors._NATIVE_SOURCE_SELECTOR_BINDINGS
        if kind.__name__ == "ParametricGlyphSelector"
    )
    if selectors.ParametricGlyphSelector is not original or any(
        getattr(original, name) is not expected for name, expected in bindings
    ):
        raise TypeError(
            "emission_interior requires the unchanged exact ParametricGlyphSelector"
        )
    configured = graph.graph.get("glyph_selector")
    if configured is not selectors.ParametricGlyphSelector and (
        type(configured) is not selectors.ParametricGlyphSelector
        or selectors._native_selector_kind(configured) != "parametric"
    ):
        raise TypeError(
            "emission_interior requires the unchanged exact ParametricGlyphSelector"
        )
    if selector is not None and (
        type(selector) is not selectors.ParametricGlyphSelector
        or selectors._native_selector_kind(selector) != "parametric"
    ):
        raise TypeError(
            "emission_interior requires the unchanged exact ParametricGlyphSelector"
        )


def _selection_state(graph, references):
    from ._operator_growth_trigger import _materialize_graph_views
    from .event_runtime import (
        _canonical_graph_storage_references,
        _protected_graph_identity_signature,
    )
    from .network_stage import _networkx_runtime_layout

    _materialize_graph_views(graph)
    layout = _networkx_runtime_layout(graph)
    return _protected_graph_identity_signature(
        graph,
        excluded_node_keys=frozenset(),
        excluded_graph_keys=frozenset(
            {"glyph_selector", "_selector_weights", "_sel_norms"}
        ),
        retained_references=references,
        opaque_references=_canonical_graph_storage_references(graph, layout),
        layout=layout,
    )


class _EmissionSelectionContext:
    """Call-local, one-preparation binding; never stored in graph or result."""

    def __init__(self, graph, result, *, _authority):
        if _authority is not _AUTHORITY:
            raise TypeError(
                "Emission selection requires actual same-call episode production"
            )
        self._graph = graph
        self._result = result
        self._observation = observe_emission_episode(result)
        self._observation_stamp = structural_proof_signature(
            _observation_values(self._observation)
        )
        self._references = [result, result.flow_result]
        self._signature = _selection_state(graph, self._references)
        self._active = True
        self._used = False
        self._finished = False

    @property
    def result(self):
        return self._result

    @property
    def observation(self):
        return self._observation

    def validate_live(self, graph):
        if not self._active or graph is not self._graph:
            raise ValueError("Emission selection source owner is foreign or inactive")
        if (
            self.result is not self._references[0]
            or self.result.flow_result is not self._references[1]
        ):
            raise ValueError("Emission selection actual source changed")
        _require_episode_parametric_selector(graph)
        rebuilt = observe_emission_episode(self.result)
        if any(
            not proof_stamps_are_identical(
                self._observation_stamp,
                structural_proof_signature(_observation_values(value)),
            )
            for value in (rebuilt, self.observation)
        ):
            raise ValueError("Emission selection observation changed")
        if not proof_stamps_are_identical(self._signature, _selection_state(graph, [])):
            raise ValueError("Emission selection live endpoint is stale")
        return self.observation

    def admit_preparation(self, graph, nodes):
        observation = self.validate_live(graph)
        if self._used:
            raise ValueError("Emission selection source already consumed a preparation")
        actual = tuple(nodes)
        if len(actual) != len(observation.nodes) or any(
            left is not right
            for left, right in zip(actual, observation.nodes, strict=True)
        ):
            raise ValueError("Emission selection requires the full ordered population")
        self._used = True
        return observation

    def finish_preparation(self, graph):
        if not self._used or self._finished:
            raise ValueError("Emission selection has no unfinished preparation")
        self.validate_live(graph)
        self._finished = True

    def close(self):
        self._active = False


@contextmanager
def _execute_emission_for_selection(
    graph,
    targets,
    *,
    duration,
    method=None,
    n_jobs=None,
    window=None,
    _history=None,
):
    """Produce the actual AL episode, then lend its interior channel once."""
    from .network_stage import GraphTransactionSnapshot

    _require_episode_parametric_selector(graph)
    transaction = GraphTransactionSnapshot(graph)
    try:
        if _history is not None:
            if type(_history) is not dict:
                raise TypeError(
                    "typed native source requires graph-owned exact history"
                )
            if "history" not in graph.graph:
                graph.graph["history"] = _history
            elif graph.graph["history"] is not _history:
                raise ValueError("typed native source history is foreign")
        result = execute_emission_episode(
            graph,
            targets,
            duration=duration,
            method=method,
            n_jobs=n_jobs,
            window=window,
        )
    except BaseException as failure:
        transaction.restore_after_failure(graph, failure)
        raise
    # The actual AL source is complete. Later context/selection failures retain it.
    context = _EmissionSelectionContext(graph, result, _authority=_AUTHORITY)
    try:
        yield context
    finally:
        context.close()
