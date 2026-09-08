"""Immutable pointwise stage proposals for Emission and Silence.

Emission (AL) and Silence (SHA) have target-local primary kernels, but their
public operators also maintain temporal lifecycle metadata.  This module
separates those two concerns without changing the public execution path:

``snapshot -> immutable proposal -> lifecycle commit -> structural commit``.

The caller supplies timestamps so proposal construction remains deterministic
and does not read the wall clock.  Proposal construction is read-only.  Commit
helpers deliberately retain the current metadata patch rules and use canonical
``NodeNX`` setters for structural writes, including the existing frequency
cache update performed by SHA.

These helpers do not select grammar fallbacks, run monitors, append histories,
collect metrics, refresh pressure, or declare a network schedule.  A network
executor must keep those operations in its outer transactional boundary and
must use the established Gauss--Seidel path whenever AL or SHA is replaced by
another glyph.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from ..alias import get_attr, set_attr_str
from ..constants.aliases import ALIAS_EMISSION_TIMESTAMP, ALIAS_EPI, ALIAS_VF
from ..constants.canonical import COUPLING_GENTLE, SHA_VF_FACTOR
from ..dynamics.feedback import StructuralFeedbackLoop
from ..errors import TNFRValueError
from ..types import Glyph, TNFRGraph
from ._epi_domain import require_real_scalar_epi

__all__ = [
    "EmissionStageProposal",
    "SilenceStageProposal",
    "commit_emission_lifecycle",
    "commit_emission_structure",
    "commit_silence_lifecycle",
    "commit_silence_structure",
    "propose_emission_stage",
    "propose_silence_stage",
]


_LATENCY_KEYS = (
    "latent",
    "latency_start_time",
    "preserved_epi",
    "silence_duration",
    "was_initial_on_silence",
)


@dataclass(frozen=True, slots=True)
class EmissionStageProposal:
    """Frozen AL structural value and exact lifecycle patch intent."""

    node: Any
    epi_before: float
    epi_after: float
    latency_keys_to_clear: tuple[str, ...]
    warning_messages: tuple[str, ...]
    initialize_emission: bool
    emission_timestamp: str | None
    increment_lineage: bool
    lineage_activation_count_after: int | None
    glyph: Glyph = field(default=Glyph.AL, init=False)


@dataclass(frozen=True, slots=True)
class SilenceStageProposal:
    """Frozen SHA structural value and exact latency metadata values."""

    node: Any
    vf_before: float
    vf_after: float
    latency_start_time: str
    preserved_epi: float
    was_initial_on_silence: bool
    glyph: Glyph = field(default=Glyph.SHA, init=False)


@dataclass(frozen=True, slots=True)
class _GraphBoundsView:
    """Minimal read-only shape consumed by the canonical EPI boundary helper."""

    graph: Mapping[str, Any]


def _require_timestamp(timestamp: Any, *, operator: str) -> str:
    if not isinstance(timestamp, str) or not timestamp:
        raise TNFRValueError(
            f"{operator} stage timestamp must be a non-empty string",
            context={"operator": operator, "timestamp": repr(timestamp)},
        )
    return timestamp


def _raw_alias(
    graph: TNFRGraph,
    node: Any,
    aliases: tuple[str, ...],
    default: Any,
) -> Any:
    return get_attr(
        graph.nodes[node],
        aliases,
        default,
        strict=True,
        conv=lambda value: value,
    )


def _emission_lifecycle_fields(
    graph: TNFRGraph,
    node: Any,
    *,
    epi_before: float,
    timestamp: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    bool,
    str | None,
    bool,
    int | None,
]:
    """Read the AL lifecycle patch without modifying the source graph."""

    data = graph.nodes[node]
    clear_keys: tuple[str, ...] = ()
    messages: list[str] = []
    if bool(data.get("latent", False)):
        silence_duration = data.get("silence_duration", 0.0)
        max_silence = graph.graph.get("MAX_SILENCE_DURATION", float("inf"))
        if silence_duration > max_silence:
            messages.append(
                f"Node {node} reactivating after extended silence "
                f"(duration: {silence_duration:.2f}, "
                f"max: {max_silence:.2f})"
            )

        preserved_epi = data.get("preserved_epi")
        if preserved_epi is not None:
            epi_drift = abs(epi_before - preserved_epi)
            if abs(preserved_epi) < 1e-6:
                tolerance = StructuralFeedbackLoop.EPI_THRESHOLD
                should_warn = epi_drift > tolerance
            else:
                tolerance = 0.01 * abs(preserved_epi)
                should_warn = epi_drift > tolerance
            if should_warn:
                node_type = (
                    "initial" if abs(preserved_epi) < 1e-6 else "established"
                )
                messages.append(
                    f"Node {node} ({node_type}) EPI drifted during silence "
                    f"(preserved: {preserved_epi:.3f}, "
                    f"current: {epi_before:.3f}, "
                    f"drift: {epi_drift:.3f}, tolerance: {tolerance:.3f})"
                )
        clear_keys = tuple(key for key in _LATENCY_KEYS if key in data)

    initialize = "_emission_activated" not in data
    increment_lineage = not initialize and "_structural_lineage" in data
    activation_count_after = None
    if increment_lineage:
        lineage = data["_structural_lineage"]
        if not isinstance(lineage, MutableMapping):
            raise TNFRValueError(
                "Emission structural lineage must be a mutable mapping"
            )
        activation_count = lineage.get("activation_count")
        if (
            isinstance(activation_count, bool)
            or not isinstance(activation_count, Integral)
            or activation_count < 0
        ):
            raise TNFRValueError(
                "Emission activation_count must be a nonnegative integer"
            )
        activation_count_after = int(activation_count) + 1
    return (
        clear_keys,
        tuple(messages),
        initialize,
        timestamp if initialize else None,
        increment_lineage,
        activation_count_after,
    )


def propose_emission_stage(
    graph: TNFRGraph,
    node: Any,
    factors: Mapping[str, Any],
    *,
    timestamp: str,
) -> EmissionStageProposal:
    """Build a read-only AL proposal from one stage-start graph.

    ``factors`` may be the mapping returned by
    :func:`resolve_runtime_operator_factors`; an absent ``AL_boost`` retains
    the same canonical fallback as the direct glyph kernel.
    """

    timestamp = _require_timestamp(timestamp, operator="Emission")
    from . import _finite_operator_scalar, _validated_epi_assignment_value, get_factor

    epi_before = require_real_scalar_epi(
        _raw_alias(graph, node, ALIAS_EPI, 0.0),
        operator="Emission",
        label="target EPI state",
    )
    boost = get_factor(dict(factors), "AL_boost", COUPLING_GENTLE)
    raw_proposal = _finite_operator_scalar(
        epi_before + boost, "AL EPI proposal"
    )
    epi_after = _validated_epi_assignment_value(
        _GraphBoundsView(graph.graph), raw_proposal
    )
    (
        clear_keys,
        messages,
        initialize,
        emission_timestamp,
        increment_lineage,
        activation_count_after,
    ) = _emission_lifecycle_fields(
        graph,
        node,
        epi_before=epi_before,
        timestamp=timestamp,
    )
    return EmissionStageProposal(
        node=node,
        epi_before=epi_before,
        epi_after=epi_after,
        latency_keys_to_clear=clear_keys,
        warning_messages=messages,
        initialize_emission=initialize,
        emission_timestamp=emission_timestamp,
        increment_lineage=increment_lineage,
        lineage_activation_count_after=activation_count_after,
    )


def propose_silence_stage(
    graph: TNFRGraph,
    node: Any,
    factors: Mapping[str, Any],
    *,
    timestamp: str,
) -> SilenceStageProposal:
    """Build a read-only SHA proposal from one stage-start graph."""

    timestamp = _require_timestamp(timestamp, operator="Silence")
    from . import _finite_operator_scalar, get_factor

    data = graph.nodes[node]
    vf_before = _finite_operator_scalar(
        get_attr(data, ALIAS_VF, 0.0), "SHA nu_f state"
    )
    factor = get_factor(dict(factors), "SHA_vf_factor", SHA_VF_FACTOR)
    vf_after = _finite_operator_scalar(
        factor * vf_before, "SHA nu_f proposal"
    )
    preserved_epi = float(get_attr(data, ALIAS_EPI, 0.0))
    return SilenceStageProposal(
        node=node,
        vf_before=vf_before,
        vf_after=vf_after,
        latency_start_time=timestamp,
        preserved_epi=preserved_epi,
        was_initial_on_silence=abs(preserved_epi) < 1e-6,
    )


def commit_emission_lifecycle(
    graph: TNFRGraph, proposal: EmissionStageProposal
) -> None:
    """Apply AL warnings and metadata in their current pre-glyph order."""

    for message in proposal.warning_messages:
        warnings.warn(message, stacklevel=3)

    data = graph.nodes[proposal.node]
    for key in proposal.latency_keys_to_clear:
        del data[key]

    if proposal.initialize_emission:
        timestamp = proposal.emission_timestamp
        if timestamp is None:
            raise RuntimeError("first Emission proposal lacks its timestamp")
        set_attr_str(data, ALIAS_EMISSION_TIMESTAMP, timestamp)
        data["_emission_activated"] = True
        data["_emission_origin"] = timestamp
        data["_structural_lineage"] = {
            "origin": timestamp,
            "activation_count": 1,
            "derived_nodes": [],
            "parent_emission": None,
        }
    elif proposal.increment_lineage:
        activation_count = proposal.lineage_activation_count_after
        if activation_count is None:
            raise RuntimeError("repeated Emission proposal lacks activation count")
        data["_structural_lineage"]["activation_count"] = activation_count


def commit_emission_structure(
    graph: TNFRGraph, proposal: EmissionStageProposal
) -> None:
    """Commit the already bounded AL EPI through the canonical setter."""

    from ..node import NodeNX
    from . import _set_epi_with_boundary_check

    node = NodeNX.from_graph(graph, proposal.node)
    _set_epi_with_boundary_check(node, proposal.epi_after, apply_clip=False)


def commit_silence_lifecycle(
    graph: TNFRGraph, proposal: SilenceStageProposal
) -> None:
    """Apply the exact five-field SHA latency metadata overwrite."""

    data = graph.nodes[proposal.node]
    data["latent"] = True
    data["latency_start_time"] = proposal.latency_start_time
    data["preserved_epi"] = proposal.preserved_epi
    data["silence_duration"] = 0.0
    data["was_initial_on_silence"] = proposal.was_initial_on_silence


def commit_silence_structure(
    graph: TNFRGraph, proposal: SilenceStageProposal
) -> None:
    """Commit the validated SHA capacity through the cache-aware setter."""

    from ..node import NodeNX

    NodeNX.from_graph(graph, proposal.node).vf = proposal.vf_after
