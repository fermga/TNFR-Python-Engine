"""Fresh isolated node materialization shared by declared birth mechanisms.

Numerical inheritance, occurrence, hierarchy and later coupling belong to the
caller. The caller also owns rollback across this commit boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._exact_time import bounded_represented_real, finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ._epi_domain import require_real_scalar_epi
from .grammar_debt import read_grammar_state

_MISSING = object()


def require_fresh_member_identity(graph: Any, node: Any) -> None:
    """Retain NetworkX identity semantics, including bool/integer collisions."""
    if node is None:
        raise ValueError("a fresh member identity cannot be None")
    hash(node)
    if node in graph:
        raise ValueError("derived member identity already exists")


def materialize_isolated_member(
    graph: Any, node: Any, attributes: Mapping[str, Any]
) -> None:
    """Commit admitted fresh state without inheriting any support or event.

    Attribute values and insertion order are preserved. An initial own EPI
    sample is not elapsed temporal evidence. No numerical birth formula or
    universal pressure initialization is defined here.
    """
    require_fresh_member_identity(graph, node)
    if not isinstance(attributes, Mapping):
        raise TypeError("fresh member attributes must be a mapping")
    data = dict(attributes)

    def required(aliases: tuple[str, ...], label: str) -> Any:
        value = get_attr(data, aliases, _MISSING, strict=True, conv=lambda item: item)
        if value is _MISSING:
            raise ValueError(f"fresh member requires explicit {label}")
        return value

    epi = require_real_scalar_epi(
        required(ALIAS_EPI, "EPI"), operator="Birth", label="member EPI"
    )
    finite_represented_real(required(ALIAS_THETA, "phase"), "fresh member phase")
    bounded_represented_real(
        required(ALIAS_VF, "capacity"), "fresh member capacity", minimum=0
    )
    finite_represented_real(required(ALIAS_DNFR, "pressure"), "fresh member pressure")
    grammar = read_grammar_state(data)
    if (
        grammar.history
        or grammar.u2_debt
        or grammar.prior_coherence
        or grammar.operator_step
    ):
        raise ValueError("a fresh member cannot inherit operator context")
    if type(data.get("glyph_history")) is not list or data["glyph_history"]:
        raise ValueError("a fresh member requires its own empty glyph history")
    history = data.get("epi_history")
    if type(history) is not list or len(history) != 1:
        raise ValueError("a fresh member requires one own initial EPI sample")
    if (
        require_real_scalar_epi(history[0], operator="Birth", label="initial EPI")
        != epi
    ):
        raise ValueError("fresh member initial EPI sample must match its state")
    if "epi_time_history" in data or "_epi_history" in data:
        raise ValueError("a fresh member cannot inherit temporal or legacy history")
    graph.add_node(node, **data)
    if graph.degree(node) != 0:
        raise RuntimeError("fresh member materialization must leave it isolated")
