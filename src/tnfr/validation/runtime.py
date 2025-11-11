"""Runtime validation helpers exposing canonical graph contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, MutableSequence
from typing import Any, cast

import numpy as np

from ..alias import (
    get_attr,
    get_theta_attr,
    multi_recompute_abs_max,
    set_attr,
    set_attr_generic,
    set_theta,
    set_theta_attr,
    set_vf,
)
from ..constants import DEFAULTS
from ..types import (
    NodeId,
    TNFRGraph,
    ZERO_BEPI_STORAGE,
    ensure_bepi,
    serialize_bepi,
    _is_scalar,
)
from ..utils import clamp, ensure_collection
from .base import ValidationOutcome, Validator
from .graph import run_validators

HistoryLog = MutableSequence[MutableMapping[str, object]]
"""Mutable history buffer storing clamp alerts for strict graphs."""

__all__ = (
    "GraphCanonicalValidator",
    "apply_canonical_clamps",
    "validate_canon",
)


def _max_bepi_magnitude(value: Any) -> float:
    element = ensure_bepi(value)
    mags = [
        (float(np.max(np.abs(element.f_continuous))) if element.f_continuous.size else 0.0),
        float(np.max(np.abs(element.a_discrete))) if element.a_discrete.size else 0.0,
    ]
    return float(max(mags)) if mags else 0.0


def _clamp_component(values: Any, lower: float, upper: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.size == 0:
        return array
    magnitudes = np.abs(array)
    result = array.copy()
    above = magnitudes > upper
    if np.any(above):
        result[above] = array[above] * (upper / magnitudes[above])
    if lower > 0.0:
        below = (magnitudes < lower) & (magnitudes > 0.0)
        if np.any(below):
            result[below] = array[below] * (lower / magnitudes[below])
    return result


def _clamp_bepi(value: Any, lower: float, upper: float) -> Any:
    element = ensure_bepi(value)
    clamped_cont = _clamp_component(element.f_continuous, lower, upper)
    clamped_disc = _clamp_component(element.a_discrete, lower, upper)
    return ensure_bepi(
        {"continuous": clamped_cont, "discrete": clamped_disc, "grid": element.x_grid}
    )


def _log_clamp(
    hist: HistoryLog,
    node: NodeId | None,
    attr: str,
    value: float,
    lo: float,
    hi: float,
) -> None:
    if value < lo or value > hi:
        hist.append({"node": node, "attr": attr, "value": float(value)})


def apply_canonical_clamps(
    nd: MutableMapping[str, Any],
    G: TNFRGraph | None = None,
    node: NodeId | None = None,
) -> None:
    """Clamp nodal EPI, νf and θ according to canonical bounds."""

    from ..dynamics.aliases import ALIAS_EPI, ALIAS_VF

    if G is not None:
        graph_dict = cast(MutableMapping[str, Any], G.graph)
        graph_data: Mapping[str, Any] = graph_dict
    else:
        graph_dict = None
        graph_data = DEFAULTS
    eps_min = float(graph_data.get("EPI_MIN", DEFAULTS["EPI_MIN"]))
    eps_max = float(graph_data.get("EPI_MAX", DEFAULTS["EPI_MAX"]))
    vf_min = float(graph_data.get("VF_MIN", DEFAULTS["VF_MIN"]))
    vf_max = float(graph_data.get("VF_MAX", DEFAULTS["VF_MAX"]))
    theta_wrap = bool(graph_data.get("THETA_WRAP", DEFAULTS["THETA_WRAP"]))

    raw_epi = get_attr(nd, ALIAS_EPI, ZERO_BEPI_STORAGE, conv=lambda obj: obj)
    was_scalar = _is_scalar(raw_epi)
    original_scalar_value = float(raw_epi) if was_scalar else None
    epi = ensure_bepi(raw_epi)
    vf = get_attr(nd, ALIAS_VF, 0.0)
    th_val = get_theta_attr(nd, 0.0)
    th = 0.0 if th_val is None else float(th_val)

    strict = bool(graph_data.get("VALIDATORS_STRICT", DEFAULTS.get("VALIDATORS_STRICT", False)))
    if strict and graph_dict is not None:
        history = cast(MutableMapping[str, Any], graph_dict.setdefault("history", {}))
        alerts = history.get("clamp_alerts")
        if alerts is None:
            hist = cast(HistoryLog, history.setdefault("clamp_alerts", []))
        elif isinstance(alerts, MutableSequence):
            hist = cast(HistoryLog, alerts)
        else:
            materialized = ensure_collection(alerts, max_materialize=None)
            hist = cast(HistoryLog, list(materialized))
            history["clamp_alerts"] = hist
        epi_mag = _max_bepi_magnitude(epi)
        _log_clamp(hist, node, "EPI", epi_mag, eps_min, eps_max)
        _log_clamp(hist, node, "VF", float(vf), vf_min, vf_max)

    clamped_epi = _clamp_bepi(epi, eps_min, eps_max)
    if was_scalar:
        clamped_value = float(clamp(original_scalar_value, eps_min, eps_max))
        set_attr_generic(nd, ALIAS_EPI, clamped_value, conv=lambda obj: obj)
    else:
        set_attr_generic(nd, ALIAS_EPI, serialize_bepi(clamped_epi), conv=lambda obj: obj)

    vf_val = float(clamp(vf, vf_min, vf_max))
    if G is not None and node is not None:
        set_vf(G, node, vf_val, update_max=False)
    else:
        set_attr(nd, ALIAS_VF, vf_val)

    if theta_wrap:
        new_th = (th + math.pi) % (2 * math.pi) - math.pi
        if G is not None and node is not None:
            set_theta(G, node, new_th)
        else:
            set_theta_attr(nd, new_th)


class GraphCanonicalValidator(Validator[TNFRGraph]):
    """Validator enforcing canonical runtime contracts on TNFR graphs."""

    recompute_frequency_maxima: bool
    enforce_graph_validators: bool

    def __init__(
        self,
        *,
        recompute_frequency_maxima: bool = True,
        enforce_graph_validators: bool = True,
    ) -> None:
        self.recompute_frequency_maxima = bool(recompute_frequency_maxima)
        self.enforce_graph_validators = bool(enforce_graph_validators)

    def _diff_after_clamp(
        self,
        before: Mapping[str, float],
        after: Mapping[str, float],
    ) -> Mapping[str, Mapping[str, float]]:
        changes: dict[str, Mapping[str, float]] = {}
        for key, before_val in before.items():
            after_val = after.get(key)
            if after_val is None:
                continue
            if math.isclose(before_val, after_val, rel_tol=0.0, abs_tol=1e-12):
                continue
            changes[key] = {"before": before_val, "after": after_val}
        return changes

    def validate(self, subject: TNFRGraph, /, **_: Any) -> ValidationOutcome[TNFRGraph]:
        """Clamp nodal attributes, refresh νf maxima and run graph validators."""

        from ..dynamics.aliases import ALIAS_EPI, ALIAS_VF

        clamped_nodes: list[dict[str, Any]] = []
        for node, data in subject.nodes(data=True):
            mapping = cast(MutableMapping[str, Any], data)
            before = {
                "EPI": _max_bepi_magnitude(
                    get_attr(mapping, ALIAS_EPI, ZERO_BEPI_STORAGE, conv=lambda obj: obj)
                ),
                "VF": float(get_attr(mapping, ALIAS_VF, 0.0)),
                "THETA": float(get_theta_attr(mapping, 0.0) or 0.0),
            }
            apply_canonical_clamps(mapping, subject, cast(NodeId, node))
            after = {
                "EPI": _max_bepi_magnitude(
                    get_attr(mapping, ALIAS_EPI, ZERO_BEPI_STORAGE, conv=lambda obj: obj)
                ),
                "VF": float(get_attr(mapping, ALIAS_VF, 0.0)),
                "THETA": float(get_theta_attr(mapping, 0.0) or 0.0),
            }
            changes = self._diff_after_clamp(before, after)
            if changes:
                clamped_nodes.append({"node": node, "attributes": changes})

        maxima: Mapping[str, float] = {}
        if self.recompute_frequency_maxima:
            maxima = multi_recompute_abs_max(subject, {"_vfmax": ALIAS_VF})
            subject.graph.update(maxima)

        errors: list[str] = []
        if self.enforce_graph_validators:
            try:
                run_validators(subject)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(str(exc))

        summary: dict[str, Any] = {
            "clamped": tuple(clamped_nodes),
            "maxima": dict(maxima),
        }
        if errors:
            summary["errors"] = tuple(errors)

        passed = not errors
        artifacts: dict[str, Any] = {}
        if clamped_nodes:
            artifacts["clamped_nodes"] = clamped_nodes
        if maxima:
            artifacts["maxima"] = dict(maxima)

        return ValidationOutcome(
            subject=subject,
            passed=passed,
            summary=summary,
            artifacts=artifacts or None,
        )

    def report(self, outcome: ValidationOutcome[TNFRGraph]) -> str:
        """Return a concise textual summary of ``outcome``."""

        if outcome.passed:
            clamped = outcome.summary.get("clamped")
            if clamped:
                return "Graph canonical validation applied clamps successfully."
            return "Graph canonical validation passed without adjustments."

        errors = outcome.summary.get("errors")
        if not errors:
            return "Graph canonical validation failed without diagnostic details."
        if isinstance(errors, (list, tuple)):
            return "Graph canonical validation errors: " + ", ".join(map(str, errors))
        return f"Graph canonical validation error: {errors}"


def validate_canon(G: TNFRGraph) -> ValidationOutcome[TNFRGraph]:
    """Validate ``G`` using :class:`GraphCanonicalValidator`."""

    validator = GraphCanonicalValidator()
    return validator.validate(G)
