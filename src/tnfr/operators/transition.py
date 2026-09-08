"""Transition (NAV) operator.

Purpose: controlled regime handoff (latent/active/resonant).
Physics: adjusts theta, nu_f, and DeltaNFR for smooth state change.
Grammar: generator/closure compatible; sequence bridge.
Telemetry: stores origin regime and before/after values.
Typical: AL->NAV->IL, SHA->NAV->AL, NAV->ZHIR, IL->NAV->OZ.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

from ..alias import get_attr, set_attr
from ..config.operator_names import TRANSITION
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..types import Glyph, TNFRGraph
from ..utils import angle_diff
from ._argument_validation import (
    finite_node_real,
    finite_real,
    reject_operator_argument,
    require_list_sink,
    strict_bool,
    validate_common_execution_arguments,
)
from .definitions_base import Operator
from .factor_contracts import resolve_runtime_operator_factors

if TYPE_CHECKING:  # pragma: no cover - import used only by static analyzers
    from .jitter import JitterProposal

_VF_LATENT_THRESHOLD = 0.05
_EPI_RESONANT_THRESHOLD = 0.5
_VF_RESONANT_THRESHOLD = 0.8
_EPI_DRIFT_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class _LatencyProposal:
    """Validated latency metadata whose commit is deferred until NAV succeeds."""

    active: bool
    duration: float | None = None
    max_duration: float | None = None
    extended: bool = False
    preserved_epi: float | None = None
    current_epi: float | None = None
    epi_drift: float | None = None
    drifted: bool = False


@dataclass(frozen=True, slots=True)
class _TransitionPreflight:
    """Finite NAV request and output envelope prepared before glyph dispatch."""

    regime: str
    epi_before: float
    vf_before: float
    vf_after: float
    theta_before: float
    theta_after: float
    phase_shift_requested: float
    phase_shift_applied: float
    dnfr_before: float
    handler_base: float
    jitter_amplitude: float
    random_mode: bool
    retention: float
    handler_dnfr_after: float | None
    dnfr_after: float | None
    handler_dnfr_bounds: tuple[float, float]
    dnfr_bounds: tuple[float, float]
    latency: _LatencyProposal


@dataclass(frozen=True, slots=True)
class TransitionNetworkStageProposal:
    """One fully bound NAV target proposal from an immutable stage snapshot."""

    node: Any
    transition: _TransitionPreflight
    handler_dnfr_after: float
    dnfr_after: float
    jitter_proposal: JitterProposal | None = None
    glyph: Glyph = field(default=Glyph.NAV, init=False)


def _transition_event(
    node: Any,
    proposal: _TransitionPreflight,
    handler_dnfr: float,
    dnfr_after: float,
) -> dict[str, Any]:
    """Return the canonical NAV audit event from bound scalar values."""

    return {
        "node": node,
        "regime_origin": proposal.regime,
        "vf_before": proposal.vf_before,
        "vf_after": proposal.vf_after,
        "theta_before": proposal.theta_before,
        "theta_after": proposal.theta_after,
        "dnfr_before": proposal.dnfr_before,
        "dnfr_handler_after": handler_dnfr,
        "dnfr_after": dnfr_after,
        "phase_shift": proposal.phase_shift_applied,
        "phase_shift_requested": proposal.phase_shift_requested,
    }


def commit_transition_network_structure(
    G: TNFRGraph,
    proposal: TransitionNetworkStageProposal,
) -> None:
    """Commit one already validated NAV structural and RNG proposal."""

    data = G.nodes[proposal.node]
    if proposal.jitter_proposal is not None:
        from .jitter import commit_jitter_proposal

        commit_jitter_proposal(data, proposal.jitter_proposal)
    set_attr(data, ALIAS_VF, proposal.transition.vf_after)
    set_attr(data, ALIAS_THETA, proposal.transition.theta_after)
    set_attr(data, ALIAS_DNFR, proposal.dnfr_after)


def commit_transition_network_lifecycle(
    G: TNFRGraph,
    proposal: TransitionNetworkStageProposal,
) -> None:
    """Merge NAV latency and ordered audit state after history commit."""

    data = G.nodes[proposal.node]
    Transition._commit_latency(G, proposal.node, proposal.transition.latency)
    data["_regime_before"] = proposal.transition.regime
    G.graph.setdefault("_nav_transitions", []).append(
        _transition_event(
            proposal.node,
            proposal.transition,
            proposal.handler_dnfr_after,
            proposal.dnfr_after,
        )
    )


def emit_transition_network_warnings(
    G: TNFRGraph,
    proposal: TransitionNetworkStageProposal,
) -> None:
    """Publish latency warnings before the network stage's first write."""

    Transition()._emit_latency_warnings(
        G,
        proposal.node,
        proposal.transition.latency,
    )


class Transition(Operator):
    """Guide structural handoff; adjust theta, nu_f, and DeltaNFR per regime."""

    __slots__ = ()
    name: ClassVar[str] = TRANSITION
    glyph: ClassVar[Glyph] = Glyph.NAV

    def _validate_application_preconditions(
        self,
        G: TNFRGraph,
        node: Any,
        *,
        _stage_now: datetime | None = None,
        **kw: Any,
    ) -> None:
        """Validate NAV's full public boundary before grammar or metadata."""

        self._validate_request_boundary(G, node, kw, now=_stage_now)
        run_preconditions = bool(kw.get("validate_preconditions", True)) or bool(
            G.graph.get("VALIDATE_PRECONDITIONS", False)
        )
        if run_preconditions:
            self._validate_precondition_config(G)
            self._validate_preconditions(G, node)

    def _validate_request_boundary(
        self,
        G: TNFRGraph,
        node: Any,
        kw: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> None:
        """Reject malformed state, flags, arguments, and sinks without mutation."""

        validate_common_execution_arguments(G.graph, kw, operator=self.name)
        for key in (
            "VALIDATE_PRECONDITIONS",
            "NAV_STRICT",
            "NAV_RANDOM",
        ):
            if key in G.graph:
                strict_bool(G.graph[key], operator=self.name, label=key)

        epi, vf, _, _, latent = self._read_state(G, node)
        regime = self._regime_from_state(epi, vf, latent)
        finite_real(
            kw.get("vf_factor", 1.0),
            operator=self.name,
            label="vf_factor",
            lower=0.0,
        )
        finite_real(
            kw.get("phase_shift", self._default_phase_shift(regime)),
            operator=self.name,
            label="phase_shift",
        )
        self._validate_sinks_and_monitor(G, node, kw)
        self._prepare_latency(G, node, now=now)

    def _validate_precondition_config(self, G: TNFRGraph) -> None:
        """Validate every threshold consumed by ``validate_transition``."""

        if "NAV_STRICT_SEQUENCE_CHECK" in G.graph:
            strict_bool(
                G.graph["NAV_STRICT_SEQUENCE_CHECK"],
                operator=self.name,
                label="NAV_STRICT_SEQUENCE_CHECK",
            )
        finite_real(
            G.graph.get("NAV_MIN_VF", 0.01),
            operator=self.name,
            label="NAV_MIN_VF",
            lower=0.0,
        )
        finite_real(
            G.graph.get("NAV_MAX_DNFR", 1.0),
            operator=self.name,
            label="NAV_MAX_DNFR",
            lower=0.0,
        )
        finite_real(
            G.graph.get("NAV_MIN_EPI_FROM_LATENCY", 0.05),
            operator=self.name,
            label="NAV_MIN_EPI_FROM_LATENCY",
            lower=0.0,
        )

    def _validate_sinks_and_monitor(
        self, G: TNFRGraph, node: Any, kw: Mapping[str, Any]
    ) -> None:
        """Validate every append target used by an accepted NAV request."""

        for key in ("_nav_transitions", "recognized_coherence_patterns"):
            require_list_sink(G.graph, key, operator=self.name)
        collect_metrics = bool(kw.get("collect_metrics", False)) or bool(
            G.graph.get("COLLECT_OPERATOR_METRICS", False)
        )
        if collect_metrics:
            require_list_sink(G.graph, "operator_metrics", operator=self.name)
            if "silence_duration" in G.nodes[node]:
                finite_real(
                    G.nodes[node]["silence_duration"],
                    operator=self.name,
                    label="silence_duration",
                    lower=0.0,
                )

        monitor = G.graph.get("integrity_monitor")
        if monitor is not None and not all(
            callable(getattr(monitor, method, None))
            for method in ("before_operator", "after_operator")
        ):
            reject_operator_argument(
                self.name,
                "integrity_monitor must provide callable before_operator and "
                "after_operator methods",
            )

    def _read_state(
        self, G: TNFRGraph, node: Any
    ) -> tuple[float, float, float, float, bool]:
        """Return one finite physical NAV state without permissive coercions."""

        epi = finite_real(
            get_attr(G.nodes[node], ALIAS_EPI, 0.0, strict=True),
            operator=self.name,
            label="EPI state",
        )
        vf = finite_node_real(
            G.nodes[node],
            ALIAS_VF,
            0.0,
            operator=self.name,
            label="nu_f state",
            lower=0.0,
        )
        dnfr = finite_node_real(
            G.nodes[node],
            ALIAS_DNFR,
            0.0,
            operator=self.name,
            label="DeltaNFR state",
        )
        theta = finite_node_real(
            G.nodes[node],
            ALIAS_THETA,
            0.0,
            operator=self.name,
            label="theta state",
        )
        latent = strict_bool(
            G.nodes[node].get("latent", False),
            operator=self.name,
            label="latent",
        )
        return epi, vf, dnfr, theta, latent

    @staticmethod
    def _regime_from_state(epi: float, vf: float, latent: bool) -> str:
        if latent or vf < _VF_LATENT_THRESHOLD:
            return "latent"
        if epi > _EPI_RESONANT_THRESHOLD and vf > _VF_RESONANT_THRESHOLD:
            return "resonant"
        return "active"

    @staticmethod
    def _default_phase_shift(regime: str) -> float:
        return {"latent": 0.1, "active": 0.2, "resonant": 0.15}[regime]

    @staticmethod
    def _retention(regime: str) -> float:
        return {"latent": 0.7, "active": 0.8, "resonant": 0.9}[regime]

    def _prepare_latency(
        self,
        G: TNFRGraph,
        node: Any,
        *,
        now: datetime | None = None,
    ) -> _LatencyProposal:
        """Parse latency inputs without clearing or creating any metadata."""

        data = G.nodes[node]
        active = strict_bool(
            data.get("latent", False),
            operator=self.name,
            label="latent",
        )
        if not active:
            return _LatencyProposal(active=False)

        duration: float | None = None
        max_duration: float | None = None
        extended = False
        if "latency_start_time" in data:
            raw_start = data["latency_start_time"]
            if not isinstance(raw_start, str):
                reject_operator_argument(
                    self.name, "latency_start_time must be an ISO-8601 string"
                )
            try:
                start = datetime.fromisoformat(raw_start)
            except ValueError as exc:
                reject_operator_argument(
                    self.name,
                    f"latency_start_time must be valid ISO-8601: {exc}",
                )
            if start.tzinfo is None or start.utcoffset() is None:
                reject_operator_argument(
                    self.name, "latency_start_time must include a UTC offset"
                )
            observed_at = datetime.now(timezone.utc) if now is None else now
            if observed_at.tzinfo is None or observed_at.utcoffset() is None:
                raise RuntimeError("NAV stage time must include a UTC offset")
            duration = finite_real(
                (observed_at - start.astimezone(observed_at.tzinfo)).total_seconds(),
                operator=self.name,
                label="silence duration",
                lower=0.0,
            )
            if "MAX_SILENCE_DURATION" in G.graph:
                max_duration = finite_real(
                    G.graph["MAX_SILENCE_DURATION"],
                    operator=self.name,
                    label="MAX_SILENCE_DURATION",
                    lower=0.0,
                )
                extended = duration > max_duration

        preserved_epi: float | None = None
        current_epi: float | None = None
        epi_drift: float | None = None
        drifted = False
        if data.get("preserved_epi") is not None:
            preserved_epi = finite_real(
                data["preserved_epi"],
                operator=self.name,
                label="preserved_epi",
            )
            current_epi = finite_real(
                get_attr(data, ALIAS_EPI, 0.0, strict=True),
                operator=self.name,
                label="EPI state",
            )
            epi_drift = finite_real(
                abs(current_epi - preserved_epi),
                operator=self.name,
                label="EPI latency drift",
                lower=0.0,
            )
            drifted = epi_drift > _EPI_DRIFT_TOLERANCE * abs(preserved_epi)

        return _LatencyProposal(
            active=True,
            duration=duration,
            max_duration=max_duration,
            extended=extended,
            preserved_epi=preserved_epi,
            current_epi=current_epi,
            epi_drift=epi_drift,
            drifted=drifted,
        )

    def _build_preflight(
        self,
        G: TNFRGraph,
        node: Any,
        *,
        now: datetime | None = None,
        **kw: Any,
    ) -> _TransitionPreflight:
        """Build the deterministic proposal or the random output envelope."""

        self._validate_request_boundary(G, node, kw, now=now)
        epi, vf, dnfr, theta, latent = self._read_state(G, node)
        regime = self._regime_from_state(epi, vf, latent)
        vf_factor = finite_real(
            kw.get("vf_factor", 1.0),
            operator=self.name,
            label="vf_factor",
            lower=0.0,
        )
        phase_shift = finite_real(
            kw.get("phase_shift", self._default_phase_shift(regime)),
            operator=self.name,
            label="phase_shift",
        )

        vf_multiplier = {"latent": 1.2, "resonant": 0.95}.get(
            regime, vf_factor
        )
        vf_after = finite_real(
            vf * vf_multiplier,
            operator=self.name,
            label="nu_f proposal",
            lower=0.0,
        )
        theta_normalized = finite_real(
            theta % math.tau,
            operator=self.name,
            label="normalized theta state",
            lower=0.0,
            upper=math.tau,
        )
        theta_after = finite_real(
            (theta_normalized + phase_shift) % math.tau,
            operator=self.name,
            label="theta proposal",
            lower=0.0,
            upper=math.tau,
        )
        phase_applied = finite_real(
            angle_diff(theta_after, theta_normalized),
            operator=self.name,
            label="applied phase shift",
        )

        factors = resolve_runtime_operator_factors(
            G.graph.get("GLYPH_FACTORS"), self.glyph, G.graph
        )
        strict = strict_bool(
            G.graph.get("NAV_STRICT", False),
            operator=self.name,
            label="NAV_STRICT",
        )
        random_mode = strict_bool(
            G.graph.get("NAV_RANDOM", True),
            operator=self.name,
            label="NAV_RANDOM",
        )
        if strict:
            base = vf
        else:
            eta = factors["NAV_eta"]
            target = vf if dnfr >= 0.0 else -vf
            base = finite_real(
                (1.0 - eta) * dnfr + eta * target,
                operator=self.name,
                label="deterministic DeltaNFR proposal",
            )
        base = finite_real(
            base,
            operator=self.name,
            label="deterministic DeltaNFR proposal",
        )
        jitter = factors["NAV_jitter"]

        handler_after: float | None
        if random_mode and jitter > 0.0:
            handler_after = None
            handler_low = finite_real(
                base - jitter,
                operator=self.name,
                label="random DeltaNFR lower proposal bound",
            )
            handler_high = finite_real(
                base + jitter,
                operator=self.name,
                label="random DeltaNFR upper proposal bound",
            )
        else:
            signed_jitter = (
                0.0 if random_mode else (jitter if base >= 0.0 else -jitter)
            )
            handler_after = finite_real(
                base + signed_jitter,
                operator=self.name,
                label="DeltaNFR handler proposal",
            )
            if handler_after == dnfr:
                reject_operator_argument(
                    self.name,
                    "NAV must change DeltaNFR before recording its history",
                )
            handler_low = handler_high = handler_after

        retention = self._retention(regime)
        dnfr_low = finite_real(
            handler_low * retention,
            operator=self.name,
            label="final DeltaNFR lower proposal bound",
        )
        dnfr_high = finite_real(
            handler_high * retention,
            operator=self.name,
            label="final DeltaNFR upper proposal bound",
        )
        dnfr_after = (
            finite_real(
                handler_after * retention,
                operator=self.name,
                label="final DeltaNFR proposal",
            )
            if handler_after is not None
            else None
        )
        if (
            dnfr_after is not None
            and vf_after == vf
            and phase_applied == 0.0
            and dnfr_after == dnfr
        ):
            reject_operator_argument(
                self.name, "NAV proposal must change at least one nodal channel"
            )

        return _TransitionPreflight(
            regime=regime,
            epi_before=epi,
            vf_before=vf,
            vf_after=vf_after,
            theta_before=theta,
            theta_after=theta_after,
            phase_shift_requested=phase_shift,
            phase_shift_applied=phase_applied,
            dnfr_before=dnfr,
            handler_base=base,
            jitter_amplitude=jitter,
            random_mode=random_mode,
            retention=retention,
            handler_dnfr_after=handler_after,
            dnfr_after=dnfr_after,
            handler_dnfr_bounds=(handler_low, handler_high),
            dnfr_bounds=(min(dnfr_low, dnfr_high), max(dnfr_low, dnfr_high)),
            latency=self._prepare_latency(G, node, now=now),
        )

    def _build_network_stage_proposal(
        self,
        G: TNFRGraph,
        node: Any,
        *,
        now: datetime,
        resolved_seed: int | None,
        node_offset: int | None,
        **kw: Any,
    ) -> TransitionNetworkStageProposal:
        """Bind NAV's random draw and all outputs without changing ``G``."""

        proposal = self._build_preflight(G, node, now=now, **kw)
        handler_dnfr = proposal.handler_dnfr_after
        jitter_proposal = None
        if handler_dnfr is None:
            from .jitter import _JITTER_PROGRESS_KEY, propose_jitter_draw

            if resolved_seed is None or node_offset is None:
                raise RuntimeError("Random NAV stage proposal lacks stream identity")
            try:
                jitter_proposal = propose_jitter_draw(
                    proposal.jitter_amplitude,
                    seed=resolved_seed,
                    offset=node_offset,
                    progress_state=G.nodes[node].get(_JITTER_PROGRESS_KEY),
                )
            except ValueError as exc:
                reject_operator_argument(self.name, str(exc))
            handler_dnfr = finite_real(
                proposal.handler_base + jitter_proposal.value,
                operator=self.name,
                label="random DeltaNFR proposal",
            )
            if handler_dnfr == proposal.dnfr_before:
                reject_operator_argument(
                    self.name,
                    "NAV must change DeltaNFR before recording its history",
                )

        dnfr_after = finite_real(
            handler_dnfr * proposal.retention,
            operator=self.name,
            label="final DeltaNFR proposal",
        )
        low, high = proposal.dnfr_bounds
        if not low <= dnfr_after <= high:
            raise RuntimeError("NAV result escaped its immutable proposal bounds")
        if (
            proposal.vf_after == proposal.vf_before
            and proposal.phase_shift_applied == 0.0
            and dnfr_after == proposal.dnfr_before
        ):
            reject_operator_argument(
                self.name, "NAV proposal must change at least one nodal channel"
            )
        return TransitionNetworkStageProposal(
            node=node,
            transition=proposal,
            handler_dnfr_after=handler_dnfr,
            dnfr_after=dnfr_after,
            jitter_proposal=jitter_proposal,
        )

    def _execute(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Dispatch NAV only after its complete public proposal is valid."""

        proposal = self._build_preflight(G, node, **kw)
        collect_metrics = bool(kw.get("collect_metrics", False)) or bool(
            G.graph.get("COLLECT_OPERATOR_METRICS", False)
        )
        validate_equation = bool(kw.get("validate_nodal_equation", False)) or bool(
            G.graph.get("VALIDATE_NODAL_EQUATION", False)
        )
        state_before = None
        if collect_metrics or validate_equation:
            state_before = {
                "epi": proposal.epi_before,
                "vf": proposal.vf_before,
                "dnfr": proposal.dnfr_before,
                "theta": proposal.theta_before,
            }

        self._emit_latency_warnings(G, node, proposal.latency)
        integrity_monitor = G.graph.get("integrity_monitor")
        if integrity_monitor is not None:
            integrity_monitor.before_operator(G, node)

        from .grammar_application import _apply_selected_glyph

        _apply_selected_glyph(G, node, self.glyph, kw.get("window"))
        handler_dnfr = finite_node_real(
            G.nodes[node],
            ALIAS_DNFR,
            0.0,
            operator=self.name,
            label="DeltaNFR handler result",
        )
        handler_low, handler_high = proposal.handler_dnfr_bounds
        if not handler_low <= handler_dnfr <= handler_high:
            reject_operator_argument(
                self.name,
                "DeltaNFR handler result escaped the validated proposal bounds",
            )
        dnfr_after = finite_real(
            handler_dnfr * proposal.retention,
            operator=self.name,
            label="final DeltaNFR result",
        )
        dnfr_low, dnfr_high = proposal.dnfr_bounds
        if not dnfr_low <= dnfr_after <= dnfr_high:
            reject_operator_argument(
                self.name,
                "final DeltaNFR result escaped the validated proposal bounds",
            )

        set_attr(G.nodes[node], ALIAS_VF, proposal.vf_after)
        set_attr(G.nodes[node], ALIAS_THETA, proposal.theta_after)
        set_attr(G.nodes[node], ALIAS_DNFR, dnfr_after)
        self._commit_latency(G, node, proposal.latency)
        G.nodes[node]["_regime_before"] = proposal.regime
        self._record_transition(G, node, proposal, handler_dnfr, dnfr_after)

        if integrity_monitor is not None:
            integrity_monitor.after_operator(G, node, self.name)

        if validate_equation and state_before is not None:
            from .nodal_equation import validate_nodal_equation

            validate_nodal_equation(
                G,
                node,
                epi_before=state_before["epi"],
                epi_after=finite_real(
                    get_attr(G.nodes[node], ALIAS_EPI, 0.0, strict=True),
                    operator=self.name,
                    label="EPI result",
                ),
                dt=finite_real(
                    kw.get("dt", 1.0),
                    operator=self.name,
                    label="dt",
                    lower=math.nextafter(0.0, math.inf),
                ),
                operator_name=self.name,
                strict=strict_bool(
                    G.graph.get("NODAL_EQUATION_STRICT", False),
                    operator=self.name,
                    label="NODAL_EQUATION_STRICT",
                ),
            )

        if collect_metrics and state_before is not None:
            G.graph.setdefault("operator_metrics", []).append(
                self._collect_metrics(G, node, state_before)
            )

    def _record_transition(
        self,
        G: TNFRGraph,
        node: Any,
        proposal: _TransitionPreflight,
        handler_dnfr: float,
        dnfr_after: float,
    ) -> None:
        """Append telemetry using only previously validated scalar values."""

        G.graph.setdefault("_nav_transitions", []).append(
            _transition_event(node, proposal, handler_dnfr, dnfr_after)
        )

    def _emit_latency_warnings(
        self, G: TNFRGraph, node: Any, proposal: _LatencyProposal
    ) -> None:
        """Emit staged latency diagnostics while the graph is still unchanged."""

        if proposal.extended:
            duration = proposal.duration
            max_duration = proposal.max_duration
            assert duration is not None and max_duration is not None
            warnings.warn(
                f"Node {node} transitioning after extended silence "
                f"(duration: {duration:.2f}s, max: {max_duration:.2f}s)",
                stacklevel=4,
            )
        if proposal.drifted:
            epi_drift = proposal.epi_drift
            preserved_epi = proposal.preserved_epi
            current_epi = proposal.current_epi
            assert (
                epi_drift is not None
                and preserved_epi is not None
                and current_epi is not None
            )
            warnings.warn(
                f"Node {node} EPI drift drift={epi_drift:.3f} "
                f"pres={preserved_epi:.3f} cur={current_epi:.3f}",
                stacklevel=4,
            )

    @staticmethod
    def _commit_latency(
        G: TNFRGraph, node: Any, proposal: _LatencyProposal
    ) -> None:
        """Clear latency state only after low-level NAV has succeeded."""

        if not proposal.active:
            return
        if proposal.duration is not None:
            G.nodes[node]["silence_duration"] = proposal.duration
        G.nodes[node].pop("latent", None)
        G.nodes[node].pop("latency_start_time", None)
        G.nodes[node].pop("preserved_epi", None)

    def _detect_regime(self, G: TNFRGraph, node: Any) -> str:
        """Return a validated regime label: latent, active, or resonant."""

        epi, vf, _, _, latent = self._read_state(G, node)
        return self._regime_from_state(epi, vf, latent)

    def _handle_latency_transition(self, G: TNFRGraph, node: Any) -> None:
        """Validate and commit latency metadata for direct compatibility calls."""

        proposal = self._prepare_latency(G, node)
        self._emit_latency_warnings(G, node, proposal)
        self._commit_latency(G, node, proposal)

    def _apply_structural_transition(
        self, G: TNFRGraph, node: Any, regime: str, **kw: Any
    ) -> None:
        """Apply a checked post-handler transition for compatibility calls."""

        if regime not in {"latent", "active", "resonant"}:
            reject_operator_argument(self.name, f"unknown NAV regime {regime!r}")
        _, vf, dnfr, theta, _ = self._read_state(G, node)
        vf_factor = finite_real(
            kw.get("vf_factor", 1.0),
            operator=self.name,
            label="vf_factor",
            lower=0.0,
        )
        phase_shift = finite_real(
            kw.get("phase_shift", self._default_phase_shift(regime)),
            operator=self.name,
            label="phase_shift",
        )
        vf_multiplier = {"latent": 1.2, "resonant": 0.95}.get(
            regime, vf_factor
        )
        vf_after = finite_real(
            vf * vf_multiplier,
            operator=self.name,
            label="nu_f proposal",
            lower=0.0,
        )
        theta_normalized = theta % math.tau
        theta_after = finite_real(
            (theta_normalized + phase_shift) % math.tau,
            operator=self.name,
            label="theta proposal",
            lower=0.0,
            upper=math.tau,
        )
        dnfr_after = finite_real(
            dnfr * self._retention(regime),
            operator=self.name,
            label="final DeltaNFR proposal",
        )
        require_list_sink(G.graph, "_nav_transitions", operator=self.name)
        set_attr(G.nodes[node], ALIAS_VF, vf_after)
        set_attr(G.nodes[node], ALIAS_THETA, theta_after)
        set_attr(G.nodes[node], ALIAS_DNFR, dnfr_after)
        applied = finite_real(
            angle_diff(theta_after, theta_normalized),
            operator=self.name,
            label="applied phase shift",
        )
        G.graph.setdefault("_nav_transitions", []).append(
            {
                "node": node,
                "regime_origin": regime,
                "vf_before": vf,
                "vf_after": vf_after,
                "theta_before": theta,
                "theta_after": theta_after,
                "dnfr_before": dnfr,
                "dnfr_after": dnfr_after,
                "phase_shift": applied,
                "phase_shift_requested": phase_shift,
            }
        )

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Run NAV precondition validator."""

        from .preconditions import validate_transition

        validate_transition(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect NAV metrics after the complete structural transition."""

        from .metrics import transition_metrics

        return transition_metrics(
            G,
            node,
            state_before["dnfr"],
            state_before["vf"],
            state_before["theta"],
            epi_before=state_before.get("epi"),
        )
