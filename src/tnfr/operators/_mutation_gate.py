"""Runtime adapter for the canonical ZHIR positive-growth certificate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..constants.canonical import ZHIR_THRESHOLD_XI_CANONICAL
from ..physics.mutation_trigger import (
    MutationTriggerCertificate,
    MutationTriggerInputError,
    certify_mutation_trigger,
)
from ..types import BEPIProtocol, scalarize_epi

__all__ = (
    "MutationThresholdSample",
    "MutationRuntimeGate",
    "mutation_threshold_sample",
    "validate_mutation_capacity",
    "validate_mutation_runtime_gate",
    "validate_mutation_threshold",
)


@dataclass(frozen=True, slots=True)
class MutationThresholdSample:
    """Two-point observation used by the ZHIR runtime gate."""

    previous_epi: float
    current_epi: float
    depi_dt: float
    xi: float
    history_key: str
    sample_interval: float
    time_basis: str
    physical_time_resolved: bool
    certificate: MutationTriggerCertificate

    @property
    def crossed(self) -> bool:
        """Whether the signed rate strictly exceeds the configured threshold."""

        return self.depi_dt > self.xi


@dataclass(frozen=True, slots=True)
class MutationRuntimeGate:
    """Complete non-disableable ZHIR gate at one runtime state."""

    threshold: MutationThresholdSample
    nu_f: float
    minimum_nu_f: float


def _reject(detail: str) -> NoReturn:
    # Imported lazily so the pure helper does not form an import-time cycle
    # with the preconditions package that exposes its diagnostics.
    from .preconditions import OperatorPreconditionError

    raise OperatorPreconditionError("Mutation", detail)


def _first_alias(
    node_data: Mapping[str, Any], aliases: tuple[str, ...], default: Any
) -> Any:
    for key in aliases:
        if key in node_data:
            return node_data[key]
    return default


def _certificate(
    node_data: Mapping[str, Any], graph_data: Mapping[str, Any]
) -> MutationTriggerCertificate:
    """Build the shared pure certificate from runtime mappings."""

    physical_history = node_data.get("epi_time_history")
    current_epi = _first_alias(node_data, ALIAS_EPI, None)
    if current_epi is None:
        # Legacy threshold-only callers historically needed only the sampled
        # values.  Timestamped evidence, by contrast, must be fresh against an
        # actual current EPI endpoint.
        if physical_history is not None:
            _reject(
                "ZHIR timestamped evidence requires an explicit current EPI value"
            )
        current_epi = 0.0

    serialized_bepi = isinstance(current_epi, Mapping) and {
        "continuous",
        "discrete",
        "grid",
    }.issubset(current_epi)
    if isinstance(current_epi, BEPIProtocol) or serialized_bepi:
        try:
            current_epi = scalarize_epi(current_epi)
        except (OverflowError, TypeError, ValueError) as exc:
            _reject(
                "current_epi must have a finite canonical scalar projection; "
                f"got {current_epi!r}"
            )

    try:
        return certify_mutation_trigger(
            current_epi=current_epi,
            nu_f=_first_alias(node_data, ALIAS_VF, 0.0),
            delta_nfr=_first_alias(node_data, ALIAS_DNFR, 0.0),
            xi=graph_data.get(
                "ZHIR_THRESHOLD_XI", ZHIR_THRESHOLD_XI_CANONICAL
            ),
            epi_time_history=physical_history,
            epi_history=node_data.get("epi_history"),
            legacy_epi_history=node_data.get("_epi_history"),
        )
    except MutationTriggerInputError as exc:
        label = "ZHIR_THRESHOLD_XI" if exc.field == "xi" else exc.field
        _reject(f"{label} {exc.reason}; got {exc.value!r}")


def mutation_threshold_sample(
    node_data: Mapping[str, Any],
    graph_data: Mapping[str, Any],
) -> MutationThresholdSample:
    """Read and validate the two-point ZHIR sample without mutating metadata.

    Timestamped runtime evidence uses its physical interval. Legacy EPI
    histories retain their explicit unit-operator-step compatibility basis.
    The signed difference is intentional: contraction does not satisfy the
    declared ``dEPI/dt > xi`` trigger.
    """

    certificate = _certificate(node_data, graph_data)
    evidence = certificate.evidence
    if evidence is None or not certificate.evidence_valid:
        reason = certificate.reason or "invalid_history"
        source = certificate.source or "epi_history"
        if reason in {"missing_history", "insufficient_history"}:
            _reject(
                "ZHIR threshold cannot be verified: at least two EPI history "
                "samples are required"
            )
        if reason in {
            "history_not_replayable",
            "invalid_legacy_history_sample",
            "invalid_time_history_sample",
        }:
            _reject(f"{source} must contain finite replayable scalar samples")
        _reject(
            "ZHIR threshold cannot be verified from current EPI evidence: "
            f"{reason}"
        )
    if evidence.observed_depi_dt is None or evidence.sample_interval is None:
        _reject("ZHIR threshold evidence has no finite observed rate")
    return MutationThresholdSample(
        previous_epi=evidence.previous_epi,
        current_epi=evidence.current_epi,
        depi_dt=evidence.observed_depi_dt,
        xi=certificate.xi,
        history_key=evidence.source,
        sample_interval=evidence.sample_interval,
        time_basis=evidence.time_basis,
        physical_time_resolved=evidence.physical_time_resolved,
        certificate=certificate,
    )


def validate_mutation_threshold(
    node_data: Mapping[str, Any],
    graph_data: Mapping[str, Any],
) -> MutationThresholdSample:
    """Require the strict signed ZHIR trigger and return its immutable sample."""

    sample = mutation_threshold_sample(node_data, graph_data)
    if not sample.crossed:
        _reject(
            "ZHIR requires signed dEPI/dt > xi; "
            f"got {sample.depi_dt!r} <= {sample.xi!r}"
        )
    return sample


def validate_mutation_capacity(
    node_data: Mapping[str, Any],
    graph_data: Mapping[str, Any],
) -> tuple[float, float]:
    """Require active structural capacity and an optional configured floor."""

    for key in ALIAS_VF:
        if key in node_data:
            raw_nu_f = node_data[key]
            break
    else:
        _reject("ZHIR requires an explicit structural-frequency value")

    try:
        capacity = certify_mutation_trigger(
            current_epi=0.0,
            nu_f=raw_nu_f,
            delta_nfr=0.0,
            xi=0.0,
        )
    except MutationTriggerInputError as exc:
        _reject(
            "ZHIR structural frequency nu_f "
            f"{exc.reason}; got {exc.value!r}"
        )
    try:
        minimum_certificate = certify_mutation_trigger(
            current_epi=0.0,
            nu_f=graph_data.get("ZHIR_MIN_VF", 0.0),
            delta_nfr=0.0,
            xi=0.0,
        )
    except MutationTriggerInputError as exc:
        _reject(f"ZHIR_MIN_VF {exc.reason}; got {exc.value!r}")
    nu_f = capacity.nu_f
    minimum = minimum_certificate.nu_f
    if nu_f <= 0.0 or nu_f < minimum:
        _reject(
            "ZHIR requires active structural frequency; "
            f"got nu_f={nu_f!r}, configured minimum={minimum!r}"
        )
    return nu_f, minimum


def validate_mutation_runtime_gate(
    node_data: Mapping[str, Any],
    graph_data: Mapping[str, Any],
) -> MutationRuntimeGate:
    """Validate every non-disableable ZHIR execution condition, without writes."""

    nu_f, minimum = validate_mutation_capacity(node_data, graph_data)
    threshold = validate_mutation_threshold(node_data, graph_data)
    return MutationRuntimeGate(
        threshold=threshold,
        nu_f=nu_f,
        minimum_nu_f=minimum,
    )
