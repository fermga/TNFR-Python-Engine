"""Strict serializable evidence envelopes for TNFR research artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .certificates import NumericalCertificate
from .manifests import ExperimentManifest

__all__ = ["EvidenceSidecar", "EvidenceAdmissionError"]


class EvidenceAdmissionError(ValueError):
    """Raised when an evidence envelope omits required scientific context."""


@dataclass(frozen=True)
class EvidenceSidecar:
    """Attach reproducibility and scope context to a research artifact.

    The envelope is deliberately independent of a particular experiment. It
    makes the model, norm, clock, finite horizon, provenance, observation, and
    numerical context explicit before an artifact is admitted for reuse.
    """

    manifest: ExperimentManifest
    artifact: str
    model: str
    norm: str
    distance_convention: str
    clock: str
    finite_horizon: float | None
    tail_status: str
    provenance: dict[str, bool]
    certificate: NumericalCertificate | None = None
    claim_statement: str = ""
    claim_status: str = ""
    scope: str = ""
    assumptions: tuple[str, ...] = ()
    outcome: str = ""
    source_imports: tuple[str, ...] = ()
    dirty_source_hash: str = ""
    graph_context: dict[str, Any] = field(default_factory=dict)
    state_context: dict[str, Any] = field(default_factory=dict)
    numerical_context: dict[str, Any] = field(default_factory=dict)
    observation_context: dict[str, Any] = field(default_factory=dict)
    cost_context: dict[str, Any] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", dict(self.provenance))
        for name in (
            "graph_context",
            "state_context",
            "numerical_context",
            "observation_context",
            "cost_context",
            "artifact_hashes",
        ):
            object.__setattr__(self, name, dict(getattr(self, name)))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "source_imports", tuple(self.source_imports))

    def validate_for_admission(self) -> None:
        """Require enough context to distinguish evidence from an assertion."""
        self.manifest.validate_for_admission()
        if not self.artifact or not self.model or not self.norm:
            raise EvidenceAdmissionError(
                "artifact, model and norm are required"
            )
        if not self.distance_convention or not self.clock:
            raise EvidenceAdmissionError(
                "distance_convention and clock are required"
            )
        if self.finite_horizon is not None and self.finite_horizon < 0.0:
            raise EvidenceAdmissionError("finite_horizon must be nonnegative")
        if not self.tail_status:
            raise EvidenceAdmissionError("tail_status is required")
        if not self.claim_statement or not self.claim_status or not self.scope:
            raise EvidenceAdmissionError(
                "claim_statement, claim_status and scope are required"
            )
        if not self.assumptions or not self.outcome:
            raise EvidenceAdmissionError(
                "assumptions and outcome are required"
            )
        if not self.source_imports or not self.dirty_source_hash:
            raise EvidenceAdmissionError(
                "source_imports and dirty_source_hash are required"
            )
        contexts = (
            self.graph_context,
            self.state_context,
            self.numerical_context,
            self.observation_context,
            self.cost_context,
            self.artifact_hashes,
        )
        if any(not context for context in contexts):
            raise EvidenceAdmissionError(
                "graph, state, numerical, observation, cost and hash contexts "
                "must be non-empty"
            )
        required = {
            "uses_known_factors",
            "uses_target_labels",
            "uses_expected_answers",
        }
        if set(self.provenance) != required:
            raise EvidenceAdmissionError(
                "provenance must explicitly answer every admission field"
            )
        if not all(
            isinstance(value, bool) for value in self.provenance.values()
        ):
            raise EvidenceAdmissionError("provenance values must be boolean")
        if self.certificate is not None:
            self.certificate.validate_for_admission()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return {
            "manifest": self.manifest.to_dict(),
            "artifact": self.artifact,
            "model": self.model,
            "norm": self.norm,
            "distance_convention": self.distance_convention,
            "clock": self.clock,
            "finite_horizon": self.finite_horizon,
            "tail_status": self.tail_status,
            "provenance": dict(self.provenance),
            "certificate": (
                None
                if self.certificate is None
                else self.certificate.to_dict()
            ),
            "claim_statement": self.claim_statement,
            "claim_status": self.claim_status,
            "scope": self.scope,
            "assumptions": list(self.assumptions),
            "outcome": self.outcome,
            "source_imports": list(self.source_imports),
            "dirty_source_hash": self.dirty_source_hash,
            "graph_context": dict(self.graph_context),
            "state_context": dict(self.state_context),
            "numerical_context": dict(self.numerical_context),
            "observation_context": dict(self.observation_context),
            "cost_context": dict(self.cost_context),
            "artifact_hashes": dict(self.artifact_hashes),
            "stop_reason": self.stop_reason,
        }

    def write_admitted(self, path: str | Path) -> Path:
        """Validate, then atomically serialize the evidence envelope."""
        self.validate_for_admission()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
        return destination
