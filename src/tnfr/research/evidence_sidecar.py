"""Strict serializable evidence envelopes for TNFR research artifacts."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

from ..utils.io import json_dumps, safe_write
from .certificates import NumericalCertificate
from .core_manifests import CoreExperimentManifest
from .manifests import ExperimentManifest

__all__ = ["EvidenceSidecar", "EvidenceAdmissionError"]


class EvidenceAdmissionError(ValueError):
    """Raised when an evidence envelope omits required scientific context."""


_ARITHMETIC_PROVENANCE = frozenset(
    (
        "uses_known_factors",
        "uses_target_labels",
        "uses_expected_answers",
    )
)
_CORE_PROVENANCE = frozenset(
    (
        "uses_future_samples",
        "uses_outcome_derived_wiring",
        "fits_on_evaluation_data",
        "uses_evaluation_labels",
        "uses_postselection",
    )
)
_CONTEXTS = (
    "graph_context",
    "state_context",
    "numerical_context",
    "observation_context",
    "cost_context",
    "artifact_hashes",
)


def _freeze(value: Any) -> Any:
    """Detach JSON metadata and freeze nested containers; never coerce objects."""
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise EvidenceAdmissionError("context mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise EvidenceAdmissionError("context must contain finite JSON values")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _artifact_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise EvidenceAdmissionError("artifact paths must be nonempty relative paths")
    name = value.replace("\\", "/")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or PureWindowsPath(name).drive
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise EvidenceAdmissionError("artifact paths must stay within root_dir")
    return path.as_posix()


def _digest(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"(?:sha256:)?[0-9a-fA-F]{64}", value
    ):
        raise EvidenceAdmissionError("artifact hashes must be SHA-256 digests")
    return value.removeprefix("sha256:").lower()


@dataclass(frozen=True)
class EvidenceSidecar:
    """Attach reproducibility and scope context to a research artifact.

    ``validate_metadata`` checks declarations without reading files.
    ``validate_for_admission(root_dir=...)`` also verifies the declared artifact
    bytes at that invocation. Neither operation proves a claim, causal ordering,
    absence of leakage, or that the files remain unchanged afterwards.

    Core manifests use five explicit measurement-provenance questions; legacy
    arithmetic manifests retain their three original questions. True answers
    are retained for descriptive studies, not converted to discovery permission.
    """

    manifest: ExperimentManifest | CoreExperimentManifest
    artifact: str
    model: str
    norm: str
    distance_convention: str
    clock: str
    finite_horizon: float | None
    tail_status: str
    provenance: Mapping[str, bool]
    certificate: NumericalCertificate | None = None
    claim_statement: str = ""
    claim_status: str = ""
    scope: str = ""
    assumptions: tuple[str, ...] = ()
    outcome: str = ""
    source_imports: tuple[str, ...] = ()
    dirty_source_hash: str = ""
    graph_context: Mapping[str, Any] = field(default_factory=dict)
    state_context: Mapping[str, Any] = field(default_factory=dict)
    numerical_context: Mapping[str, Any] = field(default_factory=dict)
    observation_context: Mapping[str, Any] = field(default_factory=dict)
    cost_context: Mapping[str, Any] = field(default_factory=dict)
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.manifest) is CoreExperimentManifest:
            manifest = CoreExperimentManifest.from_dict(self.manifest.to_dict())
        elif type(self.manifest) is ExperimentManifest:
            manifest = ExperimentManifest.from_dict(
                self.manifest.to_dict(), strict=True
            )
            for item in fields(manifest):
                object.__setattr__(
                    manifest, item.name, _freeze(getattr(manifest, item.name))
                )
        else:
            raise EvidenceAdmissionError(
                "manifest must be a core or arithmetic manifest"
            )
        object.__setattr__(self, "manifest", manifest)
        if self.certificate is not None:
            if type(self.certificate) is not NumericalCertificate:
                raise EvidenceAdmissionError(
                    "certificate must be a NumericalCertificate"
                )
            object.__setattr__(self, "certificate", replace(self.certificate))
        for name in ("provenance", *_CONTEXTS):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise EvidenceAdmissionError(f"{name} must be a mapping")
            object.__setattr__(self, name, _freeze(value))
        for name in ("assumptions", "source_imports"):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)):
                raise EvidenceAdmissionError(f"{name} must be a sequence of strings")
            values = tuple(values)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise EvidenceAdmissionError(f"{name} must contain nonempty strings")
            object.__setattr__(self, name, values)

    def validate_metadata(self) -> None:
        """Validate declared context only; this does not authenticate files."""
        self.manifest.validate_for_admission()
        for name in (
            "artifact",
            "model",
            "norm",
            "distance_convention",
            "clock",
            "tail_status",
            "claim_statement",
            "claim_status",
            "scope",
            "outcome",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EvidenceAdmissionError(f"{name} must be nonempty text")
        if self.finite_horizon is not None:
            value = self.finite_horizon
            try:
                valid = (
                    not isinstance(value, bool)
                    and isinstance(value, Real)
                    and math.isfinite(value)
                    and value >= 0
                )
            except OverflowError:
                valid = False
            if not valid:
                raise EvidenceAdmissionError(
                    "finite_horizon must be finite and nonnegative"
                )
        if not self.assumptions:
            raise EvidenceAdmissionError("assumptions are required")
        if not self.source_imports:
            raise EvidenceAdmissionError("source_imports are required")
        is_core = isinstance(self.manifest, CoreExperimentManifest)
        if is_core:
            expected = self.manifest.dirty_source_hash or ""
            if self.dirty_source_hash != expected:
                raise EvidenceAdmissionError(
                    "dirty_source_hash must match the core manifest"
                )
            if self.claim_status != self.manifest.result_status.value:
                raise EvidenceAdmissionError(
                    "claim_status must match the core manifest"
                )
        elif not self.dirty_source_hash:
            raise EvidenceAdmissionError("dirty_source_hash is required")
        if self.dirty_source_hash:
            _digest(self.dirty_source_hash)
        if self.stop_reason is not None and (
            not isinstance(self.stop_reason, str) or not self.stop_reason.strip()
        ):
            raise EvidenceAdmissionError("stop_reason must be nonempty text or None")
        if any(not getattr(self, name) for name in _CONTEXTS):
            raise EvidenceAdmissionError(
                "graph, state, numerical, observation, cost and hash contexts "
                "must be non-empty"
            )
        required = _CORE_PROVENANCE if is_core else _ARITHMETIC_PROVENANCE
        if set(self.provenance) != required:
            raise EvidenceAdmissionError(
                "provenance must explicitly answer every admission field"
            )
        if not all(type(value) is bool for value in self.provenance.values()):
            raise EvidenceAdmissionError("provenance values must be boolean")
        if self.certificate is not None:
            self.certificate.validate_for_admission()
        hashes = {}
        for name, digest in self.artifact_hashes.items():
            normalized = _artifact_name(name)
            if normalized in hashes:
                raise EvidenceAdmissionError(
                    "artifact paths must be unique after normalization"
                )
            hashes[normalized] = _digest(digest)
        required_artifacts = (self.artifact, *self.manifest.artifacts)
        if any(_artifact_name(name) not in hashes for name in required_artifacts):
            raise EvidenceAdmissionError(
                "every declared artifact requires its SHA-256 digest"
            )
        # Also validate scalar fields and detached manifest/certificate payloads.
        try:
            json_dumps(self.to_dict(), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise EvidenceAdmissionError(
                "evidence metadata must be finite JSON"
            ) from exc

    def _verified_paths(self, root_dir: str | Path) -> tuple[Path, ...]:
        root = Path(root_dir).resolve(strict=True)
        if not root.is_dir():
            raise EvidenceAdmissionError("root_dir must be an existing directory")
        paths = []
        for name, expected in self.artifact_hashes.items():
            path = (root / _artifact_name(name)).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise EvidenceAdmissionError(
                    "artifact must be an existing file within root_dir"
                )
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 16), b""):
                    digest.update(chunk)
            if digest.hexdigest() != _digest(expected):
                raise EvidenceAdmissionError(f"artifact SHA-256 mismatch: {name}")
            paths.append(path)
        return tuple(paths)

    def validate_for_admission(self, *, root_dir: str | Path) -> None:
        """Validate metadata and actual artifact bytes under an explicit root."""
        self.validate_metadata()
        self._verified_paths(root_dir)

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
                None if self.certificate is None else self.certificate.to_dict()
            ),
            "claim_statement": self.claim_statement,
            "claim_status": self.claim_status,
            "scope": self.scope,
            "assumptions": list(self.assumptions),
            "outcome": self.outcome,
            "source_imports": list(self.source_imports),
            "dirty_source_hash": self.dirty_source_hash,
            "graph_context": _thaw(self.graph_context),
            "state_context": _thaw(self.state_context),
            "numerical_context": _thaw(self.numerical_context),
            "observation_context": _thaw(self.observation_context),
            "cost_context": _thaw(self.cost_context),
            "artifact_hashes": dict(self.artifact_hashes),
            "stop_reason": self.stop_reason,
        }

    def write_admitted(self, path: str | Path, *, root_dir: str | Path) -> Path:
        """Verify bytes and atomically write; no claim-truth or chronology proof."""
        self.validate_metadata()
        artifacts = self._verified_paths(root_dir)
        destination = Path(path)
        if destination.resolve() in artifacts:
            raise EvidenceAdmissionError(
                "sidecar destination must not overwrite an artifact"
            )
        payload = json_dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"

        def write(stream):
            stream.write(payload)
            # Detect changes during serialization before committing the sidecar.
            self._verified_paths(root_dir)

        safe_write(destination, write)
        return destination
