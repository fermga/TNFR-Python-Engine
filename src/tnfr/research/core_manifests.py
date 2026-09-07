r"""Domain-neutral reproducibility manifests for TNFR research programs.

The historical :class:`ExperimentManifest` includes arithmetic-specific input
size and known-factor fields.  Core graph-dynamics experiments use this
separate envelope so provenance requirements do not acquire number-theory
semantics by accident.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import re
import subprocess
from types import MappingProxyType
from typing import Any, Iterable

from .claims import ClaimStatus
from .manifests import ManifestValidationError

__all__ = ["CoreExperimentManifest", "current_git_source_provenance"]


def _git_output(repository: Path, *arguments: str) -> bytes:
    """Run one fixed Git query for a research-source snapshot."""
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "source provenance requires a Git checkout with git available"
        ) from exc


def current_git_source_provenance(
    repository: str | Path,
    paths: Iterable[str | Path],
) -> tuple[str, bool, str | None]:
    """Return HEAD and a content digest for a declared working-source scope.

    The digest covers scoped Git status plus the current bytes of every tracked
    or untracked, non-ignored file in ``paths``.  A deleted file remains visible
    through the status bytes.  This records the source that produced a research
    artifact even when the checkout is dirty; it does not modify the checkout.
    """
    root = Path(repository).resolve()
    if not root.is_dir():
        raise ValueError("repository must be an existing directory")
    if isinstance(paths, (str, bytes, Path)):
        raise TypeError("paths must be an iterable of relative source paths")
    source_paths = tuple(str(path).replace("\\", "/") for path in paths)
    if not source_paths or any(not path.strip() for path in source_paths):
        raise ValueError("paths must contain at least one non-empty source path")
    if any(
        Path(path).is_absolute() or ".." in Path(path).parts
        for path in source_paths
    ):
        raise ValueError("source paths must be relative to repository")

    git_sha = _git_output(root, "rev-parse", "HEAD").decode().strip()
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *source_paths,
    )
    if not status:
        return git_sha, False, None

    listed = _git_output(
        root,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        *source_paths,
    )
    digest = hashlib.sha256(b"tnfr-working-source-v1\0" + status)
    for encoded_path in sorted(item for item in listed.split(b"\0") if item):
        relative_path = encoded_path.decode("utf-8", errors="surrogateescape")
        source = root / relative_path
        digest.update(encoded_path)
        digest.update(b"\0")
        digest.update(source.read_bytes() if source.is_file() else b"<missing>")
        digest.update(b"\0")
    return git_sha, True, f"sha256:{digest.hexdigest()}"


def _required_text(value: Any, name: str) -> str:
    """Return one non-empty textual provenance field."""
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{name} is required")
    return value


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    """Normalize a sequence of non-empty strings without splitting text."""
    if isinstance(value, (str, bytes)):
        raise ManifestValidationError(f"{name} must be a sequence of strings")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ManifestValidationError(
            f"{name} must be a sequence of strings"
        ) from exc
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ManifestValidationError(
            f"{name} must contain only non-empty strings"
        )
    return items


@dataclass(frozen=True)
class CoreExperimentManifest:
    """Reproducibility envelope for a graph-dynamics research artifact."""

    claim_id: str
    git_sha: str
    versions: Mapping[str, str]
    graph_construction: str
    capacity_specification: str
    solver: str
    result_status: ClaimStatus
    seed: int | None = None
    timestep: float | None = None
    operator_sequence: tuple[str, ...] = ()
    telemetry: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    source_dirty: bool | None = None
    dirty_source_hash: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "claim_id",
            "git_sha",
            "graph_construction",
            "capacity_specification",
            "solver",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", self.git_sha):
            raise ManifestValidationError(
                "git_sha must contain 7 to 64 hexadecimal digits"
            )
        if not isinstance(self.versions, Mapping) or not self.versions:
            raise ManifestValidationError("versions must be non-empty")
        normalized_versions: dict[str, str] = {}
        for name, version in self.versions.items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(version, str)
                or not version.strip()
            ):
                raise ManifestValidationError(
                    "versions must map non-empty strings to non-empty strings"
                )
            normalized_versions[name] = version
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, Integral)
        ):
            raise ManifestValidationError("seed must be an integer or None")
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))
        if self.source_dirty is not None and not isinstance(
            self.source_dirty, bool
        ):
            raise ManifestValidationError("source_dirty must be boolean or None")
        if self.dirty_source_hash is not None:
            dirty_hash = _required_text(
                self.dirty_source_hash, "dirty_source_hash"
            )
            if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", dirty_hash):
                raise ManifestValidationError(
                    "dirty_source_hash must be a sha256 digest"
                )
            object.__setattr__(self, "dirty_source_hash", dirty_hash)
        if self.timestep is not None:
            if isinstance(self.timestep, bool) or not isinstance(self.timestep, Real):
                raise ManifestValidationError(
                    "timestep must be finite and positive"
                )
            try:
                timestep = float(self.timestep)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ManifestValidationError(
                    "timestep must be finite and positive"
                ) from exc
            if not math.isfinite(timestep) or timestep <= 0.0:
                raise ManifestValidationError(
                    "timestep must be finite and positive"
                )
            object.__setattr__(self, "timestep", timestep)
        try:
            status = ClaimStatus(self.result_status)
        except (TypeError, ValueError) as exc:
            raise ManifestValidationError("result_status must be canonical") from exc
        object.__setattr__(self, "result_status", status)
        object.__setattr__(
            self, "versions", MappingProxyType(normalized_versions)
        )
        for name in ("operator_sequence", "telemetry", "controls", "artifacts"):
            object.__setattr__(
                self, name, _string_sequence(getattr(self, name), name)
            )

    def validate_for_admission(self) -> None:
        """Require the evidence needed to reproduce a public result."""
        if "python" not in self.versions:
            raise ManifestValidationError("python version is required")
        if not self.telemetry:
            raise ManifestValidationError("at least one telemetry channel is required")
        if not self.controls:
            raise ManifestValidationError("at least one control is required")
        if not self.artifacts:
            raise ManifestValidationError("at least one artifact is required")
        if not isinstance(self.source_dirty, bool):
            raise ManifestValidationError(
                "source_dirty must be explicitly declared for admission"
            )
        if self.source_dirty and not self.dirty_source_hash:
            raise ManifestValidationError(
                "dirty_source_hash is required when source_dirty is true"
            )
        if not self.source_dirty and self.dirty_source_hash is not None:
            raise ManifestValidationError(
                "dirty_source_hash requires source_dirty=true"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "claim_id": self.claim_id,
            "git_sha": self.git_sha,
            "versions": dict(self.versions),
            "graph_construction": self.graph_construction,
            "capacity_specification": self.capacity_specification,
            "solver": self.solver,
            "result_status": self.result_status.value,
            "seed": self.seed,
            "timestep": self.timestep,
            "operator_sequence": list(self.operator_sequence),
            "telemetry": list(self.telemetry),
            "controls": list(self.controls),
            "artifacts": list(self.artifacts),
            "source_dirty": self.source_dirty,
            "dirty_source_hash": self.dirty_source_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoreExperimentManifest":
        """Reconstruct a manifest from :meth:`to_dict` output."""
        return cls(
            claim_id=data["claim_id"],
            git_sha=data["git_sha"],
            versions=data["versions"],
            graph_construction=data["graph_construction"],
            capacity_specification=data["capacity_specification"],
            solver=data["solver"],
            result_status=data["result_status"],
            seed=data.get("seed"),
            timestep=data.get("timestep"),
            operator_sequence=data.get("operator_sequence", ()),
            telemetry=data.get("telemetry", ()),
            controls=data.get("controls", ()),
            artifacts=data.get("artifacts", ()),
            source_dirty=data.get("source_dirty"),
            dirty_source_hash=data.get("dirty_source_hash"),
        )
