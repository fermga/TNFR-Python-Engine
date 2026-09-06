r"""Reproducibility manifest for a TNFR research experiment.

A manifest records everything needed to reproduce and correctly classify a
result: the claim it supports, the code revision and seed, dependency versions,
the operator sequence, whether known factors were used, the input-size model (so
complexity is stated in input *bits*, not node count), the controls run and the
stored artifacts.  A benchmark that omits ``uses_known_factors`` or a complexity
model is not admissible.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ManifestValidationError",
    "ExperimentManifest",
    "input_bit_length",
]


class ManifestValidationError(ValueError):
    """Raised when a manifest is missing a required field."""


def input_bit_length(n: int) -> int:
    """Input size in bits ``L = floor(log2 n) + 1``.

    ``poly(n)`` is exponential in this ``L``; stating cost in ``n`` (node count)
    would hide that a residue network on ``n`` nodes is exponential in ``log n``.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    return int(n).bit_length()


@dataclass(frozen=True)
class ExperimentManifest:
    """A complete, reproducible description of one research experiment."""

    claim_id: str
    git_sha: str
    versions: dict[str, str]
    seed: int | None = None
    operator_sequence: tuple[str, ...] = ()
    uses_known_factors: bool = False
    input_size_model: str = "L = floor(log2 n) + 1 bits"
    input_bits: int | None = None
    controls: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ManifestValidationError("claim_id is required")
        if not self.git_sha:
            raise ManifestValidationError("git_sha is required")
        if not self.versions:
            raise ManifestValidationError("versions must be non-empty")
        if not self.input_size_model:
            raise ManifestValidationError("input_size_model is required")
        object.__setattr__(self, "versions", dict(self.versions))
        object.__setattr__(
            self, "operator_sequence", tuple(self.operator_sequence)
        )
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "git_sha": self.git_sha,
            "versions": dict(self.versions),
            "seed": self.seed,
            "operator_sequence": list(self.operator_sequence),
            "uses_known_factors": self.uses_known_factors,
            "input_size_model": self.input_size_model,
            "input_bits": self.input_bits,
            "controls": list(self.controls),
            "artifacts": list(self.artifacts),
        }

    def validate_for_admission(self) -> None:
        """Validate the stricter provenance contract for public results."""
        if not isinstance(self.uses_known_factors, bool):
            raise ManifestValidationError("uses_known_factors must be boolean")
        if self.input_bits is not None and self.input_bits < 1:
            raise ManifestValidationError("input_bits must be positive")
        if "python" not in self.versions:
            raise ManifestValidationError("python version is required")
        if self.input_size_model != "L = floor(log2 n) + 1 bits":
            raise ManifestValidationError(
                "input_size_model must state the positive-integer bit convention"
            )
        if not self.controls:
            raise ManifestValidationError("at least one control is required")
        if not self.artifacts:
            raise ManifestValidationError("at least one artifact is required")

    @classmethod
    def from_dict(
        cls, data: dict, *, strict: bool = False
    ) -> "ExperimentManifest":
        if strict and "uses_known_factors" not in data:
            raise ManifestValidationError("uses_known_factors must be explicit")
        if strict and not isinstance(data.get("uses_known_factors"), bool):
            raise ManifestValidationError("uses_known_factors must be boolean")
        return cls(
            claim_id=data["claim_id"],
            git_sha=data["git_sha"],
            versions=dict(data["versions"]),
            seed=data.get("seed"),
            operator_sequence=tuple(data.get("operator_sequence", ())),
            uses_known_factors=bool(data.get("uses_known_factors", False)),
            input_size_model=data.get(
                "input_size_model", "L = floor(log2 n) + 1 bits"
            ),
            input_bits=data.get("input_bits"),
            controls=tuple(data.get("controls", ())),
            artifacts=tuple(data.get("artifacts", ())),
        )
