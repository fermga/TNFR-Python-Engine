r"""TNFR research infrastructure: claims, manifests, certificates, circularity.

Lightweight, non-bureaucratic scaffolding so every research result records its
epistemic status, a reproducibility manifest, numerical certificates and a
circularity audit — the C5 consolidation of the 2026-09-04 handoff programme.
Storing this outside ``AGENTS.md`` keeps the synthesized canon free of session
history while making each experiment reproducible and honestly classified.
"""

from __future__ import annotations

from .certificates import NumericalCertificate, certify_within_tolerance
from .circularity import CircularityAudit, CircularityVerdict, classify
from .claims import (
    Claim,
    ClaimRegistry,
    ClaimStatus,
    ClaimTransitionError,
    is_valid_transition,
    validate_transition,
)
from .manifests import (
    ExperimentManifest,
    ManifestValidationError,
    input_bit_length,
)

__all__ = [
    "Claim",
    "ClaimRegistry",
    "ClaimStatus",
    "ClaimTransitionError",
    "CircularityAudit",
    "CircularityVerdict",
    "ExperimentManifest",
    "ManifestValidationError",
    "NumericalCertificate",
    "certify_within_tolerance",
    "classify",
    "input_bit_length",
    "is_valid_transition",
    "validate_transition",
]
