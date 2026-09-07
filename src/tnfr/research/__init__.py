r"""TNFR research infrastructure: claims, manifests, certificates, circularity.

The utilities record epistemic status, reproducibility manifests, numerical
certificates, and circularity audits without placing experiment history in the
synthesized canonical guidance.
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
from .core_manifests import CoreExperimentManifest, current_git_source_provenance
from .evidence_sidecar import EvidenceAdmissionError, EvidenceSidecar
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
    "CoreExperimentManifest",
    "current_git_source_provenance",
    "ExperimentManifest",
    "EvidenceAdmissionError",
    "EvidenceSidecar",
    "ManifestValidationError",
    "NumericalCertificate",
    "certify_within_tolerance",
    "classify",
    "input_bit_length",
    "is_valid_transition",
    "validate_transition",
]
