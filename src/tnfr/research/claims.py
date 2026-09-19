r"""Claim-status ledger for TNFR research results.

Every research result carries an explicit epistemic status so that a measured
regularity is never silently promoted to a theorem.  The canonical statuses are
``PROVED``, ``DERIVED``, ``MEASURED``, ``CONJECTURAL``, ``NEGATIVE`` and
``SUPERSEDED``.  Status may only *strengthen*
along ``CONJECTURAL → MEASURED → DERIVED → PROVED``; a claim can always be
corrected to ``NEGATIVE`` (falsified) or ``SUPERSEDED`` (replaced by a stronger
result), but it may never silently weaken (e.g. ``PROVED → CONJECTURAL``) without
passing through ``SUPERSEDED``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

__all__ = [
    "ClaimStatus",
    "ClaimTransitionError",
    "Claim",
    "ClaimRegistry",
    "is_valid_transition",
    "validate_transition",
]


class ClaimStatus(str, Enum):
    """The six canonical epistemic statuses of a TNFR research claim."""

    PROVED = "proved"
    DERIVED = "derived"
    MEASURED = "measured"
    CONJECTURAL = "conjectural"
    NEGATIVE = "negative"
    SUPERSEDED = "superseded"


# Forward strengthening ladder (weaker -> stronger).
_STRENGTHEN: dict[ClaimStatus, frozenset[ClaimStatus]] = {
    ClaimStatus.CONJECTURAL: frozenset(
        {ClaimStatus.MEASURED, ClaimStatus.DERIVED, ClaimStatus.PROVED}
    ),
    ClaimStatus.MEASURED: frozenset({ClaimStatus.DERIVED, ClaimStatus.PROVED}),
    ClaimStatus.DERIVED: frozenset({ClaimStatus.PROVED}),
    ClaimStatus.PROVED: frozenset(),
    ClaimStatus.NEGATIVE: frozenset(),
    ClaimStatus.SUPERSEDED: frozenset(),
}

# Correction sinks reachable from any non-terminal status.
_CORRECTION = frozenset({ClaimStatus.NEGATIVE, ClaimStatus.SUPERSEDED})
_TERMINAL = frozenset({ClaimStatus.NEGATIVE, ClaimStatus.SUPERSEDED})


class ClaimTransitionError(ValueError):
    """Raised when a claim-status transition is not canonical."""


def is_valid_transition(old, new) -> bool:
    """Whether ``old → new`` is an allowed claim-status transition."""
    old = ClaimStatus(old)
    new = ClaimStatus(new)
    if old == new:
        return True
    if old in _TERMINAL:
        return False
    if new in _CORRECTION:
        return True
    return new in _STRENGTHEN[old]


def validate_transition(old, new) -> None:
    """Raise :class:`ClaimTransitionError` if ``old → new`` is not canonical."""
    if not is_valid_transition(old, new):
        raise ClaimTransitionError(
            f"invalid claim transition {ClaimStatus(old).value} -> "
            f"{ClaimStatus(new).value}; strengthen along CONJECTURAL -> "
            f"MEASURED -> DERIVED -> PROVED, or correct to NEGATIVE/SUPERSEDED."
        )


@dataclass(frozen=True)
class Claim:
    """A research claim with an explicit epistemic status and references."""

    claim_id: str
    statement: str
    status: ClaimStatus
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id must be non-empty")
        if not self.statement:
            raise ValueError("statement must be non-empty")
        object.__setattr__(self, "status", ClaimStatus(self.status))
        object.__setattr__(self, "references", tuple(self.references))

    def with_status(self, new_status) -> "Claim":
        """Return a copy at ``new_status`` after validating the transition."""
        validate_transition(self.status, new_status)
        return replace(self, status=ClaimStatus(new_status))

    def promote(
        self,
        new_status,
        *,
        justification: str,
        proof_references: tuple[str, ...] = (),
    ) -> "Claim":
        """Promote a claim only with explicit justification and proof sources."""
        target = ClaimStatus(new_status)
        validate_transition(self.status, target)
        if target not in {ClaimStatus.DERIVED, ClaimStatus.PROVED}:
            raise ClaimTransitionError(
                "promote() is reserved for DERIVED or PROVED transitions"
            )
        if not justification.strip() or not proof_references:
            raise ClaimTransitionError(
                "claim promotion requires justification and proof references"
            )
        merged = tuple(dict.fromkeys((*self.references, *proof_references)))
        return replace(self, status=target, references=merged)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "status": self.status.value,
            "references": list(self.references),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Claim":
        return cls(
            claim_id=data["claim_id"],
            statement=data["statement"],
            status=ClaimStatus(data["status"]),
            references=tuple(data.get("references", ())),
        )


@dataclass
class ClaimRegistry:
    """A mutable collection of claims keyed by ``claim_id``."""

    claims: dict[str, Claim] = field(default_factory=dict)

    def add(self, claim: Claim) -> None:
        if claim.claim_id in self.claims:
            raise ValueError(f"claim {claim.claim_id!r} already registered")
        self.claims[claim.claim_id] = claim

    def get(self, claim_id: str) -> Claim:
        return self.claims[claim_id]

    def transition(self, claim_id: str, new_status) -> Claim:
        updated = self.claims[claim_id].with_status(new_status)
        self.claims[claim_id] = updated
        return updated

    def to_dict(self) -> dict:
        return {cid: c.to_dict() for cid, c in self.claims.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "ClaimRegistry":
        return cls({cid: Claim.from_dict(d) for cid, d in data.items()})
