r"""Circularity audit for TNFR arithmetic experiments.

An arithmetic experiment that secretly uses the answer it claims to discover is
circular.  This module records the five audit questions and classifies the
result:

* a **structural** study uses none of them;
* a **descriptive** study uses ground truth only for scoring or feature
  annotation — valid as a structural study, but never presentable as a discovery
  algorithm;
* a **circular** study needs the answer to even build the object (or calls
  factorization while constructing features).

An affirmative answer does not invalidate a structural study; it changes its
classification and forbids presenting it as a discovery algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["CircularityVerdict", "CircularityAudit", "classify"]


class CircularityVerdict(str, Enum):
    STRUCTURAL = "structural"
    DESCRIPTIVE = "descriptive"
    CIRCULAR = "circular"


@dataclass(frozen=True)
class CircularityAudit:
    """The five circularity questions for an arithmetic experiment."""

    uses_factorization_in_features: bool = False
    uses_gcd_against_candidate: bool = False
    uses_phi_omega_tau_sigma: bool = False
    factors_used_only_for_scoring: bool = False
    graph_construction_requires_answer: bool = False

    @property
    def verdict(self) -> "CircularityVerdict":
        return classify(self)

    @property
    def permits_discovery_claim(self) -> bool:
        """Only a fully structural experiment may be called a discovery."""
        return self.verdict is CircularityVerdict.STRUCTURAL

    def to_dict(self) -> dict:
        return {
            "uses_factorization_in_features": (
                self.uses_factorization_in_features
            ),
            "uses_gcd_against_candidate": self.uses_gcd_against_candidate,
            "uses_phi_omega_tau_sigma": self.uses_phi_omega_tau_sigma,
            "factors_used_only_for_scoring": (
                self.factors_used_only_for_scoring
            ),
            "graph_construction_requires_answer": (
                self.graph_construction_requires_answer
            ),
            "verdict": self.verdict.value,
        }


def classify(audit: "CircularityAudit") -> CircularityVerdict:
    """Classify an experiment as structural, descriptive or circular."""
    if (
        audit.graph_construction_requires_answer
        or audit.uses_factorization_in_features
    ):
        return CircularityVerdict.CIRCULAR
    if (
        audit.uses_gcd_against_candidate
        or audit.uses_phi_omega_tau_sigma
        or audit.factors_used_only_for_scoring
    ):
        return CircularityVerdict.DESCRIPTIVE
    return CircularityVerdict.STRUCTURAL
