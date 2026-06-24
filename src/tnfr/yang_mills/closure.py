r"""Y5 closure / obstruction classification for TNFR–Yang–Mills.

Y5 does not add a new gauge dynamics.  It classifies what Y1–Y4 establish and
what remains obstructed.  The key distinction is between:

* a finite TNFR-internal structural gauge diagnostic surface (currently U(1));
* a Clay-strength Yang–Mills mass-gap claim, which would require non-Abelian
  derivability and a continuum / thermodynamic lower-bound theorem.

The expected current programme verdict is ``BRANCH_B_OBSTRUCTION_CLASSIFIED``:
finite TNFR U(1) diagnostics exist, but Clay-strength closure requires a new
canonical non-Abelian derivation from the nodal equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .derivability import NonAbelianDerivabilityReport, audit_nonabelian_derivability
from .scaling import FiniteScalingReport, run_finite_scaling_study


@dataclass(frozen=True)
class YangMillsClosureReport:
    """Y5 classification report for the TNFR–Yang–Mills programme."""

    finite_tnfr_branch: str
    clay_strength_branch: str
    verdict: str
    clay_problem_resolved: bool
    tnfr_internal_result: str
    primary_obstruction: str
    established_results: tuple[str, ...]
    open_requirements: tuple[str, ...]
    evidence: dict[str, Any]
    metadata: dict[str, Any]


def classify_yang_mills_closure(
    *,
    derivability_report: NonAbelianDerivabilityReport | None = None,
    scaling_report: FiniteScalingReport | None = None,
    derivability_kwargs: Mapping[str, Any] | None = None,
    scaling_kwargs: Mapping[str, Any] | None = None,
) -> YangMillsClosureReport:
    """Classify the Y1–Y4 state into a Y5 closure / obstruction verdict.

    Parameters
    ----------
    derivability_report : NonAbelianDerivabilityReport, optional
        Existing Y3 report.  If omitted, Y3 is run with
        ``derivability_kwargs``.
    scaling_report : FiniteScalingReport, optional
        Existing Y4 report.  If omitted, Y4 is run with ``scaling_kwargs``.
    derivability_kwargs : mapping, optional
        Keyword arguments forwarded to ``audit_nonabelian_derivability``.
    scaling_kwargs : mapping, optional
        Keyword arguments forwarded to ``run_finite_scaling_study``.

    Returns
    -------
    YangMillsClosureReport
        Conservative programme classification.  ``clay_problem_resolved`` is
        always false unless future reports provide both non-Abelian derivation
        and continuum lower-bound evidence; no current public API provides the
        latter.
    """
    if derivability_report is None:
        derivability_report = audit_nonabelian_derivability(
            **dict(derivability_kwargs or {})
        )
    if scaling_report is None:
        scaling_report = run_finite_scaling_study(**dict(scaling_kwargs or {}))

    finite_branch = _classify_finite_branch(scaling_report)
    clay_branch = _classify_clay_branch(derivability_report, scaling_report)
    verdict = _classify_programme_verdict(clay_branch, scaling_report)
    established = _established_results(
        finite_branch,
        derivability_report,
        scaling_report,
    )
    open_requirements = _open_requirements(
        derivability_report,
        scaling_report,
    )
    primary_obstruction = _primary_obstruction(
        derivability_report,
        scaling_report,
    )
    tnfr_internal_result = _tnfr_internal_result(finite_branch, clay_branch)
    clay_problem_resolved = False

    evidence = {
        "derivability_verdict": derivability_report.verdict,
        "nonabelian_derived": derivability_report.nonabelian_derived,
        "canonical_gauge_group": derivability_report.canonical_gauge_group,
        "scaling_verdict": scaling_report.verdict,
        "finite_min_gap": scaling_report.summary.get("min_gap"),
        "finite_mean_gap": scaling_report.summary.get("mean_gap"),
        "finite_group_count": scaling_report.summary.get("n_groups"),
        "all_self_adjoint": scaling_report.summary.get("all_self_adjoint"),
        "all_gauge_invariant": scaling_report.summary.get("all_gauge_invariant"),
    }
    metadata = {
        "scope": "Y5_closure_obstruction_not_clay_proof",
        "finite_branch_definition": (
            "A = finite U(1) diagnostic surface supported; "
            "C = sampled finite collapse/failure"
        ),
        "clay_branch_definition": (
            "B = Clay-strength closure requires new canonical non-Abelian "
            "derivation and continuum lower-bound theorem"
        ),
        "continuum_lower_bound_available": False,
    }

    return YangMillsClosureReport(
        finite_tnfr_branch=finite_branch,
        clay_strength_branch=clay_branch,
        verdict=verdict,
        clay_problem_resolved=clay_problem_resolved,
        tnfr_internal_result=tnfr_internal_result,
        primary_obstruction=primary_obstruction,
        established_results=established,
        open_requirements=open_requirements,
        evidence=evidence,
        metadata=metadata,
    )


def _classify_finite_branch(scaling_report: FiniteScalingReport) -> str:
    if scaling_report.verdict == "FINITE_SCALING_EVIDENCE":
        return "A_FINITE_U1_DIAGNOSTIC_SURFACE"
    if scaling_report.verdict == "GAP_COLLAPSE_OBSERVED":
        return "C_FINITE_SAMPLE_GAP_COLLAPSE"
    if scaling_report.verdict in {
        "SCALING_FAILED_NON_SELF_ADJOINT",
        "SCALING_FAILED_GAUGE_VARIANCE",
    }:
        return "C_FINITE_DIAGNOSTIC_FAILURE"
    return "C_FINITE_STATUS_INDETERMINATE"


def _classify_clay_branch(
    derivability_report: NonAbelianDerivabilityReport,
    scaling_report: FiniteScalingReport,
) -> str:
    if not derivability_report.nonabelian_derived:
        return "B_REQUIRES_NEW_CANONICAL_NONABELIAN_DERIVATION"
    if scaling_report.verdict != "FINITE_SCALING_EVIDENCE":
        return "B_REQUIRES_STABLE_FINITE_GAP_SURFACE"
    return "B_REQUIRES_CONTINUUM_LOWER_BOUND_THEOREM"


def _classify_programme_verdict(
    clay_branch: str,
    scaling_report: FiniteScalingReport,
) -> str:
    if scaling_report.verdict in {
        "SCALING_FAILED_NON_SELF_ADJOINT",
        "SCALING_FAILED_GAUGE_VARIANCE",
    }:
        return "DIAGNOSTIC_FAILURE_REQUIRES_REPAIR"
    if clay_branch.startswith("B_REQUIRES_NEW_CANONICAL_NONABELIAN"):
        return "BRANCH_B_OBSTRUCTION_CLASSIFIED"
    if clay_branch == "B_REQUIRES_STABLE_FINITE_GAP_SURFACE":
        return "BRANCH_B_FINITE_GAP_SURFACE_UNSTABLE"
    return "BRANCH_B_CONTINUUM_GAP_OPEN"


def _established_results(
    finite_branch: str,
    derivability_report: NonAbelianDerivabilityReport,
    scaling_report: FiniteScalingReport,
) -> tuple[str, ...]:
    results = [
        "Y1 finite self-adjoint structural gauge operator constructed",
        "Y2 U6 finite confinement sweep surface constructed",
        "Y3 non-Abelian derivability audit executed",
        "Y4 finite graph-size scaling diagnostic executed",
    ]
    if finite_branch == "A_FINITE_U1_DIAGNOSTIC_SURFACE":
        results.append("Finite U(1) structural gap surface is supported")
    if derivability_report.verdict == "OPEN_DERIVABILITY_GAP":
        results.append("Canonical gauge sector remains U(1) at this stage")
    if scaling_report.verdict == "GAP_COLLAPSE_OBSERVED":
        results.append("At least one sampled finite family exhibits gap collapse")
    return tuple(results)


def _open_requirements(
    derivability_report: NonAbelianDerivabilityReport,
    scaling_report: FiniteScalingReport,
) -> tuple[str, ...]:
    requirements = []
    if not derivability_report.nonabelian_derived:
        requirements.extend(
            [
                "derive_tnfr_native_multiplet",
                "derive_component_mixing_connection",
                "derive_noncommuting_generator_algebra",
                "prove_u1_u6_compatibility_for_nonabelian_sector",
            ]
        )
    if scaling_report.verdict != "FINITE_SCALING_EVIDENCE":
        requirements.append("stabilize_positive_finite_gap_surface")
    requirements.append("prove_continuum_thermodynamic_liminf_gap_bound")
    return tuple(requirements)


def _primary_obstruction(
    derivability_report: NonAbelianDerivabilityReport,
    scaling_report: FiniteScalingReport,
) -> str:
    if not derivability_report.nonabelian_derived:
        return (
            "YMG-4 remains open: no TNFR-native non-Abelian connection or "
            "non-commuting generator algebra is derived from the nodal "
            "equation and canonical grammar."
        )
    if scaling_report.verdict != "FINITE_SCALING_EVIDENCE":
        return "Y4 finite scaling surface is not stably positive in samples."
    return (
        "YMG-5 remains open: no continuum / thermodynamic lower-bound theorem "
        "has been derived."
    )


def _tnfr_internal_result(finite_branch: str, clay_branch: str) -> str:
    if finite_branch == "A_FINITE_U1_DIAGNOSTIC_SURFACE":
        return (
            "TNFR currently supports a finite, gauge-invariant U(1) "
            "structural "
            "gap diagnostic surface; Clay-strength closure is blocked by "
            f"{clay_branch}."
        )
    return (
        "TNFR currently has a finite diagnostic surface, but sampled gap "
        "support is not stable enough for Branch A; Clay branch = "
        f"{clay_branch}."
    )
