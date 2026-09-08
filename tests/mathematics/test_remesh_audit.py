r"""Tests for scoped REMESH and declared U5 evidence (R4b, N09)."""

from __future__ import annotations

import math
import sys

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_DEPI, ALIAS_DNFR
from tnfr.mathematics import remesh_audit as ra
from tnfr.mathematics.padic_tower import RemeshContractAudit
from tnfr.mathematics.remesh_audit import (
    RemeshCandidateAudit,
    RemeshU5Evidence,
    audit_remesh_candidate,
    field_uniformity_score,
    remesh_campaign,
    remesh_coefficients,
    remesh_recurrence,
    remesh_recurrence_update,
    scale_projection_update,
    temporal_echo_residual,
)


def _declared_u5_evidence(*, alpha: float) -> RemeshU5Evidence:
    graph = nx.Graph()
    graph.add_node(
        "parent",
        **{
            ALIAS_DNFR[0]: 1.0,
            ALIAS_DEPI[0]: 0.0,
            "sub_nodes": ["child-a", "child-b"],
        },
    )
    for child in ("child-a", "child-b"):
        graph.add_node(
            child,
            **{
                ALIAS_DNFR[0]: 0.0,
                ALIAS_DEPI[0]: 0.0,
                "parent_node": "parent",
            },
        )
        graph.add_edge("parent", child)
    return RemeshU5Evidence(
        post_update_graph=graph,
        parent="parent",
        alpha=alpha,
    )


def test_remesh_coefficients_are_a_partition_of_unity():
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        current, local, global_ = remesh_coefficients(alpha)
        assert current + local + global_ == 1.0
        assert current >= 0.0
        assert local >= 0.0
        assert global_ >= 0.0


@pytest.mark.parametrize("alpha", [True, -0.1, 1.1, np.nan, np.inf])
def test_remesh_coefficients_reject_values_outside_convex_domain(alpha):
    with pytest.raises((TypeError, ValueError)):
        remesh_coefficients(alpha)


def test_recurrence_fixes_a_shared_state():
    field = np.array([1.0, -1.0, 0.5, 0.5])

    result = remesh_recurrence(field, field, field, alpha=0.5)

    np.testing.assert_allclose(result, field)


def test_temporal_echo_present_in_recurrence_and_absent_in_lift():
    size = 9
    now = np.ones(size)
    past_local = np.ones(size)
    past_global = np.ones(size)

    assert temporal_echo_residual(
        remesh_recurrence_update(alpha=0.5),
        now,
        past_local,
        past_global,
    ) > 1e-6
    assert temporal_echo_residual(
        scale_projection_update(3, 1),
        now,
        past_local,
        past_global,
    ) < 1e-9


def test_field_uniformity_is_finite_safe_and_explicitly_noncanonical():
    largest = sys.float_info.max

    uniform = field_uniformity_score([largest, largest])
    dispersed = field_uniformity_score([largest, -largest])

    assert uniform == 1.0
    assert math.isfinite(dispersed)
    assert dispersed > 0.0
    assert dispersed < uniform


@pytest.mark.parametrize(
    "values",
    [[], [[1.0]], [True, False], ["1.0"], [0.0, np.nan], [0.0, np.inf]],
)
def test_field_uniformity_rejects_invalid_fields(values):
    with pytest.raises((TypeError, ValueError)):
        field_uniformity_score(values)


def test_static_lift_field_uniformity_does_not_certify_u5():
    audit = audit_remesh_candidate(
        scale_projection_update(3, 1),
        network_size=9,
    )

    assert isinstance(audit, RemeshCandidateAudit)
    assert isinstance(audit, RemeshContractAudit)
    assert audit.network_scale_verified
    assert audit.identity_preserved_verified
    assert audit.field_uniformity_preserved
    assert not audit.u5_evidence_declared
    assert not audit.u5_multiscale_verified
    assert not audit.epi_recursion_verified
    assert not audit.realizes_remesh


def test_temporal_recurrence_remains_unverified_without_u5_hierarchy():
    audit = audit_remesh_candidate(
        remesh_recurrence_update(alpha=0.5),
        network_size=9,
    )

    assert audit.epi_recursion_verified
    assert audit.network_scale_verified
    assert audit.identity_preserved_verified
    assert audit.field_uniformity_preserved
    assert not audit.u5_evidence_declared
    assert not audit.u5_multiscale_verified
    assert not audit.realizes_remesh
    assert audit.to_dict()["field_uniformity_preserved"] is True
    assert audit.to_dict()["u5_evidence_declared"] is False


def test_satisfying_declared_hierarchy_can_complete_scoped_contract_audit():
    audit = audit_remesh_candidate(
        remesh_recurrence_update(alpha=0.5),
        network_size=9,
        u5_evidence=_declared_u5_evidence(alpha=0.2),
    )

    assert audit.u5_evidence_declared
    assert audit.u5_assessment is not None
    assert audit.u5_assessment.parent == "parent"
    assert audit.u5_assessment.satisfies_target
    assert audit.u5_multiscale_verified
    assert audit.realizes_remesh


def test_failing_declared_hierarchy_overrides_preserved_field_uniformity():
    audit = audit_remesh_candidate(
        remesh_recurrence_update(alpha=0.5),
        network_size=9,
        u5_evidence=_declared_u5_evidence(alpha=0.3),
    )

    assert audit.field_uniformity_preserved
    assert audit.u5_evidence_declared
    assert audit.u5_assessment is not None
    assert not audit.u5_assessment.satisfies_target
    assert not audit.u5_multiscale_verified
    assert not audit.realizes_remesh


def test_static_lift_remains_non_remesh_with_passing_u5_evidence():
    audit = audit_remesh_candidate(
        scale_projection_update(3, 1),
        network_size=9,
        u5_evidence=_declared_u5_evidence(alpha=0.2),
    )

    assert audit.u5_multiscale_verified
    assert not audit.epi_recursion_verified
    assert not audit.realizes_remesh


def test_remesh_gate_requires_every_contract_condition():
    fields = (
        "epi_recursion_verified",
        "network_scale_verified",
        "identity_preserved_verified",
        "u5_multiscale_verified",
    )
    for missing in fields:
        values = {field: True for field in fields}
        values[missing] = False
        assert not RemeshContractAudit(**values).realizes_remesh


def test_amplifying_echo_is_not_a_remesh_certificate():
    def amplifying(now, past_local, past_global):
        return 2.0 * (
            np.asarray(now, dtype=float)
            + np.asarray(past_local, dtype=float)
            + np.asarray(past_global, dtype=float)
        )

    audit = audit_remesh_candidate(amplifying, network_size=9)

    assert audit.epi_recursion_verified
    assert not audit.identity_preserved_verified
    assert not audit.u5_multiscale_verified
    assert not audit.realizes_remesh


def test_default_campaign_discriminates_echo_but_not_full_contract():
    campaign = remesh_campaign(p=3, e=1, alpha=0.5)

    assert campaign.tower_realizes_remesh is False
    assert campaign.temporal_echo_discriminates is True
    assert campaign.audit_discriminates is False
    assert not campaign.temporal_recurrence.u5_multiscale_verified


def test_campaign_full_gate_requires_candidate_specific_u5_evidence():
    campaign = remesh_campaign(
        p=3,
        e=1,
        alpha=0.5,
        temporal_u5_evidence=_declared_u5_evidence(alpha=0.2),
    )

    assert campaign.temporal_echo_discriminates is True
    assert campaign.audit_discriminates is True
    assert campaign.temporal_recurrence.realizes_remesh
    assert not campaign.static_lift.u5_evidence_declared
    assert not campaign.static_lift.realizes_remesh


@pytest.mark.parametrize("network_size", [True, 1, 0, -1, 2.5])
def test_candidate_audit_rejects_invalid_network_size(network_size):
    with pytest.raises((TypeError, ValueError), match="network_size"):
        audit_remesh_candidate(
            remesh_recurrence_update(),
            network_size=network_size,
        )


@pytest.mark.parametrize("tol", [True, -1.0, np.nan, np.inf])
def test_candidate_audit_rejects_invalid_uniformity_tolerance(tol):
    with pytest.raises((TypeError, ValueError), match="tol"):
        audit_remesh_candidate(
            remesh_recurrence_update(),
            tol=tol,
        )


def test_module_exports_distinguish_u5_evidence_and_field_uniformity():
    expected = {
        "field_uniformity_score",
        "remesh_coefficients",
        "remesh_recurrence",
        "remesh_recurrence_update",
        "scale_projection_update",
        "temporal_echo_residual",
        "RemeshU5Evidence",
        "RemeshCandidateAudit",
        "audit_remesh_candidate",
        "RemeshCampaign",
        "remesh_campaign",
    }

    assert expected <= set(ra.__all__)
