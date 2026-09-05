r"""Tests for the REMESH contract audit (R4b, N09).

The p-adic tower transport is a morphism (N08); REMESH additionally requires a
temporal ``EPI(t) ← EPI(t-τ)`` echo. The four-condition audit rejects the static
lift (no echo) and accepts the genuine temporal recurrence.
"""

from __future__ import annotations

import numpy as np

from tnfr.mathematics.padic_tower import RemeshContractAudit
from tnfr.mathematics import remesh_audit as ra
from tnfr.mathematics.remesh_audit import (
    audit_remesh_candidate,
    remesh_campaign,
    remesh_coefficients,
    remesh_recurrence,
    remesh_recurrence_update,
    scale_projection_update,
    temporal_echo_residual,
)


# --------------------------------------------------------------------------- #
# The temporal recurrence
# --------------------------------------------------------------------------- #
def test_remesh_coefficients_are_a_partition_of_unity():
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        c0, cl, cg = remesh_coefficients(alpha)
        assert c0 + cl + cg == 1.0            # convex combination
        assert c0 >= 0 and cl >= 0 and cg >= 0


def test_recurrence_fixes_a_coherent_state():
    x = np.array([1.0, -1.0, 0.5, 0.5])
    assert np.allclose(remesh_recurrence(x, x, x, alpha=0.5), x)


# --------------------------------------------------------------------------- #
# Temporal echo — the discriminating condition
# --------------------------------------------------------------------------- #
def test_temporal_echo_present_in_recurrence_absent_in_lift():
    n = 9
    now, pl, pg = np.ones(n), np.ones(n), np.ones(n)
    assert temporal_echo_residual(
        remesh_recurrence_update(alpha=0.5), now, pl, pg) > 1e-6
    assert temporal_echo_residual(
        scale_projection_update(3, 1), now, pl, pg) < 1e-9


# --------------------------------------------------------------------------- #
# Required: the audit rejects the static lift, accepts genuine REMESH
# --------------------------------------------------------------------------- #
def test_remesh_rejects_missing_temporal_echo():
    audit = audit_remesh_candidate(scale_projection_update(3, 1), network_size=9)
    assert not audit.epi_recursion_verified      # the static lift has no echo
    assert audit.network_scale_verified          # but it IS network-scale,
    assert audit.identity_preserved_verified     # identity-preserving,
    assert audit.u5_multiscale_verified          # and U5-coherent
    assert not audit.realizes_remesh             # => a morphism, not REMESH


def test_remesh_accepts_only_all_four_conditions():
    audit = audit_remesh_candidate(
        remesh_recurrence_update(alpha=0.5), network_size=9)
    assert audit.epi_recursion_verified
    assert audit.network_scale_verified
    assert audit.identity_preserved_verified
    assert audit.u5_multiscale_verified
    assert audit.realizes_remesh                 # all four => REMESH


def test_remesh_gate_requires_every_condition():
    # dropping any single condition denies REMESH (the AND gate)
    for missing in ("epi_recursion_verified", "network_scale_verified",
                    "identity_preserved_verified", "u5_multiscale_verified"):
        fields = {k: True for k in (
            "epi_recursion_verified", "network_scale_verified",
            "identity_preserved_verified", "u5_multiscale_verified")}
        fields[missing] = False
        assert not RemeshContractAudit(**fields).realizes_remesh


def test_remesh_rejects_partial_u5_certificate():
    # a candidate with a temporal echo but that amplifies (breaks identity/U5)
    # is not REMESH
    def amplifying(now, past_local, past_global):
        return 2.0 * (np.asarray(now, float) + np.asarray(past_local, float)
                      + np.asarray(past_global, float))
    audit = audit_remesh_candidate(amplifying, network_size=9)
    assert audit.epi_recursion_verified          # it does echo
    assert not audit.identity_preserved_verified  # but it amplifies
    assert not audit.realizes_remesh


# --------------------------------------------------------------------------- #
# The campaign
# --------------------------------------------------------------------------- #
def test_campaign_tower_is_not_remesh_but_audit_discriminates():
    c = remesh_campaign(p=3, e=1, alpha=0.5)
    assert c.tower_realizes_remesh is False      # honest close for the tower
    assert c.audit_discriminates is True         # the gate is not vacuous


def test_module_exports_complete():
    expected = {
        "remesh_coefficients", "remesh_recurrence", "remesh_recurrence_update",
        "scale_projection_update", "temporal_echo_residual",
        "audit_remesh_candidate", "RemeshCampaign", "remesh_campaign",
    }
    assert expected <= set(ra.__all__)
