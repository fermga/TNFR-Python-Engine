"""Public contract checks for the new temporal and factor-domain APIs."""

from __future__ import annotations

import inspect
from dataclasses import fields

import tnfr.operators as operators
import tnfr.physics as physics
import tnfr.sdk as sdk
from tnfr.operators import (
    event_remesh_runtime,
    event_remesh_sequence,
    event_runtime,
    event_timing,
    factor_contracts,
)
from tnfr.operators.nodal_equation import compute_d2epi_dt2
from tnfr.physics import (
    event_remesh_refinement,
    mutation_trigger,
    remesh_history_stability,
)
from tnfr.sdk import simple


def test_event_timing_exports_are_identical_at_operators_facade() -> None:
    public_names = set(event_timing.__all__)

    assert public_names <= set(operators.__all__)
    for name in public_names:
        assert getattr(operators, name) is getattr(event_timing, name)


def test_event_runtime_exports_are_identical_at_operators_facade() -> None:
    public_names = set(event_runtime.__all__)

    assert public_names <= set(operators.__all__)
    for name in public_names:
        assert getattr(operators, name) is getattr(event_runtime, name)


def test_event_remesh_exports_are_identical_at_operators_facade() -> None:
    for module in (event_remesh_runtime, event_remesh_sequence):
        public_names = set(module.__all__)

        assert public_names <= set(operators.__all__)
        for name in public_names:
            assert getattr(operators, name) is getattr(module, name)


def test_factor_contract_module_is_centralized_at_operators_facade() -> None:
    public_names = set(factor_contracts.__all__)

    assert public_names <= set(operators.__all__)
    for name in public_names:
        assert getattr(operators, name) is getattr(factor_contracts, name)


def test_mutation_certificate_exports_are_available_from_physics_facade() -> None:
    names = {
        "MutationTriggerCertificate",
        "MutationTriggerEvidence",
        "MutationTriggerInputError",
        "certify_mutation_trigger",
    }

    assert names <= set(physics.__all__)
    for name in names:
        assert getattr(physics, name) is getattr(mutation_trigger, name)


def test_new_remesh_physics_modules_are_centralized_at_physics_facade() -> None:
    for module in (event_remesh_refinement, remesh_history_stability):
        public_names = set(module.__all__)

        assert public_names <= set(physics.__all__)
        for name in public_names:
            assert getattr(physics, name) is getattr(module, name)


def test_sdk_nodal_state_report_exposes_mutation_evidence_fields() -> None:
    evidence_fields = {
        "observed_depi_dt",
        "predicted_crossed",
        "observed_crossed",
        "evidence_available",
        "evidence_valid",
        "source",
        "time_basis",
        "physical_time_resolved",
        "current_endpoint_matches_state",
        "reason",
        "rate_gap",
        "mutation_threshold_satisfied",
    }

    assert sdk.NodalStateReport is simple.NodalStateReport
    assert evidence_fields <= {field.name for field in fields(sdk.NodalStateReport)}


def test_structural_acceleration_store_flag_is_keyword_only() -> None:
    store = inspect.signature(compute_d2epi_dt2).parameters["store"]

    assert store.kind is inspect.Parameter.KEYWORD_ONLY
    assert store.default is True


def test_mutation_certificate_inputs_remain_keyword_only() -> None:
    parameters = inspect.signature(
        mutation_trigger.certify_mutation_trigger
    ).parameters

    assert parameters
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
