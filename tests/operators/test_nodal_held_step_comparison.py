"""Optional held-step observations share solver policies without claiming causality."""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.config.defaults_core import CORE_DEFAULTS
from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.dynamics import integrators
from tnfr.dynamics.structural_clip import get_clip_stats
from tnfr.errors import TNFRValueError
from tnfr.operators.definitions import Coherence, Emission
from tnfr.operators.nodal_equation import (
    DEFAULT_NODAL_EQUATION_TOLERANCE,
    NodalEquationViolation,
    validate_nodal_equation,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.self_organization import SelfOrganization


def _graph(*, epi=0.82, capacity=1.0, pressure=0.08):
    graph = nx.empty_graph(1)
    graph.nodes[0].update(
        {
            EPI_PRIMARY: epi,
            VF_PRIMARY: capacity,
            DNFR_PRIMARY: pressure,
            THETA_PRIMARY: 0.0,
        }
    )
    return graph


@pytest.mark.parametrize(
    "scalar_backend,capacity,pressure",
    [
        (False, 1.0, 0.08),
        (True, 1.0, 0.08),
        (False, 0.0, 0.3),
        (False, 1.0, 0.0),
        (False, 1.0, 2**-60),
    ],
)
def test_shared_comparison_matches_real_soft_solver_and_thol_preflight(
    monkeypatch, scalar_backend, capacity, pressure
):
    if scalar_backend:
        monkeypatch.setattr(integrators, "np", None)
    graph = _graph(capacity=capacity, pressure=pressure)
    graph.graph.update(
        CLIP_MODE="soft",
        CLIP_SOFT_K=0.4,
        NODAL_EQUATION_STRICT=True,
        DT_MIN=0.0,  # One declared step, not a composition of projected substeps.
    )
    before = get_attr(graph.nodes[0], ALIAS_EPI)
    dt = 0.5
    integrators.DefaultIntegrator().integrate(
        graph, dt=dt, t=0.0, method="euler", n_jobs=None
    )
    after = get_attr(graph.nodes[0], ALIAS_EPI)
    if capacity * pressure <= 2**-60:
        assert after == before
    else:
        assert after != before + dt * capacity * pressure
    # Gamma installs an immutable mappingproxy during integration; retain its
    # identity and snapshot the mutable scalar node data without pickling it.
    retained = (dict(graph.graph), deepcopy(dict(graph.nodes[0])))
    clip_stats = get_clip_stats().summary()

    assert validate_nodal_equation(graph, 0, before, after, dt, strict=True)
    SelfOrganization()._validate_nodal_proposal(
        graph,
        {"validate_nodal_equation": True, "dt": dt},
        epi_before=before,
        epi_after=after,
        vf=capacity,
        dnfr=pressure,
    )
    assert (dict(graph.graph), dict(graph.nodes[0])) == retained
    assert get_clip_stats().summary() == clip_stats


def test_default_tolerance_and_explicit_override_use_one_owner():
    graph = _graph(epi=0.5, pressure=0.0)
    after = 0.5 + 5e-7

    assert DEFAULT_NODAL_EQUATION_TOLERANCE == CORE_DEFAULTS["NODAL_EQUATION_TOLERANCE"]
    assert not validate_nodal_equation(graph, 0, 0.5, after, 1.0)
    graph.graph["NODAL_EQUATION_TOLERANCE"] = 1e-6
    assert validate_nodal_equation(graph, 0, 0.5, after, 1.0)
    assert not validate_nodal_equation(graph, 0, 0.5, after, 1.0, tolerance=1e-8)


@pytest.mark.parametrize(
    "clip_aware, unit, expected_error",
    [(True, "EPI", 0.0625), (False, "EPI/time", 0.25)],
)
def test_strict_failure_retains_explicit_compatibility_tolerance_units(
    clip_aware, unit, expected_error
):
    graph = _graph(epi=0.5, pressure=0.0)
    with pytest.raises(NodalEquationViolation) as failure:
        validate_nodal_equation(
            graph, 0, 0.5, 0.5625, 0.25, strict=True, clip_aware=clip_aware
        )
    assert failure.value.details["error_unit"] == unit
    assert failure.value.details["error"] == expected_error
    assert f"({unit})" in str(failure.value)


@pytest.mark.parametrize("dt", [0.0, -1.0, True, math.nan, math.inf, "0.5"])
def test_invalid_time_is_not_silently_replaced_by_zero_rate(dt):
    graph = _graph()
    with pytest.raises(TNFRValueError, match="dt"):
        validate_nodal_equation(graph, 0, 0.82, 0.82, dt)


@pytest.mark.parametrize(
    "operator_class,activation,dt",
    [
        (Emission, "keyword", True),
        (Emission, "configuration", "0.5"),
        (Coherence, "keyword", 0.0),
        (Coherence, "configuration", 0.0),
    ],
)
def test_public_upper_rail_events_reject_invalid_comparison_time_before_writes(
    operator_class, activation, dt
):
    graph = _graph(epi=1.0)
    options = {"dt": dt}
    if activation == "keyword":
        options["validate_nodal_equation"] = True
    else:
        graph.graph["VALIDATE_NODAL_EQUATION"] = True
    retained = deepcopy((dict(graph.graph), dict(graph.nodes[0])))

    with pytest.raises(OperatorPreconditionError, match="dt"):
        operator_class()(graph, 0, **options)

    # Clipped agreement at an upper rail must not hide an invalid interval.
    assert (dict(graph.graph), dict(graph.nodes[0])) == retained


@pytest.mark.parametrize(
    "configuration",
    [
        {"NODAL_EQUATION_TOLERANCE": math.nan},
        {"NODAL_EQUATION_TOLERANCE": -1.0},
        {"NODAL_EQUATION_TOLERANCE": True},
        {"NODAL_EQUATION_CLIP_AWARE": "false"},
        {"CLIP_MODE": "invalid"},
        {"CLIP_SOFT_K": 0.0},
        {"CLIP_SOFT_K": math.inf},
        {"CLIP_SOFT_K": True},
        {"EPI_MIN": 2.0},
        {"EPI_MAX": math.nan},
    ],
)
def test_invalid_active_configuration_rejects_even_for_zero_change(configuration):
    graph = _graph(capacity=0.0)
    graph.graph.update(configuration)
    retained = deepcopy((dict(graph.graph), dict(graph.nodes[0])))
    with pytest.raises(TNFRValueError):
        validate_nodal_equation(graph, 0, 0.82, 0.82, 0.5)
    with pytest.raises(OperatorPreconditionError):
        SelfOrganization()._validate_nodal_proposal(
            graph,
            {"validate_nodal_equation": True, "dt": 0.5},
            epi_before=0.82,
            epi_after=0.82,
            vf=0.0,
            dnfr=0.08,
        )
    assert (dict(graph.graph), dict(graph.nodes[0])) == retained


@pytest.mark.parametrize(
    "key, value",
    [
        (VF_PRIMARY, True),
        (VF_PRIMARY, -1.0),
        (DNFR_PRIMARY, math.nan),
        (DNFR_PRIMARY, "0.08"),
    ],
)
def test_malformed_nodal_coefficients_are_not_coerced_to_an_accepted_step(key, value):
    graph = _graph()
    graph.nodes[0][key] = value
    with pytest.raises(TNFRValueError):
        validate_nodal_equation(graph, 0, 0.82, 0.86, 0.5)


def test_thol_preflight_uses_supplied_proposal_without_overwriting_live_pressure():
    graph = _graph(pressure=0.08)
    graph.graph["NODAL_EQUATION_STRICT"] = True
    # The live post-state observation fails, but the supplied zero-pressure
    # proposal agrees with its unchanged EPI. No temporary graph edit is needed.
    with pytest.raises(NodalEquationViolation):
        validate_nodal_equation(graph, 0, 0.82, 0.82, 0.5, strict=True)
    SelfOrganization()._validate_nodal_proposal(
        graph,
        {"validate_nodal_equation": True, "dt": 0.5},
        epi_before=0.82,
        epi_after=0.82,
        vf=1.0,
        dnfr=0.0,
    )
    assert graph.nodes[0][DNFR_PRIMARY] == 0.08
