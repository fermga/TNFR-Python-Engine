"""Shared sine admission retains theorem domains and independent import paths."""

import subprocess
import sys
from copy import copy, deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.physics._sine_admission import (
    _admit_sine_source,
    _sector_source_admission,
    _sine_model_coefficients,
)
from tnfr.physics._sine_preparation import _sine_domain
from tnfr.physics.phase_cycle_geometry import derive_phase_cycle_geometry
from tnfr.physics.relational_sine_budget import certify_sine_budget_consensus
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_recovery import (
    _admit_sine_source as _legacy_source_admission,
)
from tnfr.physics.relational_sine_recovery import (
    _sector_source_admission as _legacy_sector_admission,
)
from tnfr.physics.relational_sine_symmetry import assess_sine_cycle_symmetry


@pytest.fixture(scope="module")
def source():
    graph = nx.cycle_graph(5)
    for node, value in enumerate((2, 1, -1, -1, 1)):
        graph.nodes[node].update(EPI=value, theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    model = RelationalExchangeModel(1, phase_domain="regular")
    return bound_relational_sine_exchange(graph, reference_model=model)


@pytest.fixture(scope="module")
def geometry():
    return derive_phase_cycle_geometry(nx.cycle_graph(5))


def _source_graph(source):
    graph = nx.Graph()
    graph.add_nodes_from(source.nodes)
    graph.add_edges_from(source.edges)
    graph.graph["GAMMA"] = {"type": "none"}
    for node, form, phase, capacity in zip(
        source.nodes, source.epi, source.phase, source.capacity
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity)
    return graph


def test_admission_preparation_and_symmetry_import_without_recovery(
    source_tree_environment,
):
    root = Path(__file__).resolve().parents[2]
    script = (
        "import importlib, sys\n"
        "importlib.import_module('tnfr.physics._sine_admission')\n"
        "assert 'tnfr.physics.relational_sine_pattern' not in sys.modules\n"
        "assert 'tnfr.physics.relational_sine_comparison' not in sys.modules\n"
        "importlib.import_module('tnfr.physics._sine_preparation')\n"
        "importlib.import_module('tnfr.physics.relational_sine_symmetry')\n"
        "assert 'tnfr.physics.relational_sine_recovery' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=source_tree_environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_authoritative_stored_coefficients_are_not_normalized_again(source, geometry):
    model = copy(source.reference_model)
    coefficients = Q(7, 19), Q(5, 17), Q(11, 13)
    for field, value in zip(
        ("epi_weight", "phase_weight", "storage_scale"), coefficients
    ):
        object.__setattr__(model, field, value)
    admitted, _ = _admit_sine_source(replace(source, reference_model=model))
    domain = _sine_domain(geometry, model, source.capacity)
    assert sum(coefficients[:2]) != 1
    assert _sine_model_coefficients(admitted.reference_model) == coefficients
    assert (domain.e, domain.w, domain.beta) == coefficients
    assert domain.weights == (Q(2),) * 5
    assert domain.mobility == (Q(1, 2),) * 5

    # Public graph capture consumes the same stored law. At node zero the
    # independent cycle gradient is 2, degree is 2, and capacity is 1.
    graph = _source_graph(source)
    captured = bound_relational_sine_exchange(graph, reference_model=model)
    relative = bound_relational_sine_pattern(
        graph,
        reference_node=0,
        reference_model=model,
        form_error_bounds=(0,) * 5,
        phase_error_bounds=(0,) * 5,
    )
    native = evaluate_relational_exchange(graph, model=model)
    assert captured.form_rates[0].contains(-Q(7, 19))
    assert relative.form_rate_bounds[0].contains(-Q(7, 19))
    assert captured.phase_rate_numerators()[0] == Q(65, 187)
    assert native.form_rate[0] == pytest.approx(-7 / 19, abs=2e-16)
    assert captured.reference_model is relative.reference_model is native.model is model
    assert _sine_model_coefficients(model) == coefficients


@pytest.mark.parametrize(
    "field,value",
    (
        ("epi_weight", True),
        ("epi_weight", float("inf")),
        ("epi_weight", -1),
        ("phase_weight", True),
        ("phase_weight", float("nan")),
        ("phase_weight", 0),
        ("storage_scale", True),
        ("storage_scale", float("inf")),
        ("storage_scale", 0),
    ),
)
def test_source_and_state_free_domain_revalidate_stored_coefficients(
    source, geometry, field, value
):
    model = copy(source.reference_model)
    object.__setattr__(model, field, value)
    for admission in (_admit_sine_source, _legacy_source_admission):
        with pytest.raises((TypeError, ValueError)):
            admission(replace(source, reference_model=model))
    with pytest.raises((TypeError, ValueError)):
        _sine_domain(geometry, model, source.capacity)
    graph = _source_graph(source)
    before = deepcopy(graph)
    with pytest.raises((TypeError, ValueError)):
        bound_relational_sine_exchange(graph, reference_model=model)
    with pytest.raises((TypeError, ValueError)):
        bound_relational_sine_pattern(
            graph,
            reference_node=0,
            reference_model=model,
            form_error_bounds=(0,) * 5,
            phase_error_bounds=(0,) * 5,
        )
    assert nx.utils.graphs_equal(graph, before)


def test_exact_sine_capture_preserves_tiny_positive_stored_loss(source):
    model = copy(source.reference_model)
    loss = Q(1, 2**1100)
    object.__setattr__(model, "epi_weight", loss)
    report = bound_relational_sine_exchange(
        _source_graph(source), reference_model=model
    )
    assert report.form_rates[0].contains(-loss)
    # Cycle gradients are (2, 1, -2, -2, 1), with d=2 and nu=1.
    assert report.continuous_loss == 7 * loss
    assert report.reference_model.epi_weight == loss


@pytest.mark.parametrize("consumer", ("forecast", "sampling", "resonance", "pulse"))
@pytest.mark.parametrize(
    "field,value",
    (
        ("epi_weight", True),
        ("epi_weight", -1),
        ("epi_weight", float("inf")),
        ("phase_weight", True),
        ("phase_weight", 0),
        ("phase_weight", float("nan")),
        ("storage_scale", True),
        ("storage_scale", 0),
        ("storage_scale", float("inf")),
    ),
)
def test_graph_free_consumers_reject_invalid_stored_model_before_field_or_solver(
    consumer, field, value, monkeypatch
):
    from tnfr.physics import relational_sine_forecast as forecast
    from tnfr.physics.relational_sine_resonance import assess_sine_cycle_resonance
    from tnfr.physics.relational_sine_sampling import bound_sine_sampling_smoothness
    from tnfr.physics.relational_sine_scale import assess_sine_replica_pulse

    model = RelationalExchangeModel(
        1, epi_weight=0 if consumer == "pulse" else 1, phase_domain="regular"
    )
    object.__setattr__(model, field, value)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid coefficients must reject before field or solver work")

    monkeypatch.setattr(forecast, "validated_taylor_step", forbidden)
    monkeypatch.setattr(forecast, "_sine_flow", forbidden)
    with pytest.raises((TypeError, ValueError)):
        if consumer == "forecast":
            forecast.bound_sine_flow(
                (0, 1, 0, 0, 1),
                neighbors=((1,), (0,)),
                visible_capacity=(1,),
                model=model,
                observation_time=0,
                end_time=Q(1, 100),
                time_step=Q(1, 100),
            )
        elif consumer == "sampling":
            bound_sine_sampling_smoothness(
                reference_model=model,
                form_diameter_bound=1,
                capacity_ceiling=1,
                window_start=0,
                window_end=0,
            )
        elif consumer == "resonance":
            assess_sine_cycle_resonance(
                model=model, node_count=5, mode_index=1, capacity=1
            )
        else:
            assess_sine_replica_pulse(
                reference_model=model,
                form_half_difference=Q(1, 8),
                phase_half_difference=0,
                capacity=1,
            )


@pytest.mark.parametrize("zero_premise", ("loss", "capacity"))
def test_shared_admission_does_not_transfer_stronger_analytic_premises(
    source, geometry, zero_premise
):
    model = (
        RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
        if zero_premise == "loss"
        else source.reference_model
    )
    capacity = (Q(0),) * 5 if zero_premise == "capacity" else source.capacity
    declared = replace(source, reference_model=model, capacity=capacity)
    symmetry = assess_sine_cycle_symmetry(
        declared, permutation_indices=(0, 4, 3, 2, 1), cycle=tuple(range(5))
    )
    assert symmetry.trajectory_symmetry_certified
    assert symmetry.zero_winding_when_nonantipodal
    with pytest.raises(ValueError, match="positive"):
        certify_sine_budget_consensus(
            geometry,
            reference_model=model,
            capacity=capacity,
            form_storage_budget=0,
        )


def test_budget_neutral_source_and_legacy_adapters_keep_sector_cap_separate():
    graph = nx.cycle_graph(51)
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(
        graph, reference_model=RelationalExchangeModel(1, phase_domain="regular")
    )
    for admission in (_admit_sine_source, _legacy_source_admission):
        admitted, edges = admission(source)
        assert admitted == source
        assert len(edges) == len(admitted.nodes) == 51
    for admission in (_sector_source_admission, _legacy_sector_admission):
        with pytest.raises(ValueError, match="2 to 32"):
            admission(source)
