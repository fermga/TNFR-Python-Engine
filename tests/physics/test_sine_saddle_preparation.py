"""Static rational preparation, precision and full-source admission controls.

No trajectory or stored scientific producer is evaluated here. Independent
high-precision algebra checks the actual rational state and its error from the
single correlated algebraic recipe, not unrelated eigenvector interval corners.
"""

import json
from copy import copy
from dataclasses import dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tests.physics.test_sine_cycle_barrier import _state
from tests.physics.test_sine_formation_bridge import (
    _energy,
    _flow,
    _independent_direction,
    _mp,
)
from tests.physics.test_sine_phase_offset_partition import CYCLE_LEAF_EDGES
from tnfr.mathematics._phase_resultant_chamber import certified_cosine_bounds
from tnfr.mathematics._rational_interval import I, cos
from tnfr.physics.relational_sine_comparison import _sine_phase_storage
from tnfr.physics.relational_sine_corridor import prepare_sine_saddle_state
from tnfr.sdk import export_to_json, relational_report_to_dict


@pytest.fixture(scope="module")
def anchor():
    return _state(
        epi=tuple(Q(i * i, 7) for i in range(10)),
        phase=tuple(Q(i, 9) for i in range(10)),
    )


@pytest.fixture(scope="module")
def prepared(anchor):
    return prepare_sine_saddle_state(anchor, cycle=range(5))


def test_actual_exact_rational_preparation_has_both_gates_not_a_zero_winding_source(
    prepared, anchor
):
    assert prepared.status == "certified"
    assert prepared.preparation_certified
    assert (
        prepared.outer_corridor_gate_certified
        and prepared.inner_corridor_gate_certified
    )
    assert prepared.initial_winding == 1
    assert prepared.initial_branches_certified
    assert prepared.exact_signed_reconstruction_certified
    assert not prepared.numerical_zero_winding_source_available
    assert prepared.prepared_state.epi != anchor.epi
    assert prepared.prepared_state.phase != anchor.phase
    for row in (prepared.prepared_state.epi, prepared.prepared_state.phase):
        assert all(type(value) is Q for value in row)
        for base in (0, 5):
            assert row[base + 2] == 0
            assert row[base] == -row[base + 4]
            assert row[base + 1] == -row[base + 3]
    assert prepared.full_storage_bounds.lo > Q(7, 2)
    assert prepared.full_storage_bounds.hi < Q(7, 2) + 5 * prepared.epsilon**2
    assert prepared.outer_phase_displacement_coefficient_lower_bound > 1
    assert prepared.inner_phase_displacement_coefficient_upper_bound < -1
    assert prepared.outer_momentum_gate_margin > 0
    assert prepared.inner_momentum_gate_margin > 0


def test_independent_correlated_root_and_full_rows_validate_rational_center(prepared):
    with mp.workdps(110):
        sigma, form_direction, phase_direction, target = _independent_direction()
        epsilon = _mp(prepared.epsilon)
        ideal_form = tuple(3 * epsilon * value for value in form_direction)
        ideal_phase = tuple(
            value - epsilon * direction
            for value, direction in zip(target, phase_direction)
        )
        actual_form = tuple(_mp(value) for value in prepared.prepared_state.epi)
        actual_phase = tuple(_mp(value) for value in prepared.prepared_state.phase)
        for actual, ideal, bounds in zip(
            actual_form + actual_phase,
            ideal_form + ideal_phase,
            prepared.form_recipe_bounds + prepared.phase_recipe_bounds,
        ):
            assert _mp(bounds.lo) <= ideal <= _mp(bounds.hi)
            assert abs(actual - ideal) <= _mp(prepared.preparation_error_bound)
        energy = _energy(actual_form, actual_phase)
        assert (
            _mp(prepared.full_storage_bounds.lo)
            <= energy
            <= _mp(prepared.full_storage_bounds.hi)
        )
        rows = _flow(actual_form, actual_phase)
        # The detached comparison retains original structural t, not tau.
        for actual, bound in zip(
            rows,
            prepared.prepared_state.form_rates + prepared.prepared_state.phase_rates,
        ):
            assert _mp(bound.lo) <= actual / mp.pi <= _mp(bound.hi)
        growth = mp.exp(3 * sigma)
        momentum = sum(
            weight * form_direction[i] for weight, i in zip((6, 3, 2, 1), (0, 1, 5, 6))
        )
        normalized = _mp(prepared.normalized_local_endpoint_error_bound)
        assert growth - 2 / growth - normalized >= _mp(
            prepared.outer_phase_displacement_coefficient_lower_bound
        )
        assert momentum * (growth + 2 / growth) - 12 * normalized >= _mp(
            prepared.outer_momentum_coefficient_lower_bound
        )
        assert 1 / growth - 2 * growth + normalized <= _mp(
            prepared.inner_phase_displacement_coefficient_upper_bound
        )
        assert momentum * (1 / growth + 2 * growth) - 12 * normalized >= _mp(
            prepared.inner_negative_momentum_coefficient_lower_bound
        )
    assert (
        prepared.local_endpoint_error_bound
        == prepared.formation.local_remainder_bound
        + 729 * prepared.preparation_error_bound
    )
    assert (
        prepared.normalized_local_endpoint_error_bound
        == prepared.local_endpoint_error_bound / prepared.epsilon
    )


def test_full_and_regional_storage_share_precision_that_resolves_critical_excess(
    prepared,
):
    source = prepared.prepared_state
    legacy = source.form_storage + sum(
        (
            1 - I(*certified_cosine_bounds(source.phase[j] - source.phase[i]))
            for i, j in CYCLE_LEAF_EDGES
        ),
        I(0),
    )
    assert legacy.lo < Q(7, 2) < legacy.hi
    assert source.storage.lo > Q(7, 2)
    assert source.storage.hi - source.storage.lo < prepared.epsilon**2 / 1000
    regions = source.regional_storage_balance(region=range(5))
    assert regions.regional_phase_storage == _sine_phase_storage(
        source.phase, regions.regional_edge_indices
    )
    assert regions.boundary_phase_storage == _sine_phase_storage(
        source.phase, regions.boundary_edge_indices
    )
    assert (
        regions.regional_storage + regions.complement_storage + regions.boundary_storage
    ).lo > Q(7, 2)
    shifted = tuple(value + Q(2**200) for value in source.phase)
    assert _sine_phase_storage(shifted, CYCLE_LEAF_EDGES) == source.phase_storage
    boxes = (I(0, Q(1, 10)), I(Q(1, 5), Q(3, 10)))
    assert _sine_phase_storage(boxes, ((0, 1),)) == 1 - cos(boxes[1] - boxes[0])


def test_unresolved_tiny_preparation_is_not_promoted_from_symbolic_existence(anchor):
    result = prepare_sine_saddle_state(anchor, cycle=range(5), epsilon=Q(1, 2**512))
    assert result.formation.formation_existence_certified
    assert result.status == "unavailable"
    assert not result.preparation_certified
    assert not result.actual_energy_certified
    assert "actual_nearcritical_energy_not_resolved" in result.reasons
    assert result.preparation_error_bound > 0


def test_derived_capture_fields_and_unrelated_anchor_state_do_not_select_preparation(
    anchor, prepared
):
    poisoned = replace(
        anchor,
        storage=I(-999),
        form_storage=Q(-999),
        phase_storage=I(-999),
        form_rates=(),
        phase_rates=(),
        form_gradient=(),
    )
    result = prepare_sine_saddle_state(poisoned, cycle=range(5))
    other = prepare_sine_saddle_state(
        replace(poisoned, epi=(Q(1000),) * 10, phase=(Q(99),) * 10), cycle=range(5)
    )
    for candidate in (result, other):
        assert candidate.preparation_certified
        assert candidate.prepared_state.epi == prepared.prepared_state.epi
        assert candidate.prepared_state.phase == prepared.prepared_state.phase
        assert candidate.full_storage_bounds == prepared.full_storage_bounds


@pytest.mark.parametrize(
    "epsilon",
    [True, False, 0, -1, Q(1, 2**31), float("inf"), float("nan"), "1/4294967296"],
)
def test_invalid_preparation_size_rejects(anchor, epsilon):
    with pytest.raises((ValueError, TypeError)):
        prepare_sine_saddle_state(anchor, cycle=range(5), epsilon=epsilon)


@pytest.mark.parametrize(
    "field,value",
    [("epi", True), ("phase", float("nan")), ("capacity", True), ("capacity", 0)],
)
def test_actual_source_primitives_are_readmitted_before_preparation(
    anchor, field, value
):
    values = (value,) + getattr(anchor, field)[1:]
    with pytest.raises((ValueError, TypeError)):
        prepare_sine_saddle_state(replace(anchor, **{field: values}), cycle=range(5))


def test_model_and_support_cannot_be_replaced_by_cached_certificates(anchor):
    model = copy(anchor.reference_model)
    object.__setattr__(model, "phase_weight", True)
    with pytest.raises((ValueError, TypeError)):
        prepare_sine_saddle_state(
            replace(anchor, reference_model=model), cycle=range(5)
        )
    with pytest.raises((ValueError, TypeError)):
        prepare_sine_saddle_state(
            replace(anchor, edges=anchor.edges + ((0, 2),)), cycle=range(5)
        )


def test_node_order_and_labels_preserve_prepared_physical_rows(prepared):
    order = (8, 4, 1, 6, 2, 7, 0, 5, 9, 3)
    source = _state(order=order, label=lambda i: f"n{i}")
    result = prepare_sine_saddle_state(source, cycle=tuple(f"n{i}" for i in range(5)))
    assert result.preparation_certified
    position = {node: i for i, node in enumerate(result.prepared_state.nodes)}
    for field in ("epi", "phase"):
        assert tuple(
            getattr(result.prepared_state, field)[position[f"n{i}"]] for i in range(10)
        ) == getattr(prepared.prepared_state, field)
    assert result.full_storage_bounds == prepared.full_storage_bounds


def test_sdk_preserves_exact_preparation_and_scope(prepared, tmp_path):
    payload = relational_report_to_dict(prepared)
    assert prepared.to_dict()["schema"] == "tnfr.sine-saddle-preparation.v1"
    assert payload["report_type"] == "SineSaddlePreparation"
    assert payload["report"]["initial_winding"] == 1
    assert not payload["report"]["numerical_zero_winding_source_available"]
    destination = tmp_path / "preparation.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload

    @dataclass(frozen=True)
    class Opaque:
        index: int

    bad = replace(
        prepared.prepared_state, nodes=(Opaque(0),) + prepared.prepared_state.nodes[1:]
    )
    with pytest.raises((TypeError, ValueError)):
        relational_report_to_dict(replace(prepared, prepared_state=bad))
