"""Full-state sector barriers, uncertainty boundaries and source integrity."""

import json
from copy import copy
from dataclasses import FrozenInstanceError, dataclass, replace
from fractions import Fraction as Q

import pytest

from tests.physics.test_sine_phase_offset_partition import CYCLE_LEAF_EDGES, _source
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics.phase_cycle_geometry import C5_PHASE_SECTOR_BARRIER
from tnfr.physics.relational_sine_partition import assess_sine_contact_averaging
from tnfr.physics.relational_sine_regional import assess_sine_cycle_barrier
from tnfr.sdk import export_to_json, relational_report_to_dict


def _state(
    *, phase=None, epi=None, edges=CYCLE_LEAF_EDGES, order=None, label=lambda i: i
):
    anchor = _source(edges, order=order, label=label)
    count = len(anchor.nodes)
    phase = (Q(0),) * count if phase is None else tuple(phase)
    epi = (Q(0),) * count if epi is None else tuple(epi)
    order = tuple(range(count)) if order is None else tuple(order)
    return replace(
        anchor, phase=tuple(phase[i] for i in order), epi=tuple(epi[i] for i in order)
    )


def _assess(source, *, cycle=range(5)):
    return assess_sine_cycle_barrier(source, cycle=cycle)


def test_nonflat_ordered_contacts_do_not_evade_receiver_barrier():
    deviations = (Q(1, 10), Q(1, 10), -Q(1, 10), -Q(1, 10), Q(0))
    source = _state(
        epi=(Q(1, 4),) * 5 + (-Q(3, 4),) * 5,
        phase=(Q(0),) * 5 + tuple(Q(1, 4) + value for value in deviations),
    )
    report = _assess(source)
    assert report.initial_receiver_phase_flat_certified
    assert len(set(source.phase)) > 1
    assert report.initial_winding == 0
    assert (
        Q(5, 2)
        < report.full_storage_bounds.lo
        < report.full_storage_bounds.hi
        < Q(7, 2)
    )
    assert report.regional_phase_storage_bounds == I(0)
    assert report.energy_barrier == C5_PHASE_SECTOR_BARRIER
    assert report.strict_energy_bound_certified
    assert report.status == "certified"
    assert tuple(sector.winding for sector in report.sectors) == (-1, 1)
    assert all(sector.source_membership == "outside" for sector in report.sectors)
    assert all(sector.acute_acquisition_excluded for sector in report.sectors)
    assert not any(sector.sector_invariance_certified for sector in report.sectors)


def test_arbitrary_zero_winding_state_is_covered_without_consensus():
    phases = (Q(0), Q(1, 10), -Q(1, 10), Q(1, 20), -Q(1, 20))
    report = _assess(_state(phase=phases * 2))
    assert not report.initial_receiver_phase_flat_certified
    assert report.initial_winding == 0
    assert report.strict_energy_bound_certified
    assert all(sector.acute_acquisition_excluded for sector in report.sectors)


def test_protected_wider_sector_need_not_be_acute():
    phases = tuple(map(Q, ("0", "1.7", "2.85", "4", "5.15")))
    report = _assess(_state(phase=phases * 2))
    negative, positive = report.sectors
    assert report.initial_winding == 1
    assert report.initial_acute_margin_lower_bound < 0
    assert report.sector_margin_lower_bound > 0
    assert report.full_storage_bounds.hi < Q(7, 2)
    assert positive.source_membership == "inside"
    assert positive.status == "sector_invariance_certified"
    assert positive.sector_invariance_certified
    assert not positive.acute_acquisition_excluded
    assert negative.acute_acquisition_excluded


def test_transient_nonacute_unit_winding_can_still_be_outside_both_sectors():
    phases = tuple(Q(4 * i, 5) for i in range(5))
    report = _assess(_state(phase=phases * 2))
    assert report.initial_winding == 1
    assert report.sector_margin_lower_bound < 0
    assert report.full_storage_bounds.hi < Q(7, 2)
    assert all(sector.acute_acquisition_excluded for sector in report.sectors)
    assert not any(sector.sector_invariance_certified for sector in report.sectors)


def test_antipodal_interval_ambiguity_does_not_invent_winding():
    # A rational close enough to pi for the outward branch test to remain
    # unresolved. It is not asserted to be the exact irrational antipode.
    phases = tuple(i * pi_interval().midpoint / 4 for i in range(5))
    report = _assess(_state(phase=phases * 2))
    assert report.initial_winding is None
    assert report.cycle_principal_gap_bounds is None
    assert report.full_storage_bounds.hi < Q(7, 2)
    # The enclosing cosine is strictly below -1/2, excluding the wider
    # sector regardless of the unresolved branch convention.
    assert all(sector.acute_acquisition_excluded for sector in report.sectors)


def test_exact_equal_budget_excludes_acquisition_without_asserting_source_equilibrium():
    source = _state(epi=(0,) * 5 + (2, 1, 1, 1, 0))
    report = _assess(source)
    assert report.full_storage_bounds == I(Q(7, 2))
    assert report.energy_margin_lower_bound == 0
    assert not report.strict_energy_bound_certified
    assert report.finite_time_energy_bound_certified
    assert report.status == "certified" and not report.reasons
    assert all(sector.acute_acquisition_excluded for sector in report.sectors)
    # Energy equality alone does not identify the source as a saddle. Its
    # actual scaled-time phase row at node0 is (0-2)/3, hence nonstationary.
    assert sum((source.epi[0] - source.epi[j] for j in (1, 4, 5)), Q(0)) / 3 == -Q(2, 3)
    assert report.to_dict()["report"]["finite_time_energy_bound_certified"]


@pytest.mark.parametrize(
    "leaf_forms", ((Q(2) + Q(1, 2**64), 1, 1, 1, 0), (3, 0, 0, 0, 0))
)
def test_high_full_budget_is_unavailable_not_a_formation_verdict(leaf_forms):
    report = _assess(_state(epi=(0,) * 5 + leaf_forms))
    assert report.full_storage_bounds.lo > Q(7, 2)
    assert report.energy_margin_lower_bound < 0
    assert not report.strict_energy_bound_certified
    assert not report.finite_time_energy_bound_certified
    assert report.status == "unavailable"
    assert all(sector.source_membership == "outside" for sector in report.sectors)
    assert not any(sector.acute_acquisition_excluded for sector in report.sectors)
    assert report.reasons == ("full_storage_at_most_barrier_not_certified",)


def test_outward_storage_bound_above_equality_does_not_pass_from_its_lower_endpoint():
    # The exact nonzero rational phase adds positive potential to the exact
    # form budget7/2. Its tiny potential is below interval resolution, so
    # neither a rounded midpoint nor the lower endpoint can admit equality.
    phase = (Q(1, 2**100),) + (Q(0),) * 9
    report = _assess(_state(epi=(0,) * 5 + (2, 1, 1, 1, 0), phase=phase))
    assert report.full_storage_bounds.lo <= Q(7, 2) < report.full_storage_bounds.hi
    assert not report.finite_time_energy_bound_certified
    assert report.status == "unavailable"


def test_boundary_energy_equality_reconstructs_a_full_equilibrium_with_live_extra_edges():
    from tnfr.physics.phase_cycle_geometry import (
        _derive,
        reconstruct_circular_phase_state,
    )

    # The additional environment node joins two equal-phase nodes. All
    # noncycle storage vanishes, while the C5 attains its exact boundary min.
    edges = CYCLE_LEAF_EDGES + ((0, 10), (5, 10))
    geometry = _derive(
        tuple(range(11)), tuple(sorted(tuple(sorted(edge)) for edge in edges))
    )
    turns = tuple(Q(i - 2, 6) for i in range(5)) * 2 + (-Q(1, 3),)
    differences = tuple(
        (turns[j] - turns[i] + Q(1, 2)) % 1 - Q(1, 2) for i, j in geometry.edges
    )
    state = reconstruct_circular_phase_state(geometry, edge_turns=differences)
    assert not any(state.symbolic_sine_coefficients)
    curvature = {Q(0): Q(1), Q(1, 6): Q(1, 2), Q(1, 3): -Q(1, 2)}
    assert sum(1 - curvature[abs(turn)] for turn in differences) == Q(7, 2)
    # Uniform full form is exactly in the full Laplacian's kernel. Together
    # with symbolic zero torque above, both full nonlinear rows vanish.
    form = (Q(7, 11),) * 11
    assert all(form[i] == form[j] for i, j in geometry.edges)


def test_environmental_storage_cannot_be_discarded_or_supplied_by_cached_fields():
    source = _state(epi=(0,) * 5 + (3, 0, 0, 0, 0))
    report = _assess(
        replace(
            source, storage=I(-999), form_gradient=(), phase_rates=(), form_rates=()
        )
    )
    assert report.regional_phase_storage_bounds == I(0)
    assert report.full_storage_bounds == I(Q(9, 2))
    assert not report.strict_energy_bound_certified


def test_source_order_and_cycle_reversal_preserve_scoped_conclusions():
    phases = tuple(Q(5 * i, 4) for i in range(5)) * 2
    ordinary = _assess(_state(phase=phases))
    order = (8, 0, 5, 3, 6, 9, 1, 7, 4, 2)
    reordered = _state(phase=phases, order=order, label=lambda i: ("node", i))
    actual = _assess(reordered, cycle=tuple(("node", i) for i in range(4, -1, -1)))
    assert actual.initial_winding == -ordinary.initial_winding == -1
    assert actual.full_storage_bounds == ordinary.full_storage_bounds
    assert actual.sectors[0].sector_invariance_certified
    assert actual.cycle_indices == tuple(order.index(i) for i in range(4, -1, -1))


def test_selected_cycle_can_have_extra_live_chords_and_environment():
    edges = CYCLE_LEAF_EDGES + ((0, 2), (5, 10), (10, 11))
    report = _assess(_state(edges=edges))
    assert len(report.source.nodes) == 12
    assert report.full_storage_bounds == I(0)
    assert all(sector.acute_acquisition_excluded for sector in report.sectors)


@pytest.mark.parametrize("field", ("epi", "phase", "capacity"))
def test_malformed_primitive_scalars_reject_before_using_cached_verdicts(field):
    source = _state()
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, **{field: (False,) + getattr(source, field)[1:]}))


@pytest.mark.parametrize("field", ("epi_weight", "phase_weight", "storage_scale"))
def test_boolean_law_coefficients_are_not_unit_conservative_declarations(field):
    source = _state()
    model = copy(source.reference_model)
    object.__setattr__(model, field, False if field == "epi_weight" else True)
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, reference_model=model))


@pytest.mark.parametrize(
    "cycle",
    ((0, 1, 2, 3), (0, 1, 2, 3, 3), (0, 2, 1, 3, 4), (0, 1, 2, 3, 99), {0, 1, 2, 3, 4}),
)
def test_invalid_cycle_declarations_reject(cycle):
    with pytest.raises((TypeError, ValueError)):
        _assess(_state(), cycle=cycle)


def test_unsupported_law_capacity_and_source_types_reject():
    source = _state()
    cases = (
        object(),
        replace(source, capacity=(Q(0),) + source.capacity[1:]),
        replace(source, degrees=(99,) + source.degrees[1:]),
        replace(source, law="native"),
    )
    for invalid in cases:
        with pytest.raises((TypeError, ValueError)):
            _assess(invalid)


def test_sdk_export_is_detached_and_validates_retained_labels(tmp_path):
    report = _assess(_state())
    payload = relational_report_to_dict(report)
    assert report.to_dict()["schema"] == "tnfr.sine-cycle-barrier.v1"
    assert payload["report_type"] == "SineCycleBarrier"
    assert payload["report"] == report.to_dict()["report"]
    assert payload["report"]["energy_barrier"] == {"numerator": 7, "denominator": 2}
    output = tmp_path / "barrier.json"
    export_to_json(payload, output)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FrozenInstanceError):
        report.status = "unavailable"

    @dataclass(frozen=True)
    class Opaque:
        value: int

    malformed = replace(report, cycle=(Opaque(0),) + report.cycle[1:])
    with pytest.raises(TypeError):
        relational_report_to_dict(malformed)


def test_contact_averaging_sdk_atomic_export_retains_exact_and_unavailable_bounds(
    tmp_path,
):
    source = _state(
        epi=(Q(5),) * 5 + (Q(-15),) * 5,
        phase=(Q(0),) * 5 + (Q(7, 20), Q(7, 20), Q(3, 20), Q(3, 20), Q(1, 4)),
    )
    for name, candidate in (
        ("excluded", source),
        ("unavailable", replace(source, epi=(Q(0),) * 10)),
    ):
        report = assess_sine_contact_averaging(
            candidate, cycle=range(5), scaled_horizon=Q(1)
        )
        payload = relational_report_to_dict(report)
        assert payload["report_type"] == "SineContactAveraging"
        assert payload["report"] == report.to_dict()["report"]
        assert payload["report"]["status"] == name
        assert payload["report"]["scaled_horizon"] == {
            "numerator": 1,
            "denominator": 1,
        }
        if name == "unavailable":
            assert payload["report"]["receiver_phase_storage_upper_bound"] is None
        output = tmp_path / f"contact-{name}.json"
        export_to_json(payload, output)
        assert json.loads(output.read_text(encoding="utf-8")) == payload
