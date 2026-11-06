"""An ideal structural word converts hidden phase into persistent mean error.

The fixed exact carrier and dyadic offsets are supplied at one common event
time. These static mathematical controls neither locate a guard hit nor run
a word/flow, and do not infer a represented runtime separation from an ideal
phase defect near binary64 raw-phase spacing.
"""

from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.config.thresholds import DNFR_RECEPTION_MAX, EPI_SATURATION_MAX
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, arg, cos, pi_interval, sin
from tnfr.operators import network_stage, word_execution
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators.definitions import Coherence, Reception, Silence
from tnfr.operators.definitions_base import Operator
from tnfr.operators.factor_contracts import (
    canonical_glyph_factor_defaults,
    validate_glyph_factor,
)
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.physics._organized_source_capture import (
    _derive_organized_source_capture_barrier,
    _organized_source_offset_edge_angles,
)
from tnfr.physics._organized_source_child_orbit import (
    _encode_organized_source_child_orbit,
)
from tnfr.physics._organized_source_child_quotient import (
    _lift_organized_source_child_quotient,
    _restrict_organized_source_child_quotient,
)
from tnfr.physics._organized_source_port import (
    _observe_organized_source_capacity,
    _organized_source_capacity_rows,
)
from tnfr.physics._organized_source_reuse_budget import (
    _assess_organized_source_reuse_budget,
)

PHI = Q(1, 128)
HIDDEN = Q(1, 2**20)
ZERO = (Q(0),) * 45
UNIT = (Q(1),) * 45
SWAP = tuple(9 * (0, 2, 1, 3, 4)[i // 9] + i % 9 for i in range(45))


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail(
            "phase-memory controls cannot execute events, flows or hit searches"
        )

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def _fields(geometry, offsets):
    """Independent static ninety-edge sums, admitting interval ideal offsets."""
    current, energy = [I(0) for _ in range(45)], I(0)
    alpha = 2 * pi_interval() / 9
    angles = []
    for left, right, weight in geometry.weighted_edges:
        gap = alpha * ((right % 9) - (left % 9)) + offsets[right] - offsets[left]
        edge_current = weight * sin(gap)
        current[left] += edge_current
        current[right] -= edge_current
        energy += weight * (1 - cos(gap))
        turn = int(left // 9 == right // 9 and left % 9 == 8 and right % 9 == 0)
        angles.append(gap + 2 * turn * pi_interval())
    return tuple(current), energy, tuple(angles)


@pytest.fixture(scope="module")
def preparation():
    offsets = list(ZERO)
    offsets[9], offsets[18] = PHI + HIDDEN, PHI - HIDDEN
    offsets = tuple(offsets)
    encoded = _encode_organized_source_child_orbit(
        form=ZERO, phase_offsets=offsets, capacities=UNIT
    )
    projected = _lift_organized_source_child_quotient(
        form_coordinates=encoded.form_coordinates,
        phase_offset_coordinates=encoded.phase_offset_coordinates,
        capacity_coordinates=encoded.capacity_coordinates,
    )
    fine = _observe_organized_source_capacity(
        form=ZERO,
        phase_offsets=offsets,
        capacities=UNIT,
        phase_mobility="shared_capacity",
    )
    core = _observe_organized_source_capacity(
        form=ZERO,
        phase_offsets=projected.phase_offsets,
        capacities=UNIT,
        phase_mobility="shared_capacity",
    )
    defaults = canonical_glyph_factor_defaults()
    factors = {
        key: Q(validate_glyph_factor(key, defaults[key]))
        for key in ("EN_mix", "IL_dnfr_factor", "SHA_vf_factor")
    }
    factors["phase_coefficient"] = Q(DEFAULT_PHASE_LOCKING_COEFFICIENT)
    return offsets, encoded, projected, fine, core, factors


@pytest.fixture(scope="module")
def increments(preparation):
    offsets, _, projected, fine, _, factors = preparation
    geometry = fine.unit_capacity_reference.geometry
    alpha = 2 * pi_interval() / 9
    resultants = []
    changes = []
    for incoming in (offsets, projected.phase_offsets):
        local, output = [], []
        for target in (0, 1):
            neighbors = tuple(
                right if left == target else left
                for left, right, _ in geometry.weighted_edges
                if target in (left, right)
            )
            gaps = tuple(
                alpha * ((j % 9) - target) + incoming[j] - incoming[target]
                for j in neighbors
            )
            real = sum((cos(gap) for gap in gaps), I(0))
            imaginary = sum((sin(gap) for gap in gaps), I(0))
            local.append((real, imaginary))
            output.append(factors["phase_coefficient"] * arg(real, imaginary))
        resultants.append(tuple(local))
        changes.append(tuple(output))
    r = 1 + 2 * cos(alpha)
    cb, sp, cp = cos(I(HIDDEN)), sin(I(PHI)), cos(I(PHI))
    # Cross-product formula avoids subtracting the nearly equal arguments.
    cross_real = r**2 + 2 * r * cp * (1 + cb) + 4 * cb
    cross_imaginary = -4 * r * sin(I(HIDDEN / 2)) ** 2 * sp
    delta = factors["phase_coefficient"] * arg(cross_real, cross_imaginary)
    return tuple(resultants), tuple(changes), delta


def test_exact_offset_projection_matches_initial_state_and_means(preparation):
    offsets, encoded, projected, fine, core, _ = preparation
    geometry = fine.unit_capacity_reference.geometry
    assert (
        geometry
        == core.unit_capacity_reference.geometry
        == encoded.geometry.fine_geometry
    )
    expected_edges = {
        tuple(sorted((9 * child + slot, 9 * child + (slot + 1) % 9)))
        for child in range(5)
        for slot in range(9)
    } | {
        (9 * left + slot, 9 * right + slot)
        for left, right in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
        for slot in range(9)
    }
    assert len(geometry.node_order) == 45 and len(geometry.weighted_edges) == 90
    assert {
        tuple(sorted((i, j))) for i, j, _ in geometry.weighted_edges
    } == expected_edges
    assert all(weight == 1 for _, _, weight in geometry.weighted_edges)
    assert projected.form == ZERO and projected.capacities == UNIT
    assert projected.phase_offsets == tuple(
        (offsets[i] + offsets[SWAP[i]]) / 2 for i in range(45)
    )
    assert projected.phase_offsets[9] == projected.phase_offsets[18] == PHI
    assert encoded.hidden_coordinates == (Q(0),) * 9 + (HIDDEN,) + (Q(0),) * 8
    assert encoded.hidden_gram[9][9] == HIDDEN**2
    assert sum(offsets) == sum(projected.phase_offsets) == 2 * PHI
    assert (
        fine.inverse_capacity_weighted_phase_offset_total
        == core.inverse_capacity_weighted_phase_offset_total
        == 2 * PHI
    )
    with pytest.raises(ValueError, match="exactly equal"):
        _restrict_organized_source_child_quotient(
            form=ZERO, phase_offsets=offsets, capacities=UNIT
        )
    assert (
        _restrict_organized_source_child_quotient(
            form=ZERO, phase_offsets=projected.phase_offsets, capacities=UNIT
        )
        == projected
    )


def test_exact_carrier_domain_reuses_barrier_budget_factors_and_signed_policies(
    preparation,
):
    offsets, _, projected, fine, core, factors = preparation
    barrier = _derive_organized_source_capture_barrier()
    geometry = fine.unit_capacity_reference.geometry
    assert barrier.geometry == geometry and barrier.barrier_certified
    assert barrier.boundary_energy_lower_bound > Q(1, 100)
    assert geometry.support_degrees[:2] == (5, 5)
    for incoming, observation in ((offsets, fine), (projected.phase_offsets, core)):
        independent, energy, angles = _fields(geometry, tuple(I(x) for x in incoming))
        shared_angles = _organized_source_offset_edge_angles(geometry, incoming)
        assert len(angles) == len(shared_angles) == 90
        assert all(angle.abs_max < pi_interval().lo / 2 for angle in angles)
        assert all((a - b).contains(0) for a, b in zip(angles, shared_angles))
        assert all(
            (a - b).contains(0)
            for a, b in zip(
                independent, observation.unit_capacity_reference.phase_current_bounds
            )
        )
        excess = energy - barrier.target_storage_bounds
        hidden_cosine = cos(I(HIDDEN)) if incoming == offsets else I(1)
        exact_excess = (
            4 * (1 + cos(geometry.alpha_bounds)) * (1 - cos(I(PHI)) * hidden_cosine)
        )
        assert (excess - exact_excess).contains(0)
        assert 0 < excess.lo <= excess.hi < 4 * (PHI**2 + HIDDEN**2) < Q(1, 100)
        assert all(
            observation.pressure_tau_bounds[i].hi < Q(DNFR_RECEPTION_MAX)
            for i in (0, 1)
        )
    budget = _assess_organized_source_reuse_budget(
        reception_mix=factors["EN_mix"],
        silence_capacity_factor=factors["SHA_vf_factor"],
        initial_mean_bound=Q(0),
        energy_budget=Q(1, 100),
        word_count=1,
    )
    assert budget.unit_form_rails_certified
    assert budget.form_absolute_upper_bound < Q(EPI_SATURATION_MAX)
    assert budget.reception_pressure_absolute_upper_bound < Q(DNFR_RECEPTION_MAX)
    assert Q(0.01) <= 1 and budget.final_pair_capacity == factors["SHA_vf_factor"] > 0
    assert 0 < factors["EN_mix"] <= 1
    assert 0 < factors["phase_coefficient"] < 1
    assert 0 < factors["IL_dnfr_factor"] < 1
    sequence = ValidatedSequence(
        (Reception(), Coherence(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    assert sequence.names == ("reception", "coherence", "silence")
    # AL,IL history/ordinal two, common kind, prior Coherence, zero debt,
    # initialized core grammar and IL event_domain remain supplied premises.
    # The grammar initialization flag records that supplied established context,
    # not a claim that this preparation's current zero EPI is nonzero.
    # Signed-threshold IL is not inferred from this zero-form preparation.


def test_simultaneous_resultants_check_both_targets_and_a_strict_phase_defect(
    preparation, increments
):
    _, _, _, _, _, factors = preparation
    resultants, changes, delta = increments
    alpha = 2 * pi_interval() / 9
    r = 1 + 2 * cos(alpha)
    core = (r + 2 * cos(I(PHI)), 2 * sin(I(PHI)))
    fine = (r + 2 * cos(I(PHI)) * cos(I(HIDDEN)), 2 * sin(I(PHI)) * cos(I(HIDDEN)))
    for actual, expected in ((resultants[0][0], fine), (resultants[1][0], core)):
        assert actual[0].lo > 4 and actual[1].lo > 0
        assert all((a - b).contains(0) for a, b in zip(actual, expected))
    for actual in (resultants[0][1], resultants[1][1]):
        assert (actual[0] - (3 + 2 * cos(alpha))).contains(0)
        assert actual[1].contains(0)
    # The target-1 sine terms cancel exactly in pairs on this input; its
    # increment is zero. It does not read the new target-0 phase serially.
    assert all(pair[1].contains(0) for pair in changes)
    s = pytest.importorskip("sympy")
    angle = s.Symbol("angle", real=True)
    assert s.sin(angle) + s.sin(-angle) + 3 * s.sin(0) == 0
    assert s.cos(angle) + s.cos(-angle) + 3 * s.cos(0) == 3 + 2 * s.cos(angle)
    assert delta.hi < 0
    assert (changes[0][0] - changes[1][0] - delta).contains(0)
    assert delta.abs_max < factors["phase_coefficient"] * HIDDEN**2 / 4
    assert 0 < changes[0][0].lo <= changes[0][0].hi < changes[1][0].lo
    assert changes[1][0].hi < factors["phase_coefficient"] * PHI


def test_primary_successors_keep_every_other_coordinate_and_do_passive_work(
    preparation, increments
):
    offsets, _, projected, fine, _, factors = preparation
    _, changes, delta = increments
    geometry = fine.unit_capacity_reference.geometry
    final_capacity = (factors["SHA_vf_factor"],) * 2 + UNIT[2:]
    assert final_capacity[0] == final_capacity[1] > 0
    for incoming, change in (
        (offsets, changes[0][0]),
        (projected.phase_offsets, changes[1][0]),
    ):
        # EN uses the original five neighbors, so zero form remains zero.
        received = list(ZERO)
        for target in (0, 1):
            neighbors = [
                j if i == target else i
                for i, j, _ in geometry.weighted_edges
                if target in (i, j)
            ]
            received[target] = (1 - factors["EN_mix"]) * ZERO[target] + factors[
                "EN_mix"
            ] * sum(ZERO[j] for j in neighbors) / len(neighbors)
        assert tuple(received) == ZERO
        successor = tuple(
            I(value) + (change if i == 0 else I(0)) for i, value in enumerate(incoming)
        )
        assert all(successor[i] == I(incoming[i]) for i in range(1, 45))
        assert successor[0].lo > 0
        _, before_energy, _ = _fields(geometry, tuple(I(x) for x in incoming))
        _, after_energy, after_angles = _fields(geometry, successor)
        work = after_energy - before_energy
        assert work.hi < 0
        assert all(angle.abs_max < pi_interval().lo / 2 for angle in after_angles)
        # Only IL has nonzero work: EN changes no form here; SHA changes no
        # form/phase, and this storage has no independent capacity term.
    difference = [I(0) for _ in range(45)]
    difference[0] = delta
    assert all(value == I(0) for value in difference[1:])
    # Averaging the unchanged hidden pair cancels its +/-b exactly, leaving
    # only Delta at singleton target zero in the projected primary error.


def test_fresh_postword_rows_keep_the_actual_capacity_and_both_dissipative_terms(
    preparation, increments
):
    offsets, _, projected, fine, _, factors = preparation
    geometry = fine.unit_capacity_reference.geometry
    capacity = (factors["SHA_vf_factor"],) * 2 + UNIT[2:]
    for incoming, change in (
        (offsets, increments[1][0][0]),
        (projected.phase_offsets, increments[1][1][0]),
    ):
        successor = tuple(
            I(value) + (change if i == 0 else I(0)) for i, value in enumerate(incoming)
        )
        current, _, _ = _fields(geometry, successor)
        reference = SimpleNamespace(
            geometry=geometry, form_gradient=ZERO, phase_current_bounds=current
        )
        rows = _organized_source_capacity_rows(
            reference=reference, mobilities=capacity, phase_mobility="shared_capacity"
        )
        assert rows["phase_tau_rate_bounds"] == (I(0),) * 45
        assert any(not value.contains(0) for value in rows["form_tau_rate_bounds"])
        for i in range(45):
            assert (
                rows["form_tau_rate_bounds"][i]
                - capacity[i] * geometry.gamma_bounds * current[i] / 4
            ).contains(0)
        assert rows["dissipation_tau"] == 0
        assert rows["balance_residual_bounds"].contains(0)
    # Zero immediate phase rate and zero initial dissipation do not make the
    # fresh pressure or the whole continuation stationary.


def test_complete_continuation_conserves_inverse_capacity_means_and_projection(
    preparation,
):
    s = pytest.importorskip("sympy")
    geometry = preparation[3].unit_capacity_reference.geometry
    kappa = preparation[5]["SHA_vf_factor"]
    matrix = s.zeros(45, 90)
    for column, (left, right, _) in enumerate(geometry.weighted_edges):
        matrix[left, column], matrix[right, column] = 1, -1
    forms = s.Matrix(s.symbols("form0:45", real=True))
    edge_currents = s.Matrix(s.symbols("current0:90", real=True))
    gamma = s.Symbol("gamma", positive=True)
    mobility = s.diag(s.Rational(kappa), s.Rational(kappa), *((1,) * 43))
    rho = s.Matrix((1 / s.Rational(kappa),) * 2 + (1,) * 43)
    gradient, current = matrix * matrix.T * forms, matrix * edge_currents
    form_row = mobility * (-gradient + gamma * current) / 4
    phase_row = gamma * mobility * gradient / 4
    assert s.expand(rho.dot(form_row)) == 0
    assert s.expand(rho.dot(phase_row)) == 0
    power = gradient.dot(form_row) - current.dot(phase_row)
    assert s.expand(power + gradient.dot(mobility * gradient) / 4) == 0
    exchange = s.zeros(45)
    for node, target in enumerate(SWAP):
        exchange[target, node] = 1
    projection = (s.eye(45) + exchange) / 2
    assert projection**2 == projection
    assert rho.T * projection == rho.T
    assert projection * mobility == mobility * projection
    assert projection * s.ones(45, 1) == s.ones(45, 1)
    assert sum(rho) == 43 + 2 / s.Rational(kappa)
    # This identity applies to Pi of the actual fine solution. It never
    # assumes the lossy projected coordinates form a closed autonomous law.


def test_word_created_mean_yields_an_all_time_raw_tracking_lower_bound(
    preparation, increments
):
    kappa = preparation[5]["SHA_vf_factor"]
    delta = increments[2]
    rho = (1 / kappa,) * 2 + UNIT[2:]
    total = sum(rho)
    defect = tuple(delta if i == 0 else I(0) for i in range(45))
    charge_error = sum((rho[i] * defect[i] for i in range(45)), I(0))
    mean_error = charge_error / total
    expected = delta / (43 * kappa + 2)
    assert (mean_error - expected).contains(0)
    assert mean_error.hi < 0 and expected.hi < 0
    positive_floor = -expected.hi
    assert positive_floor > PHI * HIDDEN**2 / 9000 > Q(1, 2**61)
    assert all(weight > 0 for weight in rho)
    assert sum(weight / total for weight in rho) == 1
    # For any future raw projected error e, positivity gives
    # |sum rho_i*e_i / total| <= max_i |e_i|. The preceding exact full-row
    # conservation makes its nonzero left side equal to this initial mean.
    # Thus every common elapsed continuation time has raw error >=positive_floor.


def test_strict_unfitted_memory_floor_follows_from_the_actual_phasor_factors(
    preparation, increments
):
    factors = preparation[5]
    r = 1 + 2 * cos(2 * pi_interval() / 9)
    cp, sp, cb = cos(I(PHI)), sin(I(PHI)), cos(I(HIDDEN))
    assert r.lo > 2
    fine_modulus_squared = (r + 2 * cp * cb) ** 2 + (2 * sp * cb) ** 2
    core_modulus_squared = (r + 2 * cp) ** 2 + (2 * sp) ** 2
    assert fine_modulus_squared.hi < 25 and core_modulus_squared.hi < 25
    one_minus_cosine = 2 * sin(I(HIDDEN / 2)) ** 2
    assert one_minus_cosine.lo > HIDDEN**2 / 4
    assert sp.lo > PHI / 2
    # sin(zeta)=2*r*(1-cos b)*sin(phi)/(|Zf|*|Zc|) and 0<sin(zeta)<zeta.
    angle_floor = 2 * 2 * (HIDDEN**2 / 4) * (PHI / 2) / 25
    assert angle_floor == PHI * HIDDEN**2 / 50
    assert (-increments[2] / factors["phase_coefficient"]).lo > angle_floor
    assert factors["phase_coefficient"] > Q(1, 4)
    assert 0 < factors["SHA_vf_factor"] < 1
    assert angle_floor / (4 * 45) == PHI * HIDDEN**2 / 9000 > Q(1, 2**61)


@pytest.mark.parametrize("zero_coordinate", ("hidden", "offset"))
def test_degenerate_phasors_remove_the_obstruction_without_changing_its_scope(
    zero_coordinate,
):
    phi = Q(0) if zero_coordinate == "offset" else PHI
    hidden = Q(0) if zero_coordinate == "hidden" else HIDDEN
    r = 1 + 2 * cos(2 * pi_interval() / 9)
    cb = cos(I(hidden))
    cross_real = r**2 + 2 * r * cos(I(phi)) * (1 + cb) + 4 * cb
    cross_imaginary = -4 * r * sin(I(hidden / 2)) ** 2 * sin(I(phi))
    assert cross_real.lo > 0 and cross_imaginary.contains(0)
    assert arg(cross_real, cross_imaginary).contains(0)
    s = pytest.importorskip("sympy")
    symbolic_r, symbolic_b, symbolic_phi = s.symbols("r b phi", real=True)
    expression = -4 * symbolic_r * s.sin(symbolic_b / 2) ** 2 * s.sin(symbolic_phi)
    zero_symbol = symbolic_phi if zero_coordinate == "offset" else symbolic_b
    assert expression.subs(zero_symbol, 0) == 0
    # These are algebraic zero controls, not a new source search or campaign.


def test_common_phase_quotient_discards_the_conserved_memory_coordinate():
    s = pytest.importorskip("sympy")
    relative = s.eye(45) - s.ones(45) / 45
    assert relative * s.ones(45, 1) == s.zeros(45, 1)
    # A common-phase-invariant observation can remove an arbitrary common
    # offset. This does not prove the remaining error tends to zero or that a
    # scalar mean alone suffices to predict the next structural word.
