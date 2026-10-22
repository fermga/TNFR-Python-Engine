"""Detached neighborhood proof, with retained input and live-policy controls.

The frozen pre-Reception coordinates are only the center of a supplied
neighborhood. These tests execute no operator, response producer or flow.
An ideal real word and a conditional rounding envelope do not certify every
backend's floating-point execution or select an autonomous event occurrence.
"""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import networkx as nx
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.config.thresholds import DNFR_RECEPTION_MAX, EPI_SATURATION_MAX
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions import Coherence, Reception, Silence
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.grammar_debt import reset_grammar_state_from_history
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.preconditions import OperatorPreconditionError, validate_silence
from tnfr.operators.preconditions.coherence import validate_coherence_strict
from tnfr.operators.preconditions.reception import validate_reception_strict
from tnfr.physics import _organized_source_word as owner
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
)
from tnfr.types import Glyph
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/assets/organized_source_coherence_handoff"


@pytest.fixture(scope="module", autouse=True)
def no_response_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("A word-domain proof may not execute operators or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (word_execution, "execute_network_operator_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (Reception, "_execute"),
            (Coherence, "_execute"),
            (Silence, "_execute"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(BASE / name, max_bytes=2**27))
    )


@pytest.fixture(scope="module")
def evidence():
    paths = tuple(sorted(path for path in BASE.iterdir() if path.is_file()))
    receipts = tuple(file_receipt(path) for path in paths)
    records = {
        name: _read(name)
        for name in ("protocol.json", "freeze.json", "attempt.json", "response.json")
    }
    yield records
    assert paths == tuple(sorted(path for path in BASE.iterdir() if path.is_file()))
    assert receipts == tuple(file_receipt(path) for path in paths)


@pytest.fixture(scope="module")
def center(evidence):
    stage = evidence["response.json"]["stages"][0]
    assert stage["glyph"] == "EN"
    assert stage["preparation"] == "reception_handoff"
    return deepcopy(stage["before"])


def _field(snapshot, name):
    return tuple(row[name] for row in snapshot["nodes"])


def test_input_has_a_frozen_source_association_without_replaying_it(evidence, center):
    freeze, attempt = evidence["freeze.json"], evidence["attempt.json"]
    assert freeze["protocol"] == file_receipt(
        BASE / "protocol.json", label="protocol.json"
    )
    assert attempt["freeze"] == file_receipt(BASE / "freeze.json", label="freeze.json")
    assert freeze["archive"] == file_receipt(BASE / "source.zip", label="source.zip")
    verify_archive_members(BASE / "source.zip", freeze["members"])
    protocol = evidence["protocol.json"]
    assert center["weighted_edges"] == protocol["support"]["weighted_edges"]
    assert (
        _field(center, "form") == protocol["preparations"]["reception_handoff"]["form"]
    )
    assert center["time"] == protocol["configuration"]["structural_time_tau"]
    assert tuple(row["node"] for row in center["nodes"]) == tuple(range(45))
    assert set(_field(center, "kind")) == {"organized-source"}
    assert set(_field(center, "history")) == {("AL", "IL")}
    assert set(_field(center, "ordinal")) == {2}
    assert not any(_field(center, "debt"))
    assert all(_field(center, "prior_coherence"))


def _policy_graph(center):
    """Construct only consumed policy state; never apply a public operator."""
    graph = nx.Graph()
    graph.graph.update(
        IL_PRECONDITIONS={"activity_policy": "event_domain"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
    )
    for row in center["nodes"]:
        graph.add_node(
            row["node"],
            EPI=float(row["form"]),
            theta=float(row["phase"]),
            nu_f=float(row["capacity"]),
            delta_nfr=float(row["pressure"]),
            glyph_history=list(row["history"]),
        )
        reset_grammar_state_from_history(graph.nodes[row["node"]])
    graph.add_weighted_edges_from(
        (a, b, float(weight)) for a, b, weight in center["weighted_edges"]
    )
    return graph


def test_initialized_word_and_selected_live_policy_admit_both_signed_targets(center):
    sequence = ValidatedSequence(
        (Reception(), Coherence(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    assert sequence.names == ("reception", "coherence", "silence")
    graph = _policy_graph(center)
    before = deepcopy(graph)
    for target in (0, 1):
        validate_reception_strict(graph, target)
        validate_coherence_strict(graph, target, emit_warnings=False)
        validate_silence(graph, target)
    assert nx.utils.graphs_equal(graph, before)
    # This policy distinction survives the neighborhood proof.
    graph.graph["IL_PRECONDITIONS"] = {"activity_policy": "signed_threshold"}
    with pytest.raises(ValueError, match="signed_threshold"):
        validate_coherence_strict(graph, 1, emit_warnings=False)


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("EPI", 0.9, "EPI"),
        ("delta_nfr", 0.15, "DNFR"),
        ("EPI", True, "real"),
        ("delta_nfr", float("nan"), "finite"),
    ),
)
def test_reception_policy_boundaries_remain_independent(center, field, value, match):
    graph = _policy_graph(center)
    graph.nodes[0][field] = value
    with pytest.raises((ValueError, TypeError), match=match):
        validate_reception_strict(graph, 0)


def test_silence_floor_is_inclusive_and_not_a_phase_or_form_test(center):
    graph = _policy_graph(center)
    graph.nodes[0]["nu_f"] = 0.01
    graph.nodes[0]["EPI"] = -0.5
    validate_silence(graph, 0)
    graph.nodes[0]["nu_f"] = 0.009
    with pytest.raises(OperatorPreconditionError, match="minimal"):
        validate_silence(graph, 0)


def _mp_context():
    mp = mpmath.mp.clone()
    mp.dps = 80

    def value(number):
        return mp.mpf(number.numerator) / number.denominator

    return mp, value


@pytest.fixture(scope="module")
def inputs(evidence, center):
    config = evidence["protocol.json"]["configuration"]
    for glyph, factor, key in (
        (Glyph.EN, "EN_mix", "reception_mix"),
        (Glyph.IL, "IL_dnfr_factor", "pressure_retention"),
    ):
        assert (
            Q(resolve_runtime_operator_factors(None, glyph, {})[factor]) == config[key]
        )
    return dict(
        form=_field(center, "form"),
        phases=_field(center, "phase"),
        phase_turns=(0,) * 45,
        radius=Q(1, 100000),
        reception_mix=config["reception_mix"],
        phase_coefficient=config["phase_locking_coefficient"],
        pressure_retention=config["pressure_retention"],
        silence_capacity=Q(
            resolve_runtime_operator_factors(None, Glyph.SHA, {})["SHA_vf_factor"]
        ),
        reception_epi_ceiling=Q(EPI_SATURATION_MAX),
        reception_pressure_ceiling=Q(DNFR_RECEPTION_MAX),
        silence_min_capacity=Q(0.01),
    )


@pytest.fixture(scope="module")
def reports(inputs):
    before = deepcopy(inputs)
    ideal = owner._assess_organized_source_word_domain(**inputs)
    conditional = owner._assess_organized_source_word_domain(
        **inputs,
        reception_form_error=Q(1, 2**46),
        coherence_phase_error=Q(1, 2**46),
        pressure_error=Q(1, 2**46),
    )
    assert inputs == before
    return ideal, conditional


def test_independent_full_coordinate_box_and_declared_error_budget_are_admitted(
    inputs, reports
):
    ideal, conditional = reports
    for report in reports:
        assert report.admitted
        assert report.status == "certified_conditional_word_domain"
        assert not report.unresolved_conditions
        assert report.retained_windings == (1,) * 5
        assert report.radius > 0
        assert report.form_radius >= report.radius
        assert report.phase_radius >= report.radius
        assert 0 < report.excess_upper_bound < Q(7, 1000)
        assert report.form_absolute_upper_bound < 1
        assert report.spectral_gap_lower_bound > 0
        assert min(report.resultant_real_lower_bounds) > 4
        assert report.final_capacities == (
            inputs["silence_capacity"],
            inputs["silence_capacity"],
            *((Q(1),) * 43),
        )
        assert (
            "defect_budgets_are_hypotheses_not_uniform_libm_or_backend_certificates"
            in (report.scope)
        )
    assert ideal.reception_form_error == ideal.coherence_phase_error == 0
    assert conditional.form_radius > ideal.form_radius
    assert conditional.phase_radius > ideal.phase_radius
    assert conditional.excess_upper_bound > ideal.excess_upper_bound


def _ideal_word(inputs, center, form_offsets, phase_offsets, *, ex=Q(0), et=Q(0)):
    """Independent real formulas at a static input; no runtime proposal calls."""
    mp, value = _mp_context()
    form = tuple(x + dx for x, dx in zip(inputs["form"], form_offsets))
    theta = tuple(
        value(p + dp) + 2 * mp.pi * k
        for p, dp, k in zip(inputs["phases"], phase_offsets, inputs["phase_turns"])
    )
    beta, lam = inputs["reception_mix"], value(inputs["phase_coefficient"])
    output_form, output_phase = list(form), list(theta)
    for target in (0, 1):
        peers = center["neighbors"][target]
        output_form[target] = (
            (1 - beta) * form[target]
            + beta * sum((form[j] for j in peers), Q(0)) / len(peers)
            + ex * (1 if target == 0 else -1)
        )
        z = sum((mp.exp(1j * (theta[j] - theta[target])) for j in peers), mp.mpc(0))
        output_phase[target] = (
            theta[target] + lam * mp.arg(z) + value(et) * (1 if target == 0 else -1)
        )
    storage = sum(
        (
            value(weight)
            * (
                value(output_form[b] - output_form[a]) ** 2 / 2
                + 1
                - mp.cos(output_phase[b] - output_phase[a])
            )
            for a, b, weight in center["weighted_edges"]
        ),
        mp.mpf(0),
    )
    excess = storage - 45 * (1 - mp.cos(2 * mp.pi / 9))
    weights = tuple(
        1 / inputs["silence_capacity"] if n in (0, 1) else Q(1) for n in range(45)
    )
    mean = sum((w * x for w, x in zip(weights, output_form)), Q(0)) / sum(weights)
    return mp, value, tuple(output_form), tuple(output_phase), excess, mean


def test_center_real_composition_matches_independent_high_precision_formula(
    inputs, center, reports
):
    ideal = reports[0]
    mp, value, form, phases, excess, mean = _ideal_word(
        inputs, center, (Q(0),) * 45, (Q(0),) * 45
    )
    assert form == ideal.center_form_after_reception
    assert mean == ideal.center_weighted_form_mean == 0
    for exact, interval in zip(phases, ideal.center_phase_lift_after_coherence_bounds):
        assert value(interval.lo) <= exact <= value(interval.hi)
    assert value(ideal.center_excess_bounds.lo) <= excess
    assert excess <= value(ideal.center_excess_bounds.hi)
    for target, real, imaginary in zip(
        (0, 1), ideal.resultant_real_bounds, ideal.resultant_imaginary_bounds
    ):
        z = sum(
            (
                mp.exp(
                    1j
                    * (
                        value(inputs["phases"][neighbor])
                        - value(inputs["phases"][target])
                    )
                )
                for neighbor in center["neighbors"][target]
            ),
            mp.mpc(0),
        )
        assert value(real.lo) <= z.real <= value(real.hi)
        assert value(imaginary.lo) <= z.imag <= value(imaginary.hi)


@pytest.mark.parametrize("pattern", ("mean_shift", "alternating", "asymmetric"))
@pytest.mark.parametrize("with_defects", (False, True))
def test_independent_asymmetric_states_fit_the_uniform_word_bound(
    inputs, center, reports, pattern, with_defects
):
    r = inputs["radius"]
    if pattern == "mean_shift":
        dx = (r / 2,) * 45
        dt = tuple(r / 3 if n % 3 else -r for n in range(45))
    elif pattern == "alternating":
        dx = tuple(r if n % 2 else -r for n in range(45))
        dt = tuple(-r if n % 3 else r for n in range(45))
    else:
        dx = tuple(r if n == 2 else -r if n == 3 else Q(0) for n in range(45))
        dt = tuple(r if n == 9 else Q(0) for n in range(45))
    budget = Q(1, 2**46) if with_defects else Q(0)
    report = reports[int(with_defects)]
    mp, value, form, phases, excess, mean = _ideal_word(
        inputs, center, dx, dt, ex=budget, et=budget
    )
    assert all(
        abs(actual - central) <= report.form_radius
        for actual, central in zip(form, report.center_form_after_reception)
    )
    for exact, central in zip(phases, report.center_phase_lift_after_coherence_bounds):
        assert value(central.lo - report.phase_radius) <= exact
        assert exact <= value(central.hi + report.phase_radius)
    assert abs(mean) <= report.weighted_form_mean_absolute_bound
    assert excess < value(report.excess_upper_bound)
    assert excess > 0
    assert max(map(abs, map(value, form))) < value(report.form_absolute_upper_bound)
    for a, b, _ in center["weighted_edges"]:
        turn = int(a // 9 == b // 9 and a % 9 == 8 and b % 9 == 0)
        assert abs(phases[b] - phases[a] + 2 * mp.pi * turn) < mp.pi / 2
    if pattern == "asymmetric":
        # An ordinary zero mean is not the future weighted mean after the word.
        assert sum(inputs["form"]) + sum(dx) == 0
        expected = (
            inputs["reception_mix"] * r / (5 * (43 * inputs["silence_capacity"] + 2))
        )
        assert mean == expected != 0
    elif pattern == "mean_shift":
        assert mean == r / 2


@pytest.mark.parametrize(
    "parameter,value",
    (
        ("radius", Q(0)),
        ("radius", -Q(1, 1000)),
        ("radius", True),
        ("phase_coefficient", float("nan")),
        ("phase_coefficient", Q(1001, 1000)),
        ("phase_coefficient", Q(1, 3)),
        ("reception_mix", Q(0)),
        ("reception_mix", Q(1001, 1000)),
        ("reception_mix", Q(1, 3)),
        ("pressure_retention", Q(1)),
        ("pressure_retention", Q(1, 3)),
        ("silence_capacity", Q(0)),
        ("silence_capacity", Q(1)),
        ("silence_capacity", Q(1, 3)),
        ("silence_capacity", True),
        ("reception_form_error", -Q(1, 2**46)),
        ("coherence_phase_error", float("inf")),
        ("pressure_error", True),
    ),
)
def test_invalid_inputs_and_named_factor_endpoints_reject(inputs, parameter, value):
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_source_word_domain(**{**inputs, parameter: value})


@pytest.mark.parametrize("coordinate", ("form", "phases", "phase_turns"))
def test_boolean_or_wrong_shape_primitive_vectors_cannot_enter_the_proof(
    inputs, coordinate
):
    mutated = list(inputs[coordinate])
    mutated[7] = True
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_source_word_domain(**{**inputs, coordinate: mutated})
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_source_word_domain(
            **{**inputs, coordinate: inputs[coordinate][:-1]}
        )


@pytest.mark.parametrize(
    "change,condition",
    (
        ({"radius": Q(1, 1000)}, "successor_energy_ceiling_unavailable"),
        ({"radius": Q(1)}, "resultant_chamber_unavailable"),
        (
            {"reception_epi_ceiling": Q(-1)},
            "reception_signed_form_admission_unavailable",
        ),
        (
            {"reception_pressure_ceiling": Q(-1)},
            "reception_signed_pressure_admission_unavailable",
        ),
        (
            {"silence_min_capacity": Q(2)},
            "silence_initial_capacity_admission_unavailable",
        ),
    ),
)
def test_valid_but_uncertified_domains_remain_unavailable(inputs, change, condition):
    report = owner._assess_organized_source_word_domain(**{**inputs, **change})
    assert not report.admitted
    assert report.status == "unavailable"
    assert report.retained_windings is None
    assert condition in report.unresolved_conditions


@pytest.mark.parametrize(
    "field",
    ("reception_epi_ceiling", "reception_pressure_ceiling", "silence_min_capacity"),
)
def test_policy_threshold_cannot_round_into_a_different_inequality(
    inputs, field, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Policy representation must be admitted before geometry")

    monkeypatch.setattr(owner, "_observe_organized_source_phase_state", forbidden)
    source = Q(1, 2) + Q(1, 2**60)
    assert Q(float(source)) == Q(1, 2) < source
    with pytest.raises(ValueError, match="represented"):
        owner._assess_organized_source_word_domain(**{**inputs, field: source})


def test_asymmetric_center_retains_its_exact_nonzero_successor_mean(inputs, center):
    delta = Q(1, 1000000)
    form, phases = list(inputs["form"]), list(inputs["phases"])
    form[2] += delta
    form[3] -= delta
    phases[9] += delta
    assert sum(form) == 0
    mutated = {**inputs, "form": form, "phases": phases}
    report = owner._assess_organized_source_word_domain(**mutated)
    assert report.admitted
    expected = (
        inputs["reception_mix"] * delta / (5 * (43 * inputs["silence_capacity"] + 2))
    )
    assert report.center_weighted_form_mean == expected != 0
    assert report.source.form == tuple(form)
    assert report.source.phases == tuple(phases)
    _, _, _, _, _, oracle_mean = _ideal_word(
        mutated, center, (Q(0),) * 45, (Q(0),) * 45
    )
    assert oracle_mean == expected


def test_uniform_form_shift_changes_the_retained_mean_not_storage(inputs, reports):
    shift = Q(1, 8)
    shifted = owner._assess_organized_source_word_domain(
        **{**inputs, "form": tuple(x + shift for x in inputs["form"])}
    )
    original = reports[0]
    assert shifted.admitted
    assert shifted.center_weighted_form_mean == shift
    assert shifted.center_excess_bounds == original.center_excess_bounds
    assert shifted.excess_upper_bound == original.excess_upper_bound
    assert shifted.source.pressure_tau_bounds == original.source.pressure_tau_bounds
    assert (
        shifted.form_absolute_upper_bound == original.form_absolute_upper_bound + shift
    )
    assert shifted.center_form_after_reception == tuple(
        x + shift for x in original.center_form_after_reception
    )


def test_large_common_lift_turn_cancels_before_geometric_interval_construction(
    inputs, reports
):
    shifted = owner._assess_organized_source_word_domain(
        **{**inputs, "phase_turns": (10**80,) * 45}
    )
    original = reports[0]
    assert shifted.admitted
    for field in (
        "initial_edge_angle_bounds",
        "final_edge_angle_bounds",
        "center_excess_bounds",
        "excess_upper_bound",
        "phase_radius",
        "form_absolute_upper_bound",
        "retained_windings",
    ):
        assert getattr(shifted, field) == getattr(original, field)
    assert shifted.center_phase_lift_after_coherence_bounds != (
        original.center_phase_lift_after_coherence_bounds
    )


def test_inconsistent_local_turn_is_unavailable_without_phase_or_work_invention(inputs):
    turns = list(inputs["phase_turns"])
    turns[0] = 1
    report = owner._assess_organized_source_word_domain(
        **{**inputs, "phase_turns": turns}
    )
    assert not report.admitted
    assert "initial_acute_cell_unavailable" in report.unresolved_conditions
    assert report.center_phase_lift_after_coherence_bounds is None
    assert report.center_excess_bounds is None
    assert report.excess_upper_bound is None
    assert report.form_absolute_upper_bound is None


def test_large_finite_phase_defect_has_no_energy_or_rail_certificate(inputs):
    report = owner._assess_organized_source_word_domain(
        **inputs, coherence_phase_error=Q(2)
    )
    assert not report.admitted
    assert "successor_acute_cell_unavailable" in report.unresolved_conditions
    assert report.center_phase_lift_after_coherence_bounds is not None
    assert report.final_edge_angle_bounds is not None
    assert report.center_excess_bounds is None
    assert report.excess_upper_bound is None
    assert report.centered_form_squared_upper_bound is None
    assert report.form_absolute_upper_bound is None


@pytest.mark.parametrize("with_defects", (False, True))
def test_broad_analytic_storage_bound_is_independent_of_the_interval_reader(
    evidence, center, inputs, with_defects
):
    """Check the symbolic coarse bound against primitive support/preparation."""
    mp, value = _mp_context()
    preparation = evidence["protocol.json"]["preparations"]["reception_handoff"]
    offsets = preparation["phase_offsets"]
    h, a, epsilon = Q(1, 64), Q(1, 100), Q(1, 2**48)
    # Independently reconstruct the mathematical carrier rather than rounding
    # the interval reader's midpoint or borrowing its excess certificate.
    for node, phase in enumerate(inputs["phases"]):
        intended = 2 * mp.pi * (node % 9 - 4) / 9 + value(offsets[node])
        assert abs(value(phase) - intended) < value(epsilon)
    offset_edge_squares = sum(
        (
            weight * (offsets[j] - offsets[i]) ** 2
            for i, j, weight in center["weighted_edges"]
        ),
        Q(0),
    )
    assert offset_edge_squares == 20 * a**2
    initial_form_cost = sum(
        (
            weight * (inputs["form"][j] - inputs["form"][i]) ** 2 / 2
            for i, j, weight in center["weighted_edges"]
        ),
        Q(0),
    )
    assert initial_form_cost == 6 * h**2 == Q(3, 2048)
    beta = inputs["reception_mix"]
    contraction = 1 - 6 * beta / 5
    assert 0 < contraction < 1
    contracted_form_cost = 6 * (h * contraction) ** 2
    assert contracted_form_cost < initial_form_cost
    ex = et = Q(1, 2**46) if with_defects else Q(0)
    form_allowance = inputs["radius"] + ex
    phase_allowance = 2 * (inputs["radius"] + epsilon) + et
    reference_resultant = 2 * mp.cos(2 * mp.pi / 9) + 2 * mp.cos(value(a)) + 1
    resultant_margin = reference_resultant - 10 * value(inputs["radius"] + epsilon)
    assert resultant_margin > 0
    phase_lipschitz = 1 + 10 * value(inputs["phase_coefficient"]) / resultant_margin
    assert phase_lipschitz < 2
    assert 2 * mp.pi / 9 + 2 * value(a + phase_allowance) < mp.pi / 2
    broad_bound = (
        Q(3, 2048)
        + Q(1, 1000)
        + 360 * h * form_allowance
        + 180 * form_allowance**2
        + 180 * phase_allowance
    )
    assert broad_bound < Q(7, 1000)
    assert Q(1, 1000) == offset_edge_squares / 2
