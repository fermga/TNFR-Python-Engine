"""Exact inverse observations checked against independent fine-node derivatives."""

import json
from fractions import Fraction as Q
from math import factorial

import pytest

from tnfr.physics.relational_sine_scale import observe_sine_pair_cancellation
from tnfr.sdk import export_to_json, relational_report_to_dict

PAIRS = tuple((2 * a, 2 * a + 1) for a in range(5))
EDGES = tuple((i, j) for a in range(5) for i in PAIRS[a] for j in PAIRS[(a + 1) % 5])
NEIGHBORS = tuple(
    tuple(j if i == node else i for i, j in EDGES if node in (i, j))
    for node in range(10)
)
ZERO = (Q(0), Q(0))


def _add(left, right):
    return left[0] + right[0], left[1] + right[1]


def _scale(value, factor):
    return factor * value[0], factor * value[1]


def _mul(left, right):
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _mean(left, right):
    return _scale(_add(left, right), Q(1, 2))


def _fine_evidence(forms, phasors, pair_index):
    """Observe derivatives from the original twenty rows in tau=t/pi.

    Differentiate each fine phasor twice before taking its pair mean. No
    global-state report, quotient rate or proposed inverse supplies evidence.
    """
    forms = tuple(Q(value) for value in forms)
    phasors = tuple(tuple(Q(part) for part in value) for value in phasors)
    dx = tuple(
        sum(
            (phasors[i][0] * phasors[j][1] - phasors[i][1] * phasors[j][0])
            for j in NEIGHBORS[i]
        )
        / 4
        for i in range(10)
    )
    omega = tuple(
        sum((forms[i] - forms[j] for j in NEIGHBORS[i]), Q(0)) / 4 for i in range(10)
    )
    alpha = tuple(
        sum((dx[i] - dx[j] for j in NEIGHBORS[i]), Q(0)) / 4 for i in range(10)
    )
    first = tuple(_mul((Q(0), omega[i]), phasors[i]) for i in range(10))
    second = tuple(_mul((-omega[i] ** 2, alpha[i]), phasors[i]) for i in range(10))
    i, j = PAIRS[pair_index]
    return {
        "form_means": tuple((forms[a] + forms[b]) / 2 for a, b in PAIRS),
        "resultants": tuple(_mean(phasors[a], phasors[b]) for a, b in PAIRS),
        "pair_index": pair_index,
        "resultant_first_tau_derivative": _mean(first[i], first[j]),
        "resultant_second_tau_derivative": _mean(second[i], second[j]),
    }


def _moving_state(pair_index=0, half_difference=Q(2, 7)):
    forms = tuple(
        value
        for a in range(5)
        for value in (Q(a * a, 11) + Q(a, 19), Q(a * a, 11) - Q(a, 19))
    )
    phasors = (
        (1, 0),
        (1, 0),
        (0, 1),
        (1, 0),
        (Q(3, 5), Q(4, 5)),
        (Q(4, 5), Q(-3, 5)),
        (0, -1),
        (0, -1),
        (Q(-3, 5), Q(4, 5)),
        (1, 0),
    )
    i, j = PAIRS[pair_index]
    mean = Q(pair_index * pair_index, 11)
    forms = (
        forms[:i] + (mean + half_difference, mean - half_difference) + forms[j + 1 :]
    )
    phasors = (
        phasors[:i] + ((Q(3, 5), Q(4, 5)), (Q(-3, 5), Q(-4, 5))) + phasors[j + 1 :]
    )
    return forms, phasors


def _fine_derivative_jets(forms, phasors, order=4):
    """Exact local derivatives by Cartesian convolution of the fine ODE.

    These finite Taylor coefficients are an algebraic control, not a time
    step or a trajectory enclosure. Coefficients retain the 1/k! convention
    until the final selected-pair observations are formed.
    """
    x = [[Q(value)] for value in forms]
    z = [[tuple(Q(part) for part in value)] for value in phasors]
    for k in range(order):
        next_x, next_z = [], []
        for i, neighbors in enumerate(NEIGHBORS):
            b = tuple(
                tuple(sum((z[j][r][c] for j in neighbors), Q(0)) / 4 for c in (0, 1))
                for r in range(k + 1)
            )
            h = tuple(
                x[i][r] - sum((x[j][r] for j in neighbors), Q(0)) / 4
                for r in range(k + 1)
            )
            dx_coefficient = sum(
                (_mul((z[i][r][0], -z[i][r][1]), b[k - r])[1] for r in range(k + 1)),
                Q(0),
            )
            hz_coefficient = tuple(
                sum((h[r] * z[i][k - r][c] for r in range(k + 1)), Q(0)) for c in (0, 1)
            )
            next_x.append(dx_coefficient / (k + 1))
            next_z.append(_scale((-hz_coefficient[1], hz_coefficient[0]), Q(1, k + 1)))
        for i in range(10):
            x[i].append(next_x[i])
            z[i].append(next_z[i])
    resultants = tuple(
        _scale(_mean(z[0][k], z[1][k]), factorial(k)) for k in range(order + 1)
    )
    neighborhood = tuple(
        tuple(
            factorial(k) * sum((z[j][k][c] for j in NEIGHBORS[0]), Q(0)) / 4
            for c in (0, 1)
        )
        for k in range(order + 1)
    )
    return resultants, neighborhood


def _delayed_state(kind, orientation):
    forms = [Q(0)] * 10
    phasors = [orientation, _scale(orientation, -1)] + [(1, 0)] * 6 + [(-1, 0)] * 2
    if kind == "first_neighborhood_derivative":
        forms[2] = forms[3] = Q(2)
    elif kind == "second_neighborhood_derivative":
        forms[2], forms[3] = Q(1), Q(-1)
    elif kind == "initial_alignment":
        forms[2] = forms[3] = Q(2)
        phasors[8] = phasors[9] = (1, 0)
    return forms, phasors


@pytest.mark.parametrize("pair_index", range(5))
def test_moving_antipodal_pair_is_recovered_from_independent_fine_derivatives(
    pair_index,
):
    forms, phasors = _moving_state(pair_index)
    evidence = _fine_evidence(forms, phasors, pair_index)
    observed = observe_sine_pair_cancellation(**evidence)
    i, j = PAIRS[pair_index]
    internal_form = (forms[i] - forms[j]) / 2
    assert observed.phase_product == _mul(phasors[i], phasors[j])
    assert observed.internal_form_squared == internal_form**2
    assert observed.form_phase_moment == _scale(
        _add(phasors[i], _scale(phasors[j], -1)), internal_form / 2
    )
    assert observed.form_contrast != 0
    assert observed.phase_product_reconstruction_order == 1
    assert observed.unavailable_reason is None
    assert observed.clock == "tau=t/pi"
    assert (
        observed.resultant_second_tau_derivative
        == evidence["resultant_second_tau_derivative"]
    )


def test_second_derivative_distinguishes_the_same_instantaneous_and_first_observations():
    forms = (Q(0),) * 10
    real = ((1, 0), (-1, 0)) + ((1, 0),) * 8
    imaginary = ((0, 1), (0, -1)) + real[2:]
    left_evidence = _fine_evidence(forms, real, 0)
    right_evidence = _fine_evidence(forms, imaginary, 0)
    for name in ("form_means", "resultants", "resultant_first_tau_derivative"):
        assert left_evidence[name] == right_evidence[name]
    assert left_evidence["resultant_second_tau_derivative"] == ZERO
    assert right_evidence["resultant_second_tau_derivative"] == (1, 0)
    left = observe_sine_pair_cancellation(**left_evidence)
    right = observe_sine_pair_cancellation(**right_evidence)
    assert left.phase_product == _mul(real[0], real[1]) == (-1, 0)
    assert right.phase_product == _mul(imaginary[0], imaginary[1]) == (1, 0)
    for observation in (left, right):
        assert observation.form_phase_moment == ZERO
        assert observation.internal_form_squared == 0
        assert observation.phase_product_reconstruction_order == 2
        assert observation.unavailable_reason is None


def test_two_zero_derivatives_do_not_identify_the_phase_product_or_its_future_visibility():
    forms = tuple(Q(a * a, 11) for a in range(5) for _ in (0, 1))
    phasors = ((1, 0), (-1, 0)) * 5
    changed = ((0, 1), (0, -1)) + phasors[2:]
    evidence = _fine_evidence(forms, phasors, 0)
    assert evidence == _fine_evidence(forms, changed, 0)
    assert _mul(phasors[0], phasors[1]) != _mul(changed[0], changed[1])
    observed = observe_sine_pair_cancellation(**evidence)
    assert observed.form_contrast != 0
    assert observed.neighbor_resultant == ZERO
    assert observed.internal_form_squared == 0
    assert observed.form_phase_moment == ZERO
    assert observed.phase_product is None
    assert observed.phase_product_reconstruction_order is None
    assert (
        observed.unavailable_reason
        == "higher_order_or_persistent_environment_evidence_required"
    )


def test_first_derivative_can_recover_orientation_even_when_neighbor_resultant_vanishes():
    forms = (Q(3, 7), Q(-1, 7)) + (Q(0),) * 8
    phasors = ((1, 0), (-1, 0)) * 5
    observed = observe_sine_pair_cancellation(**_fine_evidence(forms, phasors, 0))
    assert observed.neighbor_resultant == ZERO
    assert observed.form_contrast != 0
    assert observed.phase_product == (-1, 0)
    assert observed.internal_form_squared == Q(4, 49)
    assert observed.phase_product_reconstruction_order == 1


@pytest.mark.parametrize("orientation", ((1, 0), (0, 1), (Q(3, 5), Q(4, 5))))
@pytest.mark.parametrize(
    "kind,order,first_neighborhood",
    (
        ("first_neighborhood_derivative", 1, (0, 1)),
        ("second_neighborhood_derivative", 2, (Q(-1, 2), 0)),
    ),
)
def test_delayed_neighborhood_derivatives_reveal_orientation_beyond_the_two_jet(
    orientation, kind, order, first_neighborhood
):
    forms, phasors = _delayed_state(kind, orientation)
    evidence = _fine_evidence(forms, phasors, 0)
    observed = observe_sine_pair_cancellation(**evidence)
    assert observed.phase_product is None
    resultants, neighborhood = _fine_derivative_jets(forms, phasors)
    assert resultants[1] == evidence["resultant_first_tau_derivative"]
    assert resultants[2] == evidence["resultant_second_tau_derivative"]
    assert neighborhood[:order] == (ZERO,) * order
    assert neighborhood[order] == first_neighborhood
    assert resultants[: order + 2] == (ZERO,) * (order + 2)
    phase_product = _mul(phasors[0], phasors[1])
    b = neighborhood[order]
    predicted = _mean(b, _mul(phase_product, (b[0], -b[1])))
    assert resultants[order + 2] == predicted
    numerator = _add(_scale(resultants[order + 2], 2), _scale(b, -1))
    recovered = _scale(_mul(numerator, b), 1 / (b[0] ** 2 + b[1] ** 2))
    assert recovered == phase_product


def test_initial_alignment_and_zero_acceleration_do_not_establish_persistent_silence():
    forms, phasors = _delayed_state("initial_alignment", (1, 0))
    evidence = _fine_evidence(forms, phasors, 0)
    observed = observe_sine_pair_cancellation(**evidence)
    assert observed.phase_product == (-1, 0)
    assert observed.phase_product_reconstruction_order == 2
    resultants, neighborhood = _fine_derivative_jets(forms, phasors)
    assert neighborhood[:2] == ((1, 0), (0, 1))
    assert resultants[:3] == (ZERO,) * 3
    assert resultants[3] == (0, 2)


@pytest.mark.parametrize("half_difference", (Q(2, 7), Q(0)))
def test_common_circular_rotation_and_form_origin_are_covariant(half_difference):
    forms, phasors = _moving_state(half_difference=half_difference)
    rotation, shift = (Q(3, 5), Q(4, 5)), Q(7, 13)
    original = observe_sine_pair_cancellation(**_fine_evidence(forms, phasors, 0))
    moved = observe_sine_pair_cancellation(
        **_fine_evidence(
            tuple(value + shift for value in forms),
            tuple(_mul(rotation, value) for value in phasors),
            0,
        )
    )
    for name in (
        "neighbor_resultant",
        "form_phase_moment",
        "resultant_first_tau_derivative",
        "resultant_second_tau_derivative",
    ):
        assert getattr(moved, name) == _mul(rotation, getattr(original, name))
    assert moved.phase_product == _mul(_mul(rotation, rotation), original.phase_product)
    assert moved.form_contrast == original.form_contrast
    assert moved.internal_form_squared == original.internal_form_squared
    assert moved.form_means == tuple(value + shift for value in original.form_means)


def test_tiny_exact_evidence_is_not_rounded_into_the_unavailable_branch():
    tiny = Q(1, 2**1100)
    forms, phasors = _moving_state(half_difference=tiny)
    observed = observe_sine_pair_cancellation(**_fine_evidence(forms, phasors, 0))
    assert observed.internal_form_squared == tiny**2 > 0
    assert observed.form_phase_moment == (3 * tiny / 5, 4 * tiny / 5)
    assert observed.phase_product == _mul(phasors[0], phasors[1])
    assert observed.phase_product_reconstruction_order == 1


def test_second_derivative_is_checked_even_when_the_first_identifies_the_state():
    evidence = _fine_evidence(*_moving_state(), 0)
    evidence["resultant_second_tau_derivative"] = _add(
        evidence["resultant_second_tau_derivative"], (Q(1, 2**1100), Q(0))
    )
    with pytest.raises(ValueError, match="second derivative is incompatible"):
        observe_sine_pair_cancellation(**evidence)


@pytest.mark.parametrize("acceleration", ((2, 0), (1, Q(1, 2**1100))))
def test_second_derivative_must_reconstruct_an_exact_unit_phase_product(acceleration):
    evidence = _fine_evidence((0,) * 10, ((1, 0), (-1, 0)) + ((1, 0),) * 8, 0)
    evidence["resultant_second_tau_derivative"] = acceleration
    with pytest.raises(ValueError, match="exact unit norm"):
        observe_sine_pair_cancellation(**evidence)


def test_zero_first_derivative_and_zero_neighbor_mean_require_zero_second_derivative():
    evidence = _fine_evidence((0,) * 10, ((1, 0), (-1, 0)) * 5, 0)
    evidence["resultant_second_tau_derivative"] = (Q(1, 2**1100), 0)
    with pytest.raises(ValueError, match="require zero second derivative"):
        observe_sine_pair_cancellation(**evidence)


@pytest.mark.parametrize("pair_index", (True, False, -1, 5, 0.0, "0"))
def test_pair_index_requires_a_nonboolean_integer_in_the_fixed_cycle(pair_index):
    evidence = _fine_evidence(*_moving_state(), 0)
    evidence["pair_index"] = pair_index
    with pytest.raises(ValueError, match="nonboolean pair index"):
        observe_sine_pair_cancellation(**evidence)


@pytest.mark.parametrize("row", range(5))
def test_every_supplied_mean_phasor_must_be_in_the_unit_disk(row):
    evidence = _fine_evidence(*_moving_state(), 0)
    resultants = list(evidence["resultants"])
    resultants[row] = (Q(1), Q(1, 2**1100))
    evidence["resultants"] = resultants
    with pytest.raises(ValueError, match="outside the unit disk"):
        observe_sine_pair_cancellation(**evidence)


def test_selected_resultant_must_vanish_exactly():
    evidence = _fine_evidence(*_moving_state(), 0)
    evidence["resultants"] = ((Q(1, 2**1100), 0),) + evidence["resultants"][1:]
    with pytest.raises(ValueError, match="exactly zero resultant"):
        observe_sine_pair_cancellation(**evidence)


@pytest.mark.parametrize("bad", (True, float("nan"), float("inf"), 1j, "0"))
@pytest.mark.parametrize(
    "field",
    (
        "form_means",
        "resultants",
        "resultant_first_tau_derivative",
        "resultant_second_tau_derivative",
    ),
)
def test_all_primitive_evidence_is_readmitted_as_physical_real_scalars(field, bad):
    evidence = _fine_evidence(*_moving_state(), 0)
    values = list(evidence[field])
    values[0] = (bad, 0) if field == "resultants" else bad
    evidence[field] = values
    with pytest.raises((TypeError, ValueError)):
        observe_sine_pair_cancellation(**evidence)


@pytest.mark.parametrize(
    "field",
    (
        "form_means",
        "resultants",
        "resultant_first_tau_derivative",
        "resultant_second_tau_derivative",
    ),
)
def test_shapes_and_order_are_required_but_ordered_generators_are_supported(field):
    evidence = _fine_evidence(*_moving_state(), 0)
    expected = observe_sine_pair_cancellation(**evidence)
    values = evidence[field]
    for invalid in (values[:-1], values + values[:1], set(values)):
        evidence[field] = invalid
        with pytest.raises((TypeError, ValueError)):
            observe_sine_pair_cancellation(**evidence)
    evidence[field] = iter(values)
    assert observe_sine_pair_cancellation(**evidence) == expected


@pytest.mark.parametrize(
    "field",
    (
        "resultant_first_tau_derivative",
        "resultant_second_tau_derivative",
    ),
)
def test_both_derivatives_are_required_supplied_evidence(field):
    evidence = _fine_evidence(*_moving_state(), 0)
    evidence.pop(field)
    with pytest.raises(TypeError, match=field):
        observe_sine_pair_cancellation(**evidence)


@pytest.mark.parametrize("available", (True, False))
def test_sdk_projection_preserves_exact_evidence_and_unavailable_phase_product(
    tmp_path, available
):
    forms, phasors = (
        _moving_state() if available else ((Q(1, 7),) * 10, ((1, 0), (-1, 0)) * 5)
    )
    observation = observe_sine_pair_cancellation(**_fine_evidence(forms, phasors, 0))
    direct = observation.to_dict()
    assert direct["schema"] == "tnfr.sine-pair-cancellation-observation.v1"
    assert relational_report_to_dict(observation)["report"] == direct["report"]
    value = observation.internal_form_squared
    assert direct["report"]["internal_form_squared"] == {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }
    if not available:
        assert direct["report"]["phase_product"] is None
        assert direct["report"]["phase_product_reconstruction_order"] is None
    target = tmp_path / "pair-cancellation.json"
    export_to_json(direct, target)
    assert json.loads(target.read_text()) == direct
